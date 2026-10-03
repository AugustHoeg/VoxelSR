"""
InfinityTransformer3D - A volumetric super-resolution model based on Infinity
"""
import math

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from models.MaskTransformer3D import FeedForward, QKNorm, RMSNorm, AdaNorm, modulate, param_count
from models.models_3D import PixelUnshuffle3D

from models.rope import Rope3D

from models.RQTransformer3DPrefix import BodyTransformer


# ── Main model ────────────────────────────────────────────────────────────────

class InfinityTransformer3D(nn.Module):
    """
    Prefix-conditioned infinity-style transformer for 3D volumetric super-resolution.

    LR tokens are prefixed to the HR sequence (prefix-LM attention)
    3D axial RoPE is used as pos embeddings.

    Args:
        patch_nums:       token resolution schedule
        embed_dim:        shared hidden dim for transformer blocks.
        codebook_bits:    no of codebook bits for quantization
        depth:            no of transformer layers.
        num_heads:        attention heads (shared).
        mlp_ratio:        FFN hidden-dim multiplier.
        dropout:          dropout rate.
        lr_input_len:     LR token count at encoder resolution.
        lr_input_dim:     channel dim of incoming LR embeddings.
        lr_down_factor:   extra PixelUnshuffle3D downsample of the LR grid before prefixing.
        rope_theta:       RoPE base frequency (DVAR default 10000).
        rope_norm_coeffs: per-axis (x,y,z) coordinate scale for RoPE frequencies.
        use_checkpoint:   gradient checkpointing.
    """

    def __init__(
        self,
        patch_nums=(1, 2, 3, 4, 5, 6, 7, 8),
        embed_dim=512,
        codebook_bits=30,
        depth=16,
        num_heads=16,
        mlp_ratio=4.,
        drop_rate=0.,
        drop_path_rate=0.,
        lr_input_len=None,
        lr_input_dim=None,
        lr_down_factor=1,
        rope_theta=10000,
        rope_norm_coeffs=(1.0, 1.0, 1.0),
        use_checkpoint=False,
    ):
        super().__init__()

        # set hyperparameters
        self.patch_nums = patch_nums
        self.embed_dim = embed_dim
        self.codebook_bits = codebook_bits
        self.depth = depth
        self.num_heads = num_heads
        self.n_embed = n_embed
        self.lr_input_len = lr_input_len
        self.head_dim = embed_dim // num_heads
        self.mlp_dim = int(embed_dim * mlp_ratio)
        self.drop_rate = drop_rate
        self.use_checkpoint = use_checkpoint

        word_norm = False  # corresponds to nm0 in infinity

        self.C = embed_dim
        self.L = sum(pn**2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        init_std = math.sqrt(1 / self.C / 3)

        # input (word) embedding
        norm_layer = nn.LayerNorm(eps=1e-6)
        self.norm0_ve = norm_layer(self.codebook_bits) if word_norm else nn.Identity()
        self.word_embed = nn.Linear(self.codebook_bits, self.C)

        # class embedding - skipped for now

        # text embedding - skipped for now

        # AdaLN conditioning is a learned constant (pure prefix → no LR-derived cond).
        self.uncond_emb = nn.Embedding(1, embed_dim)

        # position embedding
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)

        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)

        # HR grid geometry (for RoPE coords).
        self.hr_shape = tuple(int(round(self.L ** (1. / 3))) for _ in range(3))
        self.lr_shape = tuple(int(round(lr_input_len ** (1. / 3))) for _ in range(3)) if lr_input_len else None

        # LR conditioning as a prefix.
        self.n_lr = lr_input_len // lr_down_factor ** 3
        self.lr_down = PixelUnshuffle3D(lr_down_factor) if lr_down_factor > 1 else nn.Identity()
        lr_in_dim = lr_input_dim * lr_down_factor ** 3 if lr_input_dim is not None else embed_dim
        self.lr_proj = nn.Conv3d(lr_in_dim, embed_dim, kernel_size=1, bias=False)
        self.prefix_shape = tuple(s // lr_down_factor for s in self.lr_shape) if self.lr_shape else None

        # LR prefix coords are mapped onto the HR grid inside compute_axial_cis.
        self.rope = Rope3D(
            self.head_dim, hr_shape=self.hr_shape,
            lr_shape=(self.prefix_shape if self.n_lr > 0 else None),
            theta=rope_theta, norm_coeffs=rope_norm_coeffs,
        )

        # Causal VAR-style mask with prefix-LM.
        self.compute_prefix_mask()  # TODO should match VAR-style mask (within-scale bidirectional, cross-scale causal) with prefix tokens available to all tokens.

        # backbone and head
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # dpr means drop path rate (linearly increasing)

        # prefix-LM spatial transformer backbone  TODO: Need new version of BodyTransformer where we parse dpr
        self.blocks = BodyTransformer(
            dim=self.embed_dim,
            depth=self.depth,
            heads=self.num_heads,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            use_checkpoint=self.use_checkpoint,
        )

        # head
        self.head_nm = AdaNorm(x_dim=embed_dim, y_dim=embed_dim)
        self.head = nn.Linear(self.C, self.codebook_bits)

        self._init_weights()  # TODO Needs new version of init_weights
    
    def compute_prefix_mask(self):
        
        T = self.n_lr + self.seq_len
        mask = torch.zeros(T, T, dtype=torch.bool)
        hr_causal = torch.tril(torch.ones(self.seq_len, self.seq_len, dtype=torch.bool))
        if self.n_lr > 0:
            mask[:self.n_lr, :self.n_lr] = True
            mask[self.n_lr:, :self.n_lr] = True
        mask[self.n_lr:, self.n_lr:] = hr_causal
        self.register_buffer("body_mask", mask, persistent=False)

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic)

        for tok_emb in self.tok_embs:
            nn.init.normal_(tok_emb.weight, std=0.02)
        nn.init.normal_(self.depth_emb.weight, std=0.02)
        nn.init.normal_(self.spatial_sos,      std=0.02)
        nn.init.normal_(self.uncond_emb.weight, std=0.02)

        # DiT-style zero-init: AdaLN starts as identity in both body and head.
        for block in self.body_transformer.layers:
            nn.init.constant_(block.adaln_mlp[1].weight, 0)
            nn.init.constant_(block.adaln_mlp[1].bias,   0)
        for block in self.head_transformer.layers:
            nn.init.constant_(block.adaln_mlp[1].weight, 0)
            nn.init.constant_(block.adaln_mlp[1].bias,   0)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _lr_prefix(self, lr_tokens: torch.Tensor) -> torch.Tensor:
        """lr_tokens: (B, C_lr, Dz, Dy, Dx) → prefix tokens (B, N_lr, E). RoPE handles position."""
        lr_tokens = self.lr_down(lr_tokens)
        b = lr_tokens.shape[0]
        return self.lr_proj(lr_tokens).view(b, self.embed_dim, self.n_lr).transpose(1, 2)

    # ── Forward ─────────────────

    def forward(self, x_BLC_wo_prefix: torch.Tensor, lr_tokens: torch.Tensor):
        """Prefix-LM transformer over [ LR_prefix | HR ]; returns logits BLV.

        Things to still implement:
        - How to get SOS tokens?
        - How to add lvl embedding to x_BLC?
        - How to andle condition for AdaLN modulation?
        - Get attn_mask for transformer blocks

        How VARSR does it:
        - treat LR prefix as sos, add cond from class embedding after sos and finally HR tokens [LR_prefix | SOS | HR]

        Returns:
            logits BLV
        """
        x_BLC_wo_prefix = x_BLC_wo_prefix.float()  # input should be float32
        B = x_BLC_wo_prefix.shape[0]

        # Prefix LR tokens
        sos = self._lr_prefix(lr_tokens)  # (B, N_lr, E) TODO: should we get this from lr input, text, or just learned parameter?
        sos = sos.unsqueeze(1).expand(B, 1, -1) + self.pos_start.expand(B, 1, -1)

        # Add SOS token, embed/norm on x_BLC_wo_prefix
        x_BLC = torch.cat((sos, self.word_embed(self.norm0_ve(x_BLC_wo_prefix))), dim=1)

        # TODO: add lvl embedding to x_BLC

        # Condition for AdaLN modulation
        cond = self.uncond_emb(torch.zeros(B, dtype=torch.long, device=device))

        
        x_BLC = torch.cat([lr_prefix, x_BLC], dim=1)              # (B, N_lr + SOS + L, E)

        # Block loop
        x_BLC = self.blocks(x_BLC, cond, attn_mask=self.attn_mask, rope=self.rope, rope_offset=0)
        x_BLC = x_BLC[:, self.n_lr:, :]  # HR portion

        return x_BLC

    def _depth_input(self, tok_stack: torch.Tensor, code_vectors: torch.Tensor = None) -> torch.Tensor:
        """Per-depth token stream fed to the head (see RQTransformer3D._depth_input)."""
        if not self.head_emb_vqvae:
            return tok_stack
        assert code_vectors is not None, "code_vectors must be provided when head_emb_vqvae=True"
        if self.cumsum_depth_ctx:
            code_vectors = code_vectors.cumsum(dim=2)
        return self.head_mlp(code_vectors)

    def _head_forward(self, spatial_ctx: torch.Tensor, depth_input: torch.Tensor):
        """Causal depth transformer, teacher-forced with depth_input[:, :, :-1, :]."""
        B, L, D, _ = depth_input.shape
        depth_emb = self.depth_emb(torch.arange(D, device=spatial_ctx.device))  # (D, E)

        sos = spatial_ctx.view(B, L, 1, -1)
        depth_ctx = torch.cat([sos, depth_input[:, :, :-1, :]], dim=2) + depth_emb
        depth_ctx = depth_ctx.reshape(B * L, D, -1)

        cond_head = spatial_ctx.reshape(B * L, -1)
        head_out = self.head_transformer(depth_ctx, cond_head, attn_mask=self.causal_depth_mask)
        head_out = self.head_norm(head_out).reshape(B, L, D, -1)
        return [self.heads[d](head_out[:, :, d, :]) for d in range(D)]

    def forward(self, codes: torch.Tensor, lr_tokens: torch.Tensor = None,
                code_vectors: torch.Tensor = None):
        """
        Args:
            codes:        (B, dz, dy, dx, D) int64
            lr_tokens:    (B, C_lr, Dz, Dy, Dx) pre-encoded LR embeddings, or None.
            code_vectors: (B, L, D, input_embed_dim) — iff head_emb_vqvae=True.
        Returns:
            logits: list of D tensors, each (B, L, n_embed + 1).
        """
        B, dz, dy, dx, D = codes.shape
        L = dz * dy * dx
        assert L == self.seq_len and D == self.n_rq_depth
        codes_flat = codes.reshape(B, L, D)

        spatial_ctx, tok_stack = self._body_forward(codes_flat, lr_tokens)
        depth_input = self._depth_input(tok_stack, code_vectors)
        return self._head_forward(spatial_ctx, depth_input)

    # ── Autoregressive sampling ───────────────────────────────────────────────

    def _set_body_kv_cache(self, enable: bool):
        """Toggle (and clear) the KV cache on every body attention layer."""
        for block in self.body_transformer.layers:
            block.attn.kv_caching(enable)

    def _body_run(self, body_in, cond, attn_mask, rope_offset):
        """One body pass; returns the last token's normalised spatial_ctx (B, E).

        With KV caching enabled, `body_in` is only the *new* token(s) for this step;
        the attention layers append their K/V to the per-layer cache. `rope_offset`
        must be the absolute start position of `body_in` in the [LR | HR] sequence.
        """
        out = self.body_transformer(body_in, cond, attn_mask=attn_mask,
                                    rope=self.rope, rope_offset=rope_offset)
        return self.body_norm(out[:, -1:, :], cond)[:, 0, :]        # (B, E)

    def _sample_depths(self, spatial_ctx_s, codes_flat, s, depth_emb,
                       temperature, top_k, code_emb_fn):
        """Causal depth AR for spatial position s; fills codes_flat[:, s, :] in place."""
        B, L, D = codes_flat.shape
        V = self.n_embed
        cond_head = spatial_ctx_s                                          # (B, E)
        head_input = spatial_ctx_s.unsqueeze(1) + depth_emb[0:1]           # (B, 1, E)

        for d in range(D):
            attn_mask = self.causal_depth_mask[:d + 1, :d + 1]
            h = self.head_transformer(head_input, cond_head, attn_mask=attn_mask)
            h = self.head_norm(h)
            logits = self.heads[d](h[:, -1, :])[:, :V] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float('-inf'))
            sampled = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(-1)
            codes_flat[:, s, d] = sampled

            if d + 1 < D:
                if self.head_emb_vqvae:
                    cv_s = code_emb_fn(codes_flat[:, s:s + 1, :])[:, 0, :, :]  # (B, D, input_embed_dim)
                    vec = cv_s[:, :d + 1, :].sum(dim=1) if self.cumsum_depth_ctx else cv_s[:, d, :]
                    new_tok = self.head_mlp(vec).unsqueeze(1) + depth_emb[d + 1:d + 2]
                else:
                    new_tok = self.tok_embs[d](sampled).unsqueeze(1) + depth_emb[d + 1:d + 2]
                head_input = torch.cat([head_input, new_tok], dim=1)

    @torch.no_grad()
    def sample(
        self,
        lr_tokens: torch.Tensor = None,
        batch_size: int = 1,
        temperature: float = 1.0,
        top_k: int = None,
        code_emb_fn=None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Raster-scan spatial AR + causal depth AR generation.

        The HR body is causal (LR is a fixed prefix visible to all HR positions),
        so spatial_ctx[:, s, :] depends only on the LR prefix and HR codes 0..s-1.

        use_cache=True: KV-cache the body — prefill [LR_prefix | SOS] once, then feed
            a single new token per position. Body attention cost drops from O(L³) to
            O(L·(N_lr+L)). Numerically identical to use_cache=False (verified in the
            smoke test). The head is small (D positions) and left un-cached.
        use_cache=False: reference path — full body recompute at every position.
        """
        if self.head_emb_vqvae:
            assert code_emb_fn is not None, (
                "code_emb_fn is required when head_emb_vqvae=True."
            )
        device = self.spatial_sos.device
        B = lr_tokens.shape[0] if lr_tokens is not None else batch_size
        L, D = self.seq_len, self.n_rq_depth
        depth_emb = self.depth_emb(torch.arange(D, device=device))          # (D, E)
        codes_flat = torch.zeros((B, L, D), dtype=torch.long, device=device)
        cond = self.uncond_emb(torch.zeros(B, dtype=torch.long, device=device))  # (B, E)

        if not use_cache:
            self._set_body_kv_cache(False)
            for s in range(L):
                spatial_ctx, _ = self._body_forward(codes_flat, lr_tokens)
                self._sample_depths(spatial_ctx[:, s, :], codes_flat, s, depth_emb,
                                    temperature, top_k, code_emb_fn)
            return codes_flat

        # ── KV-cached body ──
        has_prefix = lr_tokens is not None
        if has_prefix:
            assert self.n_lr > 0, "model was built unconditional (lr_input_len=None)"
        self._set_body_kv_cache(True)

        # Prefill: [LR_prefix | SOS]  (or just SOS when unconditional). The LR
        # prefix K/V is computed once here and then reused for every HR step.
        sos = self.spatial_sos.expand(B, -1, -1)                       # (B, 1, E)
        if has_prefix:
            prefill_in = torch.cat([self._lr_prefix(lr_tokens), sos], dim=1)  # (B, N_lr+1, E)
            m = self.n_lr + 1
            spatial_ctx_s = self._body_run(prefill_in, cond,
                                           self.body_mask[:m, :m], rope_offset=0)
        else:
            m0 = self.n_lr
            spatial_ctx_s = self._body_run(sos, cond,
                                           self.body_mask[m0:m0 + 1, m0:m0 + 1],
                                           rope_offset=m0)

        for s in range(L):
            self._sample_depths(spatial_ctx_s, codes_flat, s, depth_emb,
                                temperature, top_k, code_emb_fn)
            if s + 1 < L:
                # Next HR body token = Σ_d tok_embs[d](codes at position s),
                # placed at absolute position N_lr + (s+1); attends to the whole cache.
                summed = torch.stack(
                    [self.tok_embs[d](codes_flat[:, s, d]) for d in range(D)], dim=1
                ).sum(dim=1)                                           # (B, E)
                spatial_ctx_s = self._body_run(summed.unsqueeze(1), cond,
                                               attn_mask=None,
                                               rope_offset=self.n_lr + s + 1)

        self._set_body_kv_cache(False)

        return codes_flat


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hr_spatial = 8
    lr_spatial = 8
    L_hr = hr_spatial ** 3
    L_lr = lr_spatial ** 3
    D = 8
    n_embed = 4096
    lr_input_dim = 1
    lr_down_factor = 2
    num_heads = 8

    model = RQTransformer3DPrefix(
        seq_len=L_hr,
        n_rq_depth=D,
        n_embed=n_embed,
        embed_dim=768, body_depth=6, head_depth=4, num_heads=num_heads,
        lr_input_len=L_lr,
        lr_input_dim=lr_input_dim,
        lr_down_factor=lr_down_factor,
        dropout=0.1,
        use_checkpoint=True,
    ).to(device)
    param_count("RQTransformer3DPrefix", model)
    print(f"prefix N_lr={model.n_lr}, body_mask={tuple(model.body_mask.shape)}, "
          f"rope rot_dim={model.rope.rot_dim}/{model.embed_dim // num_heads}")

    codes_5d = torch.randint(0, n_embed, (2, hr_spatial, hr_spatial, hr_spatial, D), device=device)
    lr_emb = torch.randn(2, lr_input_dim, lr_spatial, lr_spatial, lr_spatial, device=device)
    logits = model(codes_5d, lr_tokens=lr_emb)

    assert len(logits) == D
    assert logits[0].shape == (2, L_hr, n_embed + 1), logits[0].shape
    print(f"forward ok — logits[0]: {tuple(logits[0].shape)}")

    model.eval()  # deterministic: disable dropout for sampling / cache checks
    B = codes_5d.shape[0]

    with torch.inference_mode():
        codes = model.sample(lr_tokens=lr_emb, temperature=1.0, top_k=100, use_cache=True)
        assert codes.shape == (2, L_hr, D), codes.shape
        print(f"sample ok — codes: {tuple(codes.shape)}")

    # ── KV-cache correctness (deterministic, teacher-forced on fixed codes) ──
    # Incremental cached body must reproduce the full-recompute body exactly.
    with torch.inference_mode():
        codes_flat = codes_5d.reshape(B, L_hr, D)
        cond = model.uncond_emb(torch.zeros(B, dtype=torch.long, device=device))
        sos = model.spatial_sos.expand(B, -1, -1)

        model._set_body_kv_cache(True)
        m = model.n_lr + 1
        prefill = torch.cat([model._lr_prefix(lr_emb), sos], dim=1)
        ctx_list = [model._body_run(prefill, cond, model.body_mask[:m, :m], rope_offset=0)]
        for s in range(L_hr - 1):
            summed = torch.stack(
                [model.tok_embs[d](codes_flat[:, s, d]) for d in range(D)], dim=1
            ).sum(dim=1)
            ctx_list.append(model._body_run(summed.unsqueeze(1), cond, None,
                                            rope_offset=model.n_lr + s + 1))
        model._set_body_kv_cache(False)
        sc_cached = torch.stack(ctx_list, dim=1)                 # (B, L, E)

        sc_full, _ = model._body_forward(codes_flat, lr_emb)     # (B, L, E)
        max_err = (sc_cached - sc_full).abs().max().item()
        print(f"kv-cache body max |Δ| vs full recompute: {max_err:.2e}")
        assert torch.allclose(sc_cached, sc_full, atol=1e-4), \
            f"KV-cache body diverged from full recompute (max |Δ|={max_err:.2e})"
        print("kv-cache equivalence ok")
