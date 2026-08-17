"""
RQTransformer3DPrefix — Two-Transformer RQ + 3D Volumetric SR, prefix-conditioned.

A "pure prefix" variant of RQTransformer3D. Instead of injecting LR context via
per-block cross-attention + AdaLN (as in RQTransformer3D.py), the LR tokens are
*prefixed* to the body sequence and attended to in-context, following VARSR / DVAR.
3D axial RoPE replaces all learned absolute positional embeddings in the body.

  Body (spatial): prefix-LM transformer over  [ LR_prefix (N_lr) | HR (L) ].
    Input:  LR tokens (projected) ++ right-shifted summed depth embeddings.
    Attn:   prefix-LM mask —
              LR ↔ LR   full (bidirectional within the prefix)
              HR → LR   full (every HR token sees the whole LR prefix)
              HR → HR   causal (raster-scan AR, unchanged from RQTransformer3D)
              LR → HR   blocked (prefix never looks ahead into HR)
    Pos:    3D axial RoPE on Q/K (no learned absolute pos embeddings).
    Cond:   pure prefix — AdaLN is driven by a learned constant (uncond_emb);
            the *only* LR conditioning signal is the prefix tokens themselves.
    Output: spatial_ctx (B, L, E)  (HR portion only)

  Head (depth): unchanged from RQTransformer3D — lightweight causal AR transformer
    per spatial position, conditioned on spatial_ctx via AdaLN, learned depth_emb.

Complexity: O((N_lr+L)²) body + O(D²·L) head.

References:
  RQTransformer: https://arxiv.org/abs/2203.01941
  VARSR:         https://github.com/quyp2000/VARSR
  DVAR:          https://github.com/YuZheng9/DVAR   (see dvar.py, rope.py)
  RoFormer:      https://arxiv.org/abs/2104.09864
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from models.MaskTransformer3D import FeedForward, QKNorm, RMSNorm, AdaNorm, modulate, param_count
from models.models_3D import PixelUnshuffle3D

from models.rope import Rope3D


# ── Shared attention module (RoPE-aware) ──────────────────────────────────────

class AttentionRQ(nn.Module):
    """Self-attention with an optional boolean mask and optional 3D RoPE on Q/K."""

    def __init__(self, embed_dim, num_heads, dropout=0., bias=False):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.n_heads  = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout  = dropout
        self.wq = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.wk = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.wv = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.wo = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.qk_norm = QKNorm(embed_dim)

        # only used during inference
        self.caching, self.cached_k, self.cached_v = False, None, None

    def kv_caching(self, enable: bool):
        self.caching, self.cached_k, self.cached_v = enable, None, None

    def forward(self, x, attn_mask=None, rope=None, rope_offset=0):
        """
        x:         (B, T, embed_dim)
        attn_mask: (T, T) bool — True where attention is allowed, or None.
        rope:      Rope3D module applied to Q/K, or None.
        """
        B, T, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq, xk = self.qk_norm(xq, xk, xv)
        xq = xq.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if rope is not None:
            xq = rope(xq, rope_offset)
            xk = rope(xk, rope_offset)

        if self.caching:
            if self.cached_k is None:
                self.cached_k = xk
                self.cached_v = xv
            else:
                xk = self.cached_k = torch.cat((self.cached_k, xk), dim=2)
                xv = self.cached_v = torch.cat((self.cached_v, xv), dim=2)

        out = F.scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


# ── Body blocks (DiT-AdaLN, prefix-LM self-attention, RoPE) ───────────────────

class BlockRQ(nn.Module):
    """DiT-AdaLN body block — prefix-LM self-attention with 3D RoPE.

    Pure prefix: no cross-attention. `cond` is a learned constant (uncond_emb),
    so AdaLN acts as a learned per-layer affine and the DiT zero-init warmup still
    applies; all LR conditioning flows through the prefix tokens instead.
    """

    def __init__(self, dim, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.adaln_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.ln1 = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
        self.attn = AttentionRQ(dim, heads, dropout=dropout)
        self.ln2 = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
        self.ff  = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x, cond, attn_mask=None, rope=None, rope_offset=0):
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaln_mlp(cond).chunk(6, dim=1)
        x = x + alpha1.unsqueeze(1) * self.attn(
            modulate(self.ln1(x), gamma1, beta1),
            attn_mask=attn_mask, rope=rope, rope_offset=rope_offset,
        )
        x = x + alpha2.unsqueeze(1) * self.ff(modulate(self.ln2(x), gamma2, beta2))
        return x


class BodyTransformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0., use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.layers = nn.ModuleList([
            BlockRQ(dim, heads, mlp_dim, dropout=dropout) for _ in range(depth)
        ])

    def forward(self, x, cond, attn_mask=None, rope=None, rope_offset=0):
        for block in self.layers:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(
                    block, x, cond, attn_mask, rope, rope_offset, use_reentrant=False
                )
            else:
                x = block(x, cond, attn_mask=attn_mask, rope=rope, rope_offset=rope_offset)
        return x


# ── Head blocks (AdaLN-conditioned, causal) — unchanged from RQTransformer3D ───

class HeadBlock(nn.Module):
    """DiT-AdaLN causal block for the depth transformer.

    spatial_ctx (B*L, E) conditions all D depth tokens via AdaLN shift/scale.
    """

    def __init__(self, dim, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.adaln_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.ln1  = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
        self.attn = AttentionRQ(dim, heads, dropout=dropout)
        self.ln2  = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
        self.ff   = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x, cond, attn_mask=None):
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaln_mlp(cond).chunk(6, dim=1)
        x = x + alpha1.unsqueeze(1) * self.attn(modulate(self.ln1(x), gamma1, beta1), attn_mask=attn_mask)
        x = x + alpha2.unsqueeze(1) * self.ff(modulate(self.ln2(x), gamma2, beta2))
        return x


class HeadTransformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0., use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.layers = nn.ModuleList([
            HeadBlock(dim, heads, mlp_dim, dropout=dropout) for _ in range(depth)
        ])

    def forward(self, x, cond, attn_mask=None):
        for block in self.layers:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x, cond, attn_mask, use_reentrant=False)
            else:
                x = block(x, cond, attn_mask=attn_mask)
        return x


# ── Main model ────────────────────────────────────────────────────────────────

class RQTransformer3DPrefix(nn.Module):
    """
    Prefix-conditioned two-transformer RQ for 3D volumetric super-resolution.

    Differs from RQTransformer3D only in how LR conditioning and positions work:
      * LR tokens are prefixed to the body sequence (prefix-LM attention), not
        cross-attended. No LR-derived AdaLN — pure prefix conditioning.
      * 3D axial RoPE on the body replaces all learned absolute pos embeddings.

    Args:
        seq_len:        HR spatial token count L = dz*dy*dx.
        n_rq_depth:     number of RQ codebook depths D.
        embed_dim:      shared hidden dim for both transformers.
        n_embed:        codebook size; mask token id = n_embed.
        body_depth:     body transformer layers.
        head_depth:     head transformer layers.
        num_heads:      attention heads (shared).
        mlp_ratio:      FFN hidden-dim multiplier.
        dropout:        dropout rate.
        lr_input_len:   LR token count at encoder resolution (None = unconditional).
        lr_input_dim:   channel dim of incoming LR embeddings.
        lr_down_factor: extra PixelUnshuffle3D downsample of the LR grid before prefixing.
        rope_theta:     RoPE base frequency (DVAR default 10000).
        rope_norm_coeffs: per-axis (x,y,z) coordinate scale for RoPE frequencies.
        use_checkpoint: gradient checkpointing.
        head_emb_vqvae / cumsum_depth_ctx / input_embed_dim: as in RQTransformer3D.
    """

    def __init__(
        self,
        seq_len,
        n_rq_depth,
        embed_dim=512,
        n_embed=1024,
        body_depth=12,
        head_depth=2,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.,
        lr_input_len=None,
        lr_input_dim=None,
        lr_down_factor=1,
        rope_theta=10000,
        rope_norm_coeffs=(1.0, 1.0, 1.0),
        use_checkpoint=False,
        head_emb_vqvae=False,
        cumsum_depth_ctx=False,
        input_embed_dim=None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_rq_depth = n_rq_depth
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.lr_input_len = lr_input_len
        head_dim = embed_dim // num_heads

        # Frozen-codebook head option (parity with RQTransformer3D).
        self.head_emb_vqvae = head_emb_vqvae
        self.cumsum_depth_ctx = bool(cumsum_depth_ctx) and head_emb_vqvae
        if head_emb_vqvae:
            assert input_embed_dim is not None, (
                "input_embed_dim (RQVAE codebook vector dim) must be set when head_emb_vqvae=True."
            )
            self.input_embed_dim = input_embed_dim
            self.head_mlp = nn.Linear(input_embed_dim, embed_dim)

        mlp_dim = int(embed_dim * mlp_ratio)

        # Per-depth token embeddings; index n_embed = [MASK] token.
        self.tok_embs = nn.ModuleList([
            nn.Embedding(n_embed + 1, embed_dim) for _ in range(n_rq_depth)
        ])
        # Depth positional embedding for the head (kept — depth is not a spatial axis).
        self.depth_emb = nn.Embedding(n_rq_depth, embed_dim)
        # Learned SOS for the causal HR body.
        self.spatial_sos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # AdaLN conditioning is a learned constant (pure prefix → no LR-derived cond).
        self.uncond_emb = nn.Embedding(1, embed_dim)

        # HR grid geometry (for RoPE coords).
        self.hr_shape = tuple(int(round(seq_len ** (1. / 3))) for _ in range(3))
        self.lr_shape = tuple(int(round(lr_input_len ** (1. / 3))) for _ in range(3)) if lr_input_len else None

        # LR conditioning as a prefix.
        if lr_input_len is not None:
            self.n_lr = lr_input_len // lr_down_factor ** 3
            self.lr_down = PixelUnshuffle3D(lr_down_factor) if lr_down_factor > 1 else nn.Identity()
            lr_in_dim = lr_input_dim * lr_down_factor ** 3 if lr_input_dim is not None else embed_dim
            self.lr_proj = nn.Conv3d(lr_in_dim, embed_dim, kernel_size=1, bias=False)
            self.prefix_shape = tuple(s // lr_down_factor for s in self.lr_shape) if self.lr_shape else None
        else:
            self.n_lr = 0
            self.prefix_shape = None

        # LR prefix coords are mapped onto the HR grid inside compute_axial_cis.
        self.rope = Rope3D(
            head_dim, hr_shape=self.hr_shape,
            lr_shape=(self.prefix_shape if self.n_lr > 0 else None),
            theta=rope_theta, norm_coeffs=rope_norm_coeffs,
        )

        # Causal mask for the body (prefix-LM). 
        self.compute_prefix_mask()

        # Body: prefix-LM spatial transformer.
        self.body_transformer = BodyTransformer(
            dim=embed_dim, depth=body_depth, heads=num_heads, mlp_dim=mlp_dim,
            dropout=dropout, use_checkpoint=use_checkpoint,
        )
        self.body_norm = AdaNorm(x_dim=embed_dim, y_dim=embed_dim)

        # Head: causal depth transformer (unchanged).
        self.head_transformer = HeadTransformer(
            dim=embed_dim, depth=head_depth, heads=num_heads, mlp_dim=mlp_dim,
            dropout=dropout, use_checkpoint=use_checkpoint,
        )
        self.head_norm = RMSNorm(embed_dim, linear=True, bias=False, eps=1e-5)

        # Per-depth prediction heads, weight-tied to token embeddings.
        self.heads = nn.ModuleList([
            nn.Linear(embed_dim, n_embed + 1, bias=False) for _ in range(n_rq_depth)
        ])
        for d in range(n_rq_depth):
            self.heads[d].weight = self.tok_embs[d].weight

        # Causal D×D mask for the head.
        self.register_buffer(
            "causal_depth_mask",
            torch.tril(torch.ones(n_rq_depth, n_rq_depth, dtype=torch.bool)),
            persistent=False,
        )

        self._init_weights()
    
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

    # ── Forward (split so sampling can re-use the body pass) ─────────────────

    def _body_forward(self, codes_flat: torch.Tensor, lr_tokens: torch.Tensor = None):
        """Prefix-LM body over [ LR_prefix | HR ]; returns HR spatial_ctx.

        Returns:
            spatial_ctx: (B, L, E) — HR portion only
            tok_stack:   (B, L, D, E) per-depth token embeddings (re-used by the head)
        """
        B, L, D = codes_flat.shape
        device = codes_flat.device

        tok_stack = torch.stack(
            [self.tok_embs[d](codes_flat[:, :, d]) for d in range(D)], dim=2  # (B, L, D, E)
        )
        xs_emb = tok_stack.sum(dim=2)                                         # (B, L, E)

        # Right-shift HR by SOS: slot 0 = SOS, slots 1..L-1 = summed codes 0..L-2.
        sos = self.spatial_sos.expand(B, -1, -1)                             # (B, 1, E)
        hr_in = torch.cat([sos, xs_emb[:, :-1, :]], dim=1)                   # (B, L, E)

        cond = self.uncond_emb(torch.zeros(B, dtype=torch.long, device=device))

        if lr_tokens is not None:
            assert self.n_lr > 0, "model was built unconditional (lr_input_len=None)"
            lr_prefix = self._lr_prefix(lr_tokens)                           # (B, N_lr, E)
            body_in = torch.cat([lr_prefix, hr_in], dim=1)                   # (B, N_lr+L, E)
            out = self.body_transformer(body_in, cond, attn_mask=self.body_mask,
                                        rope=self.rope, rope_offset=0)
            spatial_ctx = out[:, self.n_lr:, :]                              # HR portion
        else:
            # Unconditional: HR-only causal body; RoPE uses HR coords (rows n_lr:).
            out = self.body_transformer(hr_in, cond, attn_mask=self.body_mask[self.n_lr:, self.n_lr:],
                                        rope=self.rope, rope_offset=self.n_lr)
            spatial_ctx = out

        spatial_ctx = self.body_norm(spatial_ctx, cond)
        return spatial_ctx, tok_stack

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
                if s % 50 == 0:
                    print(f"AR token: {s}/{L}")
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
            if s % 50 == 0:
                print(f"AR token: {s}/{L}")
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
