"""
InfinityTransformer3D — Infinity-style bitwise multi-scale AR transformer for 3D
volumetric super-resolution, matching the BSQVAE3D tokenizer.

Next-scale autoregression (VAR / Infinity / VARSR), bitwise prediction head
(Infinity: V = 2 * codebook_bits), LR conditioning as a prefix (VARSR), and 3D
axial RoPE + learned absolute position embeddings (coexisting, as in VARSR).

  Sequence:  [ LR_prefix (n_lr) | SOS (l_0) | scale_1 .. scale_{K-1} ]
    - LR_prefix + SOS are level 0 (bidirectional, prefix-LM); scales 1..K-1 are
      levels 1..K-1 (block-causal: within-scale bidirectional, cross-scale causal).
    - word_embed consumes the continuous teacher-forcing input x_BLC_wo_prefix built
      externally by BitwiseSelfCorrection3D.flip_requant (BSC lives in the trainer).
    - Output at the SOS block predicts scale 0; output at scale-k block predicts
      scale k. Logits are (B, L_total, 2*L) → reshape (B, L_total, L, 2) for a
      per-bit 2-class CE loss (== per-bit BCE).

  Pos:   3D axial RoPE on Q/K (shared multi-scale coordinate frame) + learned
         pos_start / pos_1LC absolute embeddings + per-scale lvl_embed.
  Cond:  pure prefix — AdaLN is driven by a learned constant (uncond_emb); the only
         LR signal is the prefix tokens. CFG toggles the prefix (learned null prefix).

References:
  VAR:      https://github.com/FoundationVision/VAR/blob/main/models/var.py
  Infinity: https://github.com/FoundationVision/Infinity/blob/main/infinity/models/infinity.py
  VARSR:    https://github.com/quyp2000/VARSR/blob/master/models/var.py
"""
import math
import numpy as np

import torch
from torch import nn
import torch.utils.checkpoint as checkpoint

from models.MaskTransformer3D import AdaNorm, RMSNorm, FeedForward, modulate, param_count
from models.models_3D import PixelUnshuffle3D
from models.rope import Rope3DMultiScale
from models.RQTransformer3DPrefix import AttentionRQ


# ── Transformer backbone (DiT-AdaLN, prefix-LM self-attn, RoPE, stochastic depth) ──

class DropPath(nn.Module):
    """Per-sample stochastic depth on a residual branch (timm-style)."""

    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)          # per-sample mask
        mask = x.new_empty(shape).bernoulli_(keep)
        if keep > 0:
            mask.div_(keep)
        return x * mask

    def extra_repr(self):
        return f"drop_prob={self.drop_prob:.3f}"


class RoPEAttnBlock3D(nn.Module):
    """DiT-AdaLN body block: prefix-LM self-attention w/ 3D RoPE + per-branch DropPath.

    Same structure as RQTransformer3DPrefix.BlockRQ, extended with stochastic depth —
    Infinity plumbs a per-layer drop_path rate from a linear schedule (see
    https://github.com/FoundationVision/Infinity/blob/main/infinity/models/infinity.py#L254).
    """

    def __init__(self, dim, heads, mlp_dim, dropout=0., drop_path=0.):
        super().__init__()
        self.adaln_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.ln1 = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
        self.attn = AttentionRQ(dim, heads, dropout=dropout)
        self.ln2 = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, cond, attn_mask=None, rope=None, rope_offset=0):
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaln_mlp(cond).chunk(6, dim=1)
        x = x + self.drop_path(alpha1.unsqueeze(1) * self.attn(
            modulate(self.ln1(x), gamma1, beta1),
            attn_mask=attn_mask, rope=rope, rope_offset=rope_offset,
        ))
        x = x + self.drop_path(alpha2.unsqueeze(1) * self.ff(modulate(self.ln2(x), gamma2, beta2)))
        return x


class RoPETransformer3D(nn.Module):
    """Stack of transformer blocks with a linear drop-path-rate schedule (Infinity)."""

    def __init__(self, dim, depth, heads, mlp_dim, dropout=0., drop_path_rate=0.,
                 use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # per-layer drop-path
        self.layers = nn.ModuleList([
            RoPEAttnBlock3D(dim, heads, mlp_dim, dropout=dropout, drop_path=dpr[i])
            for i in range(depth)
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


# ── Main model ────────────────────────────────────────────────────────────────

class InfinityTransformer3D(nn.Module):
    """Prefix-conditioned, bitwise next-scale transformer for 3D volumetric SR.

    Args:
        patch_nums:       multi-scale token schedule (must match vae.quantizer.v_patch_nums).
        embed_dim:        shared hidden dim for the transformer blocks.
        codebook_bits:    L, number of BSQ bits per token (== vae.quantizer.L).
        depth:            number of transformer layers.
        num_heads:        attention heads.
        mlp_ratio:        FFN hidden-dim multiplier.
        drop_rate:        dropout rate.
        drop_path_rate:   max stochastic-depth rate; InfinityBackbone applies a linear
                          per-layer schedule linspace(0, drop_path_rate, depth).
        lr_input_len:     LR token count at encoder resolution (None = unconditional).
        lr_input_dim:     channel dim of incoming LR embeddings.
        lr_down_factor:   extra PixelUnshuffle3D downsample of the LR grid before prefixing.
        rope_theta:       RoPE base frequency.
        rope_norm_coeffs: per-axis (x,y,z) coordinate scale for RoPE frequencies.
        rope_ref_grid:    shared RoPE coordinate frame (default = finest patch_num).
        word_norm:        LayerNorm over bits before word_embed (Infinity norm0_ve); default off.
        use_checkpoint:   gradient checkpointing.
    """

    def __init__(
        self,
        patch_nums=(1, 2, 3, 4, 5, 6, 8),
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
        rope_ref_grid=None,
        word_norm=False,
        use_checkpoint=False,
    ):
        super().__init__()

        # ── hyperparameters / scale bookkeeping ──
        self.patch_nums = tuple(patch_nums)
        self.K = len(self.patch_nums)
        self.pn_dims = [(pn, pn, pn) if isinstance(pn, int) else tuple(pn) for pn in self.patch_nums]
        self.l_k = [pd * ph * pw for (pd, ph, pw) in self.pn_dims]       # tokens per scale
        self.L_total = sum(self.l_k)                                     # predicted sequence length
        self.begins = [sum(self.l_k[:k]) for k in range(self.K)]         # start of scale k in predicted region
        self.l0 = self.l_k[0]

        self.C = self.embed_dim = embed_dim
        self.codebook_bits = codebook_bits
        self.depth = depth
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.mlp_dim = int(embed_dim * mlp_ratio)
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.use_checkpoint = use_checkpoint
        init_std = math.sqrt(1 / self.C / 3)
        self.init_std = init_std

        # ── word (bit) embedding ──
        self.norm0_ve = nn.LayerNorm(codebook_bits, eps=1e-6) if word_norm else nn.Identity()
        self.word_embed = nn.Linear(codebook_bits, self.C)

        # ── conditioning (pure prefix): learned AdaLN constant ──
        self.uncond_emb = nn.Embedding(1, embed_dim)

        # ── LR conditioning as a prefix ──
        self.lr_input_len = lr_input_len
        if lr_input_len is not None:
            self.n_lr = lr_input_len // lr_down_factor ** 3
            self.lr_shape = tuple(int(round(lr_input_len ** (1. / 3))) for _ in range(3))
            self.prefix_shape = tuple(s // lr_down_factor for s in self.lr_shape)
            self.lr_down = PixelUnshuffle3D(lr_down_factor) if lr_down_factor > 1 else nn.Identity()
            lr_in_dim = lr_input_dim * lr_down_factor ** 3 if lr_input_dim is not None else embed_dim
            self.lr_proj = nn.Conv3d(lr_in_dim, embed_dim, kernel_size=1, bias=False)
            # learned null prefix for classifier-free guidance
            self.null_lr_prefix = nn.Parameter(torch.empty(1, self.n_lr, self.C))
        else:
            self.n_lr = 0
            self.lr_shape = self.prefix_shape = None

        self.context_token = self.n_lr + self.l0   # LR + SOS region (level 0)

        # ── learned absolute position embeddings (coexist with RoPE, as in VARSR) ──
        self.pos_start = nn.Parameter(torch.empty(1, self.l0, self.C))        # SOS block
        self.pos_1LC = nn.Parameter(torch.empty(1, self.L_total, self.C))     # predicted region
        self.lvl_embed = nn.Embedding(self.K, self.C)                         # per-scale level embedding

        # ── 3D axial RoPE over [ LR_prefix | scale_0 .. scale_{K-1} ] ──
        self.rope = Rope3DMultiScale(
            self.head_dim, patch_nums=self.pn_dims,
            lr_shape=(self.prefix_shape if self.n_lr > 0 else None),
            ref_grid=rope_ref_grid, theta=rope_theta, norm_coeffs=rope_norm_coeffs,
        )

        # ── VAR block-causal + prefix-LM attention mask (bool) and lvl index ──
        self.compute_attention_mask()

        # ── Transformer blocks ──
        self.blocks = RoPETransformer3D(
            dim=self.C, depth=self.depth, heads=self.num_heads, mlp_dim=self.mlp_dim,
            dropout=self.drop_rate, drop_path_rate=self.drop_path_rate,
            use_checkpoint=self.use_checkpoint,
        )

        # ── bitwise prediction head (Infinity: V = 2 * codebook_bits) ──
        self.head_nm = AdaNorm(x_dim=embed_dim, y_dim=embed_dim)
        self.head = nn.Linear(self.C, 2 * codebook_bits)

        self._init_weights()

    # ── masks ──────────────────────────────────────────────────────────────────

    def compute_attention_mask(self):
        """Bool mask (True = attention allowed), RQTransformer3DPrefix convention.

        LR_prefix + SOS at level 0 (bidirectional, prefix-LM); scales 1..K-1 at
        levels 1..K-1. Token i attends to token j iff level_i >= level_j
        (within-scale bidirectional, cross-scale causal) — == VAR's d >= dT.
        """
        T = self.n_lr + self.L_total
        levels = torch.zeros(T, dtype=torch.long)
        ptr = self.context_token                         # n_lr + l0 already at level 0
        for k in range(1, self.K):
            levels[ptr:ptr + self.l_k[k]] = k
            ptr += self.l_k[k]
        d = levels.view(T, 1)
        attn_mask = (d >= d.transpose(0, 1))             # (T, T) bool
        self.register_buffer("attn_mask", attn_mask, persistent=False)

        # per-scale level index for the predicted region (used by lvl_embed gather)
        lvl_1L = torch.cat([torch.full((l,), k, dtype=torch.long)
                            for k, l in enumerate(self.l_k)])  # (L_total,)
        self.register_buffer("lvl_1L", lvl_1L, persistent=False)

    # ── initialisation ──────────────────────────────────────────────────────────

    def _init_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=self.init_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic)

        nn.init.trunc_normal_(self.lvl_embed.weight, std=self.init_std)
        nn.init.trunc_normal_(self.uncond_emb.weight, std=self.init_std)
        nn.init.trunc_normal_(self.pos_start, std=self.init_std)
        nn.init.trunc_normal_(self.pos_1LC, std=self.init_std)
        if self.n_lr > 0:
            nn.init.trunc_normal_(self.null_lr_prefix, std=self.init_std)

        # DiT-style zero-init: AdaLN starts as identity in body blocks and the head.
        for block in self.blocks.layers:
            nn.init.constant_(block.adaln_mlp[1].weight, 0)
            nn.init.constant_(block.adaln_mlp[1].bias, 0)
        nn.init.constant_(self.head_nm.mlp[1].weight, 0)
        nn.init.constant_(self.head_nm.mlp[1].bias, 0)

    # ── helpers ──────────────────────────────────────────────────────────────────

    def _lr_prefix(self, lr_tokens: torch.Tensor) -> torch.Tensor:
        """(B, C_lr, Dz, Dy, Dx) → (B, n_lr, C). RoPE handles position."""
        lr_tokens = self.lr_down(lr_tokens)
        b = lr_tokens.shape[0]
        return self.lr_proj(lr_tokens).view(b, self.C, self.n_lr).transpose(1, 2)

    def _build_prefix(self, lr_tokens, B):
        """Prefix tokens (B, n_lr, C). lr_tokens=None → learned null prefix (CFG uncond)."""
        if self.n_lr == 0:
            return torch.empty(B, 0, self.C, device=self.pos_1LC.device, dtype=self.pos_1LC.dtype)
        if lr_tokens is None:
            return self.null_lr_prefix.expand(B, -1, -1)
        return self._lr_prefix(lr_tokens)

    def _sos_block(self, cond, B):
        """SOS = cond (uncond_emb) expanded to l0 tokens + pos_start + lvl/pos (scale 0)."""
        sos = cond.unsqueeze(1).expand(B, self.l0, -1) + self.pos_start
        return self._add_lvl_pos(sos, 0)

    def _add_lvl_pos(self, tok, si):
        """Add scale-`si` level embedding and absolute pos_1LC slice. tok: (B, l_si, C)."""
        b0, l = self.begins[si], self.l_k[si]
        lvl = self.lvl_embed(torch.full((l,), si, device=tok.device, dtype=torch.long))  # (l, C)
        return tok + lvl.unsqueeze(0) + self.pos_1LC[:, b0:b0 + l]

    def get_logits(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """(B, T, C) → (B, T, 2*L) bit logits (reshape to (B, T, L, 2) for CE/BCE)."""
        return self.head(self.head_nm(h, cond))

    # ── forward (teacher-forced) ─────────────────────────────────────────────────

    def forward(self, x_BLC_wo_prefix: torch.Tensor, lr_tokens: torch.Tensor = None):
        """
        Args:
            x_BLC_wo_prefix: (B, L_total - l0, L) continuous teacher-forcing input for
                             scales 1..K-1 (from BitwiseSelfCorrection3D.flip_requant).
            lr_tokens:       (B, C_lr, Dz, Dy, Dx) LR embeddings, or None (→ null prefix).
        Returns:
            logits: (B, L_total, 2*L)  — reshape (B, L_total, L, 2) for the per-bit loss.
        """
        x_BLC_wo_prefix = x_BLC_wo_prefix.float()
        B = x_BLC_wo_prefix.shape[0]

        # Cond (SOS) + pos embedding
        cond = self.uncond_emb(torch.zeros(B, dtype=torch.long, device=x_BLC_wo_prefix.device))
        sos = cond.unsqueeze(1).expand(B, self.l0, -1) + self.pos_start  # (B, l0, C)

        # LR token prefix
        prefix = self._build_prefix(lr_tokens, B)  # (B, n_lr, C)

        # Add HR tokens + lvl embed + pos embedding for scales 0..K-1
        x_BLC = torch.cat([sos, self.word_embed(self.norm0_ve(x_BLC_wo_prefix))], dim=1)  # (B, L_total, C)
        x_BLC = x_BLC + self.lvl_embed(self.lvl_1L).unsqueeze(0) + self.pos_1LC

        # Concatenate LR prefix + SOS + HR tokens
        x_BLC = torch.cat([prefix, x_BLC], dim=1)  # (B, n_lr + L_total, C)

        # Block loop
        x_BLC = self.blocks(x_BLC, cond, attn_mask=self.attn_mask, rope=self.rope, rope_offset=0)

        # Head
        return self.get_logits(x_BLC[:, self.n_lr:, :], cond)  # (B, L_total, 2L)

    # ── autoregressive sampling ──────────────────────────────────────────────────

    def _set_kv_cache(self, enable: bool):
        for block in self.blocks.layers:
            block.attn.kv_caching(enable)

    def _sample_bits(self, logits_BlLv, temperature, top_k=None, top_p=None):
        """logits: (B, l, L, 2) → sampled bits (B, l, L) long in {0,1}.
        (top_k / top_p are no-ops for a 2-class axis; kept for API parity.)"""
        B, l, L, _ = logits_BlLv.shape
        if temperature <= 0:
            return logits_BlLv.argmax(dim=-1)
        probs = (logits_BlLv / temperature).softmax(dim=-1).reshape(-1, 2)
        return torch.multinomial(probs, num_samples=1).reshape(B, l, L)

    @torch.no_grad()
    def autoregressive_infer(
        self,
        vae,
        lr_tokens: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
    ):
        """Next-scale bitwise AR generation WITHOUT classifier-free guidance.

        Single-branch (batch B) conditioned on the real LR prefix only — no uncond
        branch, no batch doubling, no logit mixing. Same per-scale / KV-cache structure
        as `autoregressive_infer_cfg`; `vae.quantizer` supplies `bsq.indices_to_code`
        and `get_next_autoregressive_input`. Assumes LR tokens are always provided.

        Returns:
            ms_bits: list of K tensors (B, l_k, L) bits → vae.decode_multiscale(ms_bits).
        """
        assert self.n_lr > 0, "autoregressive_infer assumes LR tokens are always provided"
        q = vae.quantizer
        B = lr_tokens.shape[0]
        device = lr_tokens.device

        cond = self.uncond_emb(torch.zeros(B, dtype=torch.long, device=device))
        lvl_pos = self.lvl_embed(self.lvl_1L).unsqueeze(0) + self.pos_1LC         # (1, L_total, C)

        # First input = [ LR_prefix | SOS ]; SOS = cond + pos_start + lvl_pos[:, :l0].
        prefix = self._lr_prefix(lr_tokens)                                      # (B, n_lr, C)
        sos = cond.unsqueeze(1).expand(B, self.l0, -1) + self.pos_start + lvl_pos[:, :self.l0]
        next_token_map = torch.cat([prefix, sos], dim=1)                         # (B, n_lr + l0, C)

        f_hat = torch.zeros(B, q.L, *self.pn_dims[-1], device=device)
        ms_bits = []
        cur_L = 0

        self._set_kv_cache(True)
        for si in range(self.K):
            pd, ph, pw = self.pn_dims[si]

            attn_mask = self.attn_mask[:self.context_token, :self.context_token] if si == 0 else None
            rope_offset = 0 if si == 0 else self.n_lr + self.begins[si]
            x = self.blocks(next_token_map, cond, attn_mask=attn_mask, rope=self.rope, rope_offset=rope_offset)

            x = x[:, -self.l_k[si]:, :]                                          # last l_si tokens → scale si
            logits_BlV = self.get_logits(x, cond).reshape(B, self.l_k[si], q.L, 2)

            bits = self._sample_bits(logits_BlV, temperature, top_k, top_p)     # (B, l_si, L)
            ms_bits.append(bits)
            code = q.bsq.indices_to_code(bits.view(B, pd, ph, pw, q.L))         # (B,pd,ph,pw,L)
            q_BLDHW = code.permute(0, 4, 1, 2, 3).contiguous()                  # (B,L,pd,ph,pw)
            f_hat, next_scale = q.get_next_autoregressive_input(si, f_hat, q_BLDHW)
            cur_L += self.l_k[si]

            if si == self.K - 1:
                break

            # Next scale's input map: cumulative latent ↓ scale si+1, embedded + lvl_pos.
            next_token_map = next_scale.reshape(B, q.L, -1).permute(0, 2, 1)    # (B, l_{si+1}, L)
            next_token_map = self.word_embed(self.norm0_ve(next_token_map)) \
                + lvl_pos[:, cur_L:cur_L + self.l_k[si + 1]]
        self._set_kv_cache(False)

        return ms_bits

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self,
        vae,
        lr_tokens: torch.Tensor,
        cfg: float = 1.5,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
    ):
        """Next-scale bitwise AR generation with classifier-free guidance (VAR-style).

        Structured after VAR.autoregressive_infer_cfg
        (https://github.com/FoundationVision/VAR/blob/main/models/var.py#L127):
        the batch is always doubled — a cond branch (real LR prefix) and an uncond
        branch (learned null prefix) — and at each scale the two logit sets are mixed
        with VAR's guidance schedule  t = cfg * si/(K-1). The body is KV-cached, so each
        step feeds only the current scale's `next_token_map`; `vae.quantizer` plays the
        role of VAR's `vae_quant_proxy` (`bsq.indices_to_code` + `get_next_autoregressive_input`).

        Assumes LR tokens are always provided (self.n_lr > 0).

        Returns:
            ms_bits: list of K tensors (B, l_k, L) bits → vae.decode_multiscale(ms_bits).
        """
        assert self.n_lr > 0, "autoregressive_infer_cfg assumes LR tokens are always provided"
        q = vae.quantizer
        B = lr_tokens.shape[0]
        device = lr_tokens.device

        # Learned AdaLN constant (shared by both CFG branches; guidance lives in the prefix).
        cond = self.uncond_emb(torch.zeros(2 * B, dtype=torch.long, device=device))

        # VAR `lvl_pos`: level + absolute position for the whole predicted region.
        lvl_pos = self.lvl_embed(self.lvl_1L).unsqueeze(0) + self.pos_1LC  # (1, L_total, C)

        # Doubled LR prefix: [ cond = real LR ; uncond = learned null LR ].
        prefix = torch.cat([self._lr_prefix(lr_tokens), self.null_lr_prefix.expand(B, -1, -1)], dim=0)  # (2B, n_lr, C)

        # First input = [ prefix | SOS ]; SOS = cond + pos_start + lvl_pos[:, :l0].
        sos = cond.unsqueeze(1).expand(2 * B, self.l0, -1) + self.pos_start + lvl_pos[:, :self.l0]
        next_token_map = torch.cat([prefix, sos], dim=1)  # (2B, n_lr + l0, C)

        f_hat = torch.zeros(B, q.L, *self.pn_dims[-1], device=device)             # VAR f_hat
        ms_bits = []
        cur_L = 0

        self._set_kv_cache(True)
        for si in range(self.K):
            pd, ph, pw = self.pn_dims[si]
            ratio = si / (self.K - 1) if self.K > 1 else 0.0

            # Step si: only the prefill needs the prefix-LM mask (rope_offset 0); later
            # scales attend to the whole cache in generation order (rope_offset = abs pos).
            attn_mask = self.attn_mask[:self.context_token, :self.context_token] if si == 0 else None
            rope_offset = 0 if si == 0 else self.n_lr + self.begins[si]
            x = self.blocks(next_token_map, cond, attn_mask=attn_mask, rope=self.rope, rope_offset=rope_offset)

            x = x[:, -self.l_k[si]:, :]                                           # last l_si tokens → scale si
            logits_BlV = self.get_logits(x, cond).reshape(2 * B, self.l_k[si], q.L, 2)

            # Classifier-free guidance (VAR schedule).
            t = cfg * ratio
            logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]           # (B, l_si, L, 2)

            # Sample bits, decode to this scale's code, advance f_hat.
            bits = self._sample_bits(logits_BlV, temperature, top_k, top_p)      # (B, l_si, L)
            ms_bits.append(bits)
            code = q.bsq.indices_to_code(bits.view(B, pd, ph, pw, q.L))          # (B,pd,ph,pw,L)
            q_BLDHW = code.permute(0, 4, 1, 2, 3).contiguous()                   # (B,L,pd,ph,pw)
            f_hat, next_scale = q.get_next_autoregressive_input(si, f_hat, q_BLDHW)
            cur_L += self.l_k[si]

            if si == self.K - 1:
                break

            # Next scale's input map: cumulative latent ↓ scale si+1, embedded + lvl_pos,
            # then doubled so both CFG branches share the same HR input (VAR repeat(2,1,1)).
            next_token_map = next_scale.reshape(B, q.L, -1).permute(0, 2, 1)     # (B, l_{si+1}, L)
            next_token_map = self.word_embed(self.norm0_ve(next_token_map)) \
                + lvl_pos[:, cur_L:cur_L + self.l_k[si + 1]]
            next_token_map = next_token_map.repeat(2, 1, 1)                      # (2B, l_{si+1}, C)
        self._set_kv_cache(False)

        return ms_bits


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch.nn.functional as F
    from models.BSQVAE3D import BSQVAE3D
    from models.bitwise_self_correction_3d import BitwiseSelfCorrection3D

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    patch_size = 64
    patch_nums = (1, 2, 3, 4, 5, 6, 8)           # finest == encoder latent (down_factor 2)
    codebook_bits = 30
    lr_spatial = 16
    lr_input_dim = 1
    lr_down_factor = 2

    vae = BSQVAE3D(
        in_channels=1,
        latent_dim=768,
        codebook_bits=codebook_bits,                 # implicit vocab 2**48
        channels_enc=[64, 64, 256, 512, 512],   # down_factor = 8 -> latent 8^3
        channels_dec=[512, 512, 256, 64, 64],
        resolution=patch_size,
        num_res_blocks_enc=2,
        num_res_blocks_dec=4,
        v_patch_nums=patch_nums,        # -> 953 tokens per volume
        use_decay_factor=True,                  # fallback if use_prog_quant_resi is False
        quant_resi=0.5,                         # learned Phi refine (only used if use_prog_quant_resi is True)
        use_prog_quant_resi=True,              # Enables progressive upsampling phi refinement
        use_stochastic_depth=True,
        scale_drop_rate=0.10,
        keep_last_quant=True,
        keep_first_quant=False,
        skip_attn=False,
        use_checkpoint=True,
    ).to(device).eval()

    model = InfinityTransformer3D(
        patch_nums=patch_nums,
        embed_dim=768,
        codebook_bits=codebook_bits,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.1,          # exercise the Infinity-style per-layer DropPath schedule
        lr_input_len=lr_spatial ** 3,
        lr_input_dim=lr_input_dim,
        lr_down_factor=lr_down_factor,
        rope_theta=10000,
        use_checkpoint=False,
    ).to(device)

    param_count("InfinityTransformer3D", model)
    print(f"n_lr={model.n_lr}, L_total={model.L_total}, attn_mask={tuple(model.attn_mask.shape)}, "
          f"rope rot_dim={model.rope.rot_dim}/{model.head_dim}")

    B = 2
    x_hr = torch.randn(B, 1, patch_size, patch_size, patch_size, device=device)
    lr_emb = torch.randn(B, lr_input_dim, lr_spatial, lr_spatial, lr_spatial, device=device)

    # ── teacher-forcing input + gt bit targets via BSC (trainer-side) ──
    bsc = BitwiseSelfCorrection3D(vae, noise_apply_layers=0, noise_apply_strength=0.0)
    raw = vae.encode(x_hr)
    x_BLC_wo_prefix, gt_ms_idx_Bl = bsc.flip_requant(raw, device)

    # ── forward + dummy per-bit loss ──
    logits = model(x_BLC_wo_prefix, lr_tokens=lr_emb)                  # (B, L_total, 2L)
    assert logits.shape == (B, model.L_total, 2 * codebook_bits), logits.shape
    targets = torch.cat(gt_ms_idx_Bl, dim=1).long()                   # (B, L_total, L)
    loss = F.cross_entropy(logits.reshape(-1, 2), targets.reshape(-1))
    print(f"forward ok - logits {tuple(logits.shape)}, bit-CE loss {loss.item():.4f}")

    # ── teacher-forced KV-cache equivalence (cached per-scale == single full forward) ──
    model.eval()  # disable dropout / stochastic depth for the deterministic comparison
    with torch.inference_mode():
        full = model(x_BLC_wo_prefix, lr_tokens=lr_emb)               # (B, L_total, 2L)

        model._set_kv_cache(True)
        condB = model.uncond_emb(torch.zeros(B, dtype=torch.long, device=device))
        prefix = model._build_prefix(lr_emb, B)
        sos = model._sos_block(condB, B)
        m = model.context_token
        out = model.blocks(torch.cat([prefix, sos], dim=1), condB,
                           attn_mask=model.attn_mask[:m, :m], rope=model.rope, rope_offset=0)
        cached = [model.get_logits(out[:, -model.l0:, :], condB)]
        tf = model.norm0_ve(x_BLC_wo_prefix.float())
        for si in range(model.K - 1):
            b0 = model.begins[si + 1] - model.l0
            blk = tf[:, b0:b0 + model.l_k[si + 1], :]
            tok = model._add_lvl_pos(model.word_embed(blk), si + 1)
            out = model.blocks(tok, condB, attn_mask=None, rope=model.rope,
                               rope_offset=model.n_lr + model.begins[si + 1])
            cached.append(model.get_logits(out, condB))
        model._set_kv_cache(False)
        cached = torch.cat(cached, dim=1)                             # (B, L_total, 2L)

        max_err = (cached - full).abs().max().item()
        print(f"kv-cache teacher-forced max abs-diff vs full forward: {max_err:.2e}")
        assert torch.allclose(cached, full, atol=1e-4), f"KV-cache diverged (max abs-diff={max_err:.2e})"
        print("kv-cache equivalence ok")

    # ── sampling (with CFG) ──
    model.eval()
    with torch.inference_mode():
        ms_bits = model.autoregressive_infer_cfg(vae, lr_tokens=lr_emb, cfg=2.0, temperature=1.0)
        assert len(ms_bits) == model.K
        for k, b in enumerate(ms_bits):
            assert b.shape == (B, model.l_k[k], codebook_bits), (k, b.shape)
        x_rec = vae.decode_multiscale(ms_bits)
        print(f"sample (cfg) ok - {len(ms_bits)} scales, decoded {tuple(x_rec.shape)}")

        ms_bits = model.autoregressive_infer(vae, lr_tokens=lr_emb, temperature=1.0)
        assert len(ms_bits) == model.K
        for k, b in enumerate(ms_bits):
            assert b.shape == (B, model.l_k[k], codebook_bits), (k, b.shape)
        x_rec = vae.decode_multiscale(ms_bits)
        print(f"sample (no-cfg) ok - {len(ms_bits)} scales, decoded {tuple(x_rec.shape)}")
