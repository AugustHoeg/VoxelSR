"""
MaskRQTransformer3Dv3 — Windowed (Swin-style) Attention RQ + MaskGIT for 3D SR

Identical axial layout to v2 (interleaved spatial + depth blocks, AdaLN-Zero
conditioning, LR cross-attention, per-depth MaskGIT heads), with one change:

  v2 SpatialBlock:  global bidirectional AttentionRQ over all L spatial tokens
  v3 SpatialBlock:  windowed swin-style attention over the 3D spatial grid
                    (W-MSA / SW-MSA with a continuous 3D relative position bias)

Every spatial block reshapes (B, L, D, E) → (B*D, dz, dy, dx, E) so each RQ
depth slice is attended as an independent 3D volume, partitioned into
`window_size³` windows. Odd-indexed blocks shift the window grid by
`window_size // 2` (SW-MSA) with the corresponding cyclic-shift attention mask,
recovering cross-window connectivity exactly like Swin.

Depth blocks are unchanged from v2 — global bidirectional attention over the D
RQ depths at each spatial position.

The windowed-attention primitives (window_partition3D / window_reverse3D /
compute_mask / WindowAttention3D_FAST) are streamlined ports of the STLayerV2
implementation in FlashAttentionTest.py, with the image-space upsampling and the
PatchPositionalBias1D module removed.

References:
  RQTransformer:  https://arxiv.org/abs/2203.01941
  MaskGIT:        https://arxiv.org/abs/2202.04200
  Swin / SwinV2:  https://arxiv.org/abs/2103.14030 , https://arxiv.org/abs/2111.09883
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from models.MaskTransformer3D import (
    FeedForward, RMSNorm, AdaNorm, CrossAttention, modulate, param_count,
)
# Depth block + schedule are unchanged from v2 — reuse directly
from models.MaskRQTransformer3Dv2 import DepthBlock, _make_depth_schedule, chunked_sdpa


def _to_3tuple(x):
    return tuple(x) if isinstance(x, (tuple, list)) else (x, x, x)


# ── Windowed-attention primitives (streamlined from FlashAttentionTest.py) ─────

def window_partition3D(x, window_size):
    """(B, Hp, Wp, Dp, C) → (num_windows, window_size³, C)."""
    B, Hp, Wp, Dp, C = x.shape
    x = x.view(B, Hp // window_size, window_size,
                  Wp // window_size, window_size,
                  Dp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size ** 3, C)
    return windows


def window_reverse3D(windows, window_size, dims):
    """(num_windows, window_size³, C) → (B, H, W, D, C)."""
    B, H, W, D = dims
    x = windows.view(B, H // window_size, W // window_size, D // window_size,
                     window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, H, W, D, -1)
    return x


def compute_mask(x_dims, window_size, shift_size):
    """SW-MSA cyclic-shift attention mask, shape (num_windows, N, N) with N=window_size³.

    From "Liu et al., Swin Transformer" (https://arxiv.org/abs/2103.14030).
    """
    d, h, w = x_dims
    img_mask = torch.zeros((1, d, h, w, 1))
    cnt = 0
    for ds in (slice(-window_size), slice(-window_size, -shift_size), slice(-shift_size, None)):
        for hs in (slice(-window_size), slice(-window_size, -shift_size), slice(-shift_size, None)):
            for ws in (slice(-window_size), slice(-window_size, -shift_size), slice(-shift_size, None)):
                img_mask[:, ds, hs, ws, :] = cnt
                cnt += 1

    mask_windows = window_partition3D(img_mask, window_size).squeeze(-1)  # (nW, N)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)     # (nW, N, N)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
    return attn_mask


class WindowAttention3D(nn.Module):
    """SwinV2 window multi-head self-attention with a learned 3D relative position bias.

    Streamlined port of WindowAttention3D_FAST: fused QKV, scaled-dot-product
    attention with the relative position bias (and optional SW-MSA mask) supplied
    as an additive attention bias.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wd, Wh, Ww)
        self.num_heads = num_heads

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                num_heads,
            )
        )
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size[0]),
            torch.arange(window_size[1]),
            torch.arange(window_size[2]),
            indexing="ij",
        ))                                                     # (3, Wd, Wh, Ww)
        coords_flatten = torch.flatten(coords, 1)              # (3, N)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (N, N, 3)
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)      # (N, N)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = attn_drop

    def forward(self, x, mask=None):
        """x: (num_windows*B, N, C); mask: (num_windows, N, N) or None."""
        B_, N, C = x.shape

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))

        qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        rel_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1)
        rel_bias = rel_bias.permute(2, 0, 1).contiguous().unsqueeze(0)  # (1, num_heads, N, N)

        if mask is not None:
            nW = mask.shape[0]
            attn_bias = mask.unsqueeze(1).repeat(B_ // nW, 1, 1, 1) + rel_bias  # (B_, num_heads, N, N)
        else:
            attn_bias = rel_bias

        out = chunked_sdpa(
            q, k, v, attn_mask=attn_bias,
            dropout_p=self.attn_drop if self.training else 0.,
        )
        return out.transpose(1, 2).reshape(B_, N, C)


class WindowSelfAttention3D(nn.Module):
    """Swin W-MSA / SW-MSA wrapper over a batch of 3D volumes.

    Accepts flattened tokens (M, L, E) with L = dz*dy*dx, applies the optional
    cyclic shift, window-partitions, runs windowed attention, reverses, and
    un-shifts — returning (M, L, E). The cyclic-shift mask is precomputed.
    """

    def __init__(self, dim, num_heads, spatial_shape, window_size, shift_size=0,
                 qkv_bias=True, attn_drop=0.):
        super().__init__()
        self.dz, self.dy, self.dx = spatial_shape
        self.window_size = window_size
        self.shift_size = shift_size
        self.attn = WindowAttention3D(
            dim, window_size=_to_3tuple(window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop,
        )
        if shift_size > 0:
            attn_mask = compute_mask((self.dz, self.dy, self.dx), window_size, shift_size)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        M, L, E = x.shape
        x = x.view(M, self.dz, self.dy, self.dx, E)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(1, 2, 3))

        x_windows = window_partition3D(x, self.window_size)
        x_windows = self.attn(x_windows, mask=self.attn_mask)
        x = window_reverse3D(x_windows, self.window_size, dims=(M, self.dz, self.dy, self.dx))

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(1, 2, 3))

        return x.reshape(M, L, E)


# ── Windowed spatial block (v2 SpatialBlock with windowed attention) ───────────

class WindowedSpatialBlock(nn.Module):
    """Windowed spatial attention over the 3D grid, applied per RQ-depth slice.

    Reshapes (B, L, D, E) → (B*D, L, E) so each depth level is an independent
    volume; self-attention is windowed (Swin W-MSA / SW-MSA). AdaLN-Zero
    modulation and LR cross-attention are identical to v2 — cross-attention stays
    global over the full LR sequence, broadcast across the D depth slices.
    """

    def __init__(self, dim, heads, mlp_dim, spatial_shape, window_size, shift_size=0,
                 dropout=0., use_cross_attn=False):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.adaln_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.ln1  = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
        self.attn = WindowSelfAttention3D(dim, heads, spatial_shape, window_size,
                                          shift_size=shift_size, attn_drop=dropout)
        self.ln2  = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
        self.ff   = FeedForward(dim, mlp_dim, dropout=dropout)
        if use_cross_attn:
            self.ln_cross   = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
            self.cross_attn = CrossAttention(dim, heads, dropout=dropout)

    def forward(self, x, cond, lr_tokens=None):
        """
        x:         (B, L, D, E)
        cond:      (B, E)        — mean-pooled LR for AdaLN
        lr_tokens: (B, N_lr, E)  — full LR sequence for cross-attn, or None
        """
        B, L, D, E = x.shape

        # Merge depth into batch; each depth slice is an independent 3D volume
        x_s = x.permute(0, 2, 1, 3).reshape(B * D, L, E)              # (B*D, L, E)
        cond_s = cond.unsqueeze(1).expand(B, D, E).reshape(B * D, E)  # (B*D, E)

        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaln_mlp(cond_s).chunk(6, dim=1)
        x_s = x_s + alpha1.unsqueeze(1) * self.attn(modulate(self.ln1(x_s), gamma1, beta1))

        if self.use_cross_attn and lr_tokens is not None:
            N_lr = lr_tokens.shape[1]
            lr_s = lr_tokens.unsqueeze(1).expand(B, D, N_lr, E).reshape(B * D, N_lr, E)
            x_s  = x_s + self.cross_attn(self.ln_cross(x_s), lr_s)

        x_s = x_s + alpha2.unsqueeze(1) * self.ff(modulate(self.ln2(x_s), gamma2, beta2))

        return x_s.reshape(B, D, L, E).permute(0, 2, 1, 3)           # (B, L, D, E)


# ── Axial transformer (windowed spatial + global depth) ────────────────────────

class AxialWindowedTransformer(nn.Module):
    """Interleaves windowed SpatialBlocks and global DepthBlocks.

    Odd-indexed spatial blocks use a shifted window grid (SW-MSA); shifting is
    disabled automatically when a window already spans the full grid.
    """

    def __init__(self, dim, spatial_depth, depth_depth, heads, mlp_dim,
                 spatial_shape, window_size, dropout=0., use_checkpoint=False,
                 use_cross_attn=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        shift = window_size // 2
        can_shift = all(s > window_size for s in spatial_shape)  # no point shifting a single window

        self.spatial_blocks = nn.ModuleList([
            WindowedSpatialBlock(
                dim, heads, mlp_dim, spatial_shape, window_size,
                shift_size=(shift if (can_shift and i % 2 == 1) else 0),
                dropout=dropout, use_cross_attn=use_cross_attn,
            )
            for i in range(spatial_depth)
        ])
        self.depth_blocks = nn.ModuleList([
            DepthBlock(dim, heads, mlp_dim, dropout=dropout)
            for _ in range(depth_depth)
        ])
        self.depth_schedule = _make_depth_schedule(spatial_depth, depth_depth)

    def forward(self, x, cond, lr_tokens=None):
        """x: (B, L, D, E), cond: (B, E), lr_tokens: (B, N_lr, E) or None"""
        for i, s_block in enumerate(self.spatial_blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(s_block, x, cond, lr_tokens, use_reentrant=False)
            else:
                x = s_block(x, cond, lr_tokens=lr_tokens)

            d_idx = self.depth_schedule[i]
            if d_idx >= 0:
                d_block = self.depth_blocks[d_idx]
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(d_block, x, cond, use_reentrant=False)
                else:
                    x = d_block(x, cond)
        return x


# ── Main model ────────────────────────────────────────────────────────────────

class MaskRQTransformer3Dv3(nn.Module):
    """
    Windowed-attention RQ + MaskGIT model for 3D volumetric super-resolution.

    Same architecture as MaskRQTransformer3Dv2 except the spatial self-attention
    is windowed (Swin W-MSA / SW-MSA over the 3D grid) instead of global. Depth
    attention, AdaLN-Zero conditioning, LR cross-attention and per-depth MaskGIT
    heads are unchanged.

    Args (additions over v2):
        spatial_shape:  (dz, dy, dx) HR token grid. If None, inferred as cubic
                        from seq_len. Each dim must be divisible by window_size.
        window_size:    edge length of the cubic attention window.

    Remaining args match v2:
        seq_len:        spatial token count L = dz*dy*dx
        n_rq_depth:     number of RQ codebook depths D
        embed_dim:      shared hidden dimension
        n_embed:        codebook size; mask token index = n_embed
        body_depth:     number of windowed spatial blocks
        head_depth:     number of depth blocks interleaved among body_depth
        num_heads:      attention heads
        mlp_ratio:      FFN hidden-dim multiplier
        dropout:        dropout rate
        lr_seq_len:     LR spatial token count (None → unconditional)
        lr_embed_dim:   channel dim of incoming LR encoder embeddings
        use_checkpoint: gradient checkpointing
    """

    def __init__(
        self,
        seq_len,
        n_rq_depth,
        spatial_shape=None,
        window_size=4,
        embed_dim=512,
        n_embed=1024,
        body_depth=12,
        head_depth=2,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.,
        lr_seq_len=None,
        lr_embed_dim=None,
        use_checkpoint=False,
    ):
        super().__init__()

        if spatial_shape is None:
            side = round(seq_len ** (1 / 3))
            assert side ** 3 == seq_len, (
                f"seq_len={seq_len} is not a perfect cube; pass spatial_shape explicitly."
            )
            spatial_shape = (side, side, side)
        spatial_shape = tuple(spatial_shape)
        assert len(spatial_shape) == 3
        assert spatial_shape[0] * spatial_shape[1] * spatial_shape[2] == seq_len, (
            f"prod(spatial_shape)={spatial_shape} != seq_len={seq_len}"
        )
        for s in spatial_shape:
            assert s % window_size == 0, (
                f"spatial_shape {spatial_shape} not divisible by window_size {window_size}"
            )

        self.seq_len       = seq_len
        self.n_rq_depth    = n_rq_depth
        self.embed_dim     = embed_dim
        self.n_embed       = n_embed
        self.lr_seq_len    = lr_seq_len
        self.spatial_shape = spatial_shape
        self.window_size   = window_size

        mlp_dim = int(embed_dim * mlp_ratio)

        # Per-depth token embeddings; index n_embed = [MASK] token
        self.tok_embs = nn.ModuleList([
            nn.Embedding(n_embed + 1, embed_dim) for _ in range(n_rq_depth)
        ])

        # Independent spatial (absolute) and depth positional embeddings
        self.pos_emb   = nn.Embedding(seq_len, embed_dim)
        self.depth_emb = nn.Embedding(n_rq_depth, embed_dim)

        # LR conditioning
        if lr_seq_len is not None:
            lr_in_dim = lr_embed_dim if lr_embed_dim is not None else embed_dim
            self.lr_proj    = nn.Linear(lr_in_dim, embed_dim, bias=False)
            self.lr_pos_emb = nn.Embedding(lr_seq_len, embed_dim)
        else:
            self.uncond_emb = nn.Embedding(1, embed_dim)

        # Axial transformer: windowed spatial blocks interleaved with depth blocks
        self.axial_transformer = AxialWindowedTransformer(
            dim=embed_dim,
            spatial_depth=body_depth,
            depth_depth=head_depth,
            heads=num_heads,
            mlp_dim=mlp_dim,
            spatial_shape=spatial_shape,
            window_size=window_size,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
            use_cross_attn=lr_seq_len is not None,
        )
        self.out_norm = AdaNorm(x_dim=embed_dim, y_dim=embed_dim)

        # Per-depth prediction heads, weight-tied to token embeddings
        self.heads = nn.ModuleList([
            nn.Linear(embed_dim, n_embed + 1, bias=False) for _ in range(n_rq_depth)
        ])
        for d in range(n_rq_depth):
            self.heads[d].weight = self.tok_embs[d].weight

        self._init_weights()

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
        nn.init.normal_(self.pos_emb.weight,   std=0.02)
        nn.init.normal_(self.depth_emb.weight, std=0.02)

        if self.lr_seq_len is not None:
            nn.init.normal_(self.lr_pos_emb.weight, std=0.02)
        else:
            nn.init.normal_(self.uncond_emb.weight, std=0.02)

        # DiT-style zero-init: AdaLN outputs start as identity, cross-attn output starts at zero
        for block in self.axial_transformer.spatial_blocks:
            nn.init.constant_(block.adaln_mlp[1].weight, 0)
            nn.init.constant_(block.adaln_mlp[1].bias,   0)
            if block.use_cross_attn:
                nn.init.constant_(block.cross_attn.wo.weight, 0)

        for block in self.axial_transformer.depth_blocks:
            nn.init.constant_(block.adaln_mlp[1].weight, 0)
            nn.init.constant_(block.adaln_mlp[1].bias,   0)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _prepare_lr_context(self, lr_tokens: torch.Tensor):
        """lr_tokens: (B, N_lr, lr_embed_dim) → (lr_ctx (B,N_lr,E), cond (B,E))"""
        lr_pos = torch.arange(lr_tokens.shape[1], device=lr_tokens.device)
        lr_ctx = self.lr_proj(lr_tokens) + self.lr_pos_emb(lr_pos)
        return lr_ctx, lr_ctx.mean(dim=1)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, codes: torch.Tensor, lr_tokens: torch.Tensor = None):
        """
        Args:
            codes:     (B, dz, dy, dx, D) int64; masked positions hold self.n_embed
            lr_tokens: (B, N_lr, lr_embed_dim) or None
        Returns:
            logits: list of D tensors, each (B, L, n_embed + 1)
        """
        B, dz, dy, dx, D = codes.shape
        L = dz * dy * dx
        assert (dz, dy, dx) == self.spatial_shape, (
            f"codes grid {(dz, dy, dx)} != spatial_shape {self.spatial_shape}"
        )
        assert L == self.seq_len and D == self.n_rq_depth
        codes = codes.reshape(B, L, D)

        # Embed every (position, depth) pair independently — no summing
        tok_stack = torch.stack(
            [self.tok_embs[d](codes[:, :, d]) for d in range(D)], dim=2
        )                                                                   # (B, L, D, E)

        pos_emb   = self.pos_emb(torch.arange(L, device=codes.device))      # (L, E)
        depth_emb = self.depth_emb(torch.arange(D, device=codes.device))    # (D, E)

        # Additive spatial and depth positional encodings, broadcast over the other axis
        x = tok_stack + pos_emb[None, :, None, :] + depth_emb[None, None, :, :]  # (B, L, D, E)

        # LR conditioning
        if lr_tokens is not None:
            lr_ctx, cond = self._prepare_lr_context(lr_tokens)
        else:
            cond = self.uncond_emb(torch.zeros(B, dtype=torch.long, device=codes.device))
            lr_ctx = None

        # Axial transformer — representation stays (B, L, D, E) throughout
        x = self.axial_transformer(x, cond, lr_tokens=lr_ctx)               # (B, L, D, E)

        # Final norm: reshape L×D into one sequence dim for AdaNorm, then restore
        x = self.out_norm(x.reshape(B, L * D, self.embed_dim), cond)        # (B, L*D, E)
        x = x.reshape(B, L, D, self.embed_dim)                              # (B, L, D, E)

        # Per-depth logits
        return [self.heads[d](x[:, :, d, :]) for d in range(D)]


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / 10**9 if torch.cuda.is_available() else 0

    # print("Flash Attention:", torch.backends.cuda.flash_sdp_enabled())
    print("Mem Efficient  :", torch.backends.cuda.mem_efficient_sdp_enabled())
    # print("Math SDP       :", torch.backends.cuda.math_sdp_enabled())

    hr_spatial = 16
    lr_spatial = 16
    window_size = 8
    L_hr = hr_spatial ** 3
    L_lr = lr_spatial ** 3
    D = 4
    n_embed = 512
    lr_embed_dim = 256
    use_checkpoint = True

    configs = {
        "tiny":  dict(embed_dim=512, body_depth=3, head_depth=3, num_heads=4),
        #"small": dict(embed_dim=384, body_depth=4, head_depth=4, num_heads=6),
        #"base":  dict(embed_dim=512, body_depth=6, head_depth=6, num_heads=8),
    }

    for name, cfg in configs.items():
        L_lr = None
        lr_embed_dim = None

        model = MaskRQTransformer3Dv3(
            seq_len=L_hr, n_rq_depth=D, n_embed=n_embed,
            spatial_shape=(hr_spatial, hr_spatial, hr_spatial), window_size=window_size,
            lr_seq_len=L_lr, lr_embed_dim=lr_embed_dim, dropout=0.1, use_checkpoint=use_checkpoint, **cfg,
        ).to(device)
        param_count(f"MaskRQTransformer3Dv3-{name}", model)

        codes_5d = torch.randint(0, n_embed, (2, hr_spatial, hr_spatial, hr_spatial, D), device=device)
        # lr_emb   = torch.randn(2, L_lr, lr_embed_dim, device=device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = model(codes_5d, lr_tokens=None)

    max_memory_reserved = torch.cuda.max_memory_reserved()
    print("Maximum memory reserved: %0.3f Gb / %0.3f Gb" % (max_memory_reserved / 10**9, total_gpu_mem))
