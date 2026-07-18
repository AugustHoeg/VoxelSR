"""
MaskRQTransformer3Dv4 — Windowed Cross-Attention RQ + MaskGIT for 3D SR

Identical to v3 (Swin-style windowed spatial self-attention, global depth attention,
AdaLN-Zero, per-depth MaskGIT heads) with one change:

  v3 cross-attention:  full attention — each HR token queries all N_lr LR tokens
                       complexity O(L_hr × L_lr) per block

  v4 cross-attention:  windowed — HR and LR token grids are each partitioned into
                       the same number of windows; each HR window attends only to
                       its spatially matching LR window.
                       complexity O(L_hr × lr_win_len) per block

Window matching: HR grid (dz, dy, dx) with window_size W has
  nW = (dz/W)(dy/W)(dx/W) windows.
LR grid (lz, ly, lx) is partitioned into the same nW windows with per-dim sizes
  lr_ws[i] = lr_shape[i] * W // hr_shape[i]
which requires  lr_shape[i] * W  divisible by  hr_shape[i]  for each i.

Example — 4× isotropic SR, hr_shape=(16,16,16), lr_shape=(4,4,4), W=4:
  lr_ws = (1, 1, 1): every HR window of W³=64 queries attends to 1 LR token.

Cross-attention windows are matched by spatial index with no cyclic shift in
even blocks (W-MSA style).  In odd blocks both HR and LR grids are shifted by
the same *relative* fraction before partitioning (SW-MSA style), giving boundary
HR tokens access to their genuine LR spatial neighbours rather than wrap-around
context from the far side of the volume.  A precomputed boundary mask — shaped
(nW, W_hr³, W_lr³) — suppresses the geometrically invalid cross-region pairs
that appear in boundary windows after the cyclic shift, exactly as the
self-attention SW-MSA mask does for HR self-pairs.

LR shift per dim = hr_shift * lr_shape[i] // hr_shape[i]  (must be an integer).
For the common hr_shape == lr_shape case this equals hr_shift exactly.

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
    FeedForward, RMSNorm, AdaNorm, modulate, param_count,
)
from models.MaskRQTransformer3Dv2 import DepthBlock, _make_depth_schedule
from models.MaskRQTransformer3Dv3 import (
    window_partition3D, window_reverse3D, compute_mask,
    WindowAttention3D, WindowSelfAttention3D,
)


# ── Windowed cross-attention primitives ──────────────────────────────────────

def _partition_general(x, ws):
    """(B, Hz, Hy, Hx, C) → (nW*B, wz*wy*wx, C) for per-dim window sizes ws=(wz,wy,wx)."""
    B, Hz, Hy, Hx, C = x.shape
    wz, wy, wx = ws
    x = x.view(B, Hz // wz, wz, Hy // wy, wy, Hx // wx, wx, C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return x.view(-1, wz * wy * wx, C)


def compute_cross_mask(hr_dims, lr_dims, hr_ws, lr_ws, hr_shift, lr_shift):
    """Precompute (nW, hr_ws³, lr_win_len) additive cross-attention mask.

    After a proportional cyclic shift, boundary windows contain tokens from two
    spatial regions.  This mask zeroes out cross-region HR↔LR pairs (assigns
    -100) so boundary HR tokens only attend to their genuine LR neighbours.

    Both HR and LR label maps use the same 27-region scheme (3 slices per dim)
    with identical region IDs for geometrically corresponding regions.  A pair is
    valid iff both tokens carry the same region label.

    Args:
        hr_dims:   (dz, dy, dx)
        lr_dims:   (lz, ly, lx)
        hr_ws:     scalar cubic HR window size
        lr_ws:     (wz, wy, wx) per-dim LR window sizes
        hr_shift:  scalar HR shift (= window_size // 2)
        lr_shift:  (sz, sy, sx) per-dim LR shifts proportional to hr_shift
    """
    dz, dy, dx = hr_dims
    lz, ly, lx = lr_dims
    lwz, lwy, lwx = lr_ws
    lsz, lsy, lsx = lr_shift

    # HR region labels — same 27-region scheme as compute_mask
    hr_map = torch.zeros(1, dz, dy, dx, 1)
    cnt = 0
    for sz in (slice(-hr_ws), slice(-hr_ws, -hr_shift), slice(-hr_shift, None)):
        for sy in (slice(-hr_ws), slice(-hr_ws, -hr_shift), slice(-hr_shift, None)):
            for sx in (slice(-hr_ws), slice(-hr_ws, -hr_shift), slice(-hr_shift, None)):
                hr_map[:, sz, sy, sx, :] = cnt
                cnt += 1
    hr_labels = window_partition3D(hr_map, hr_ws).squeeze(-1)   # (nW, hr_ws³)

    # LR region labels — same 27 IDs, proportional slice boundaries
    lr_map = torch.zeros(1, lz, ly, lx, 1)
    cnt = 0
    for sz in (slice(-lwz), slice(-lwz, -lsz), slice(-lsz, None)):
        for sy in (slice(-lwy), slice(-lwy, -lsy), slice(-lsy, None)):
            for sx in (slice(-lwx), slice(-lwx, -lsx), slice(-lsx, None)):
                lr_map[:, sz, sy, sx, :] = cnt
                cnt += 1
    lr_labels = _partition_general(lr_map, lr_ws).squeeze(-1)   # (nW, lr_win_len)

    # True = valid pair (same spatial region); False = wrapped boundary pair, masked out
    return hr_labels.unsqueeze(2) == lr_labels.unsqueeze(1)  # (nW, hr_ws³, lr_win_len)


# ── Windowed cross-attention ──────────────────────────────────────────────────

class WindowedCrossAttention3D(nn.Module):
    """Windowed 3D cross-attention: HR queries × spatially matching LR keys/values.

    Both grids are partitioned into the same number of windows; SDPA runs only
    within matched window pairs.

    When shift_size > 0 (SW-MSA style): both HR and LR grids are cyclically
    shifted by the same relative fraction before partitioning, giving boundary HR
    tokens access to genuine LR neighbours from adjacent windows.  A precomputed
    boundary mask suppresses the geometrically invalid cross-region pairs that
    would otherwise appear in wrapped boundary windows — no wrap-around context
    leaks through.

    Constraint for shifting: shift_size * lr_shape[i] must be divisible by
    hr_shape[i] for all i (ensures an integer LR shift per dim).

    Complexity: O(L_hr × lr_win_len)  vs  O(L_hr × L_lr) for full attention.
    """

    def __init__(self, dim, num_heads, hr_shape, lr_shape, hr_window_size,
                 shift_size=0, dropout=0.):
        super().__init__()
        self.hr_shape = tuple(hr_shape)
        self.lr_shape = tuple(lr_shape)
        self.hr_ws    = hr_window_size
        self.hr_shift = shift_size
        self.n_heads  = num_heads
        self.head_dim = dim // num_heads
        self.dropout  = dropout

        # Per-dim LR window sizes
        lr_ws = []
        for h, l in zip(hr_shape, lr_shape):
            assert (l * hr_window_size) % h == 0, (
                f"lr_shape dim {l} × window_size {hr_window_size} "
                f"not divisible by hr_shape dim {h}"
            )
            lr_ws.append(l * hr_window_size // h)
        self.lr_ws      = tuple(lr_ws)
        self.lr_win_len = lr_ws[0] * lr_ws[1] * lr_ws[2]

        # Proportional per-dim LR shift and boundary mask
        if shift_size > 0:
            lr_shift = []
            for h, l in zip(hr_shape, lr_shape):
                assert (shift_size * l) % h == 0, (
                    f"Proportional LR shift not integer: "
                    f"shift={shift_size} × lr_dim={l} / hr_dim={h}"
                )
                lr_shift.append(shift_size * l // h)
            self.lr_shift = tuple(lr_shift)
            cross_mask = compute_cross_mask(
                hr_dims=hr_shape, lr_dims=lr_shape,
                hr_ws=hr_window_size, lr_ws=self.lr_ws,
                hr_shift=shift_size, lr_shift=self.lr_shift,
            )
            self.register_buffer("cross_mask", cross_mask)
        else:
            self.lr_shift = (0, 0, 0)
            self.register_buffer("cross_mask", None)

        self.wq  = nn.Linear(dim, dim, bias=False)
        self.wkv = nn.Linear(dim, 2 * dim, bias=False)  # fused K+V — one matmul
        self.wo  = nn.Linear(dim, dim, bias=False)
        # Separate norms so K can be normalised at B level before depth expansion
        self.q_norm = RMSNorm(dim, linear=False, bias=False)
        self.k_norm = RMSNorm(dim, linear=False, bias=False)

    def forward(self, x_hr, lr_ctx):
        """
        x_hr:   (M, L_hr, E)  HR queries  (M = B*D, one per RQ depth slice)
        lr_ctx: (M, L_lr, E)  LR context  (already expanded across depth slices)
        Returns: (M, L_hr, E)
        """
        M, L_hr, E = x_hr.shape
        dz, dy, dx = self.hr_shape
        lz, ly, lx = self.lr_shape

        hr_3d = x_hr.view(M, dz, dy, dx, E)
        lr_3d = lr_ctx.view(M, lz, ly, lx, E)

        # Proportional cyclic shift (SW-MSA cross-attention, odd blocks only)
        if self.hr_shift > 0:
            s = self.hr_shift
            lsz, lsy, lsx = self.lr_shift
            hr_3d = torch.roll(hr_3d, shifts=(-s, -s, -s), dims=(1, 2, 3))
            lr_3d = torch.roll(lr_3d, shifts=(-lsz, -lsy, -lsx), dims=(1, 2, 3))

        # Partition HR → (M*nW, W³, E)
        hr_wins = window_partition3D(hr_3d, self.hr_ws)
        Mw   = hr_wins.shape[0]       # M * nW
        nW   = Mw // M
        W_hr = hr_wins.shape[1]
        W_lr = self.lr_win_len

        # Partition LR → (M*nW, W_lr, E)
        lr_wins = _partition_general(lr_3d, self.lr_ws)

        # Q from HR
        xq = self.q_norm(self.wq(hr_wins)).to(lr_wins)           # (M*nW, W_hr, E)
        q  = xq.view(Mw, W_hr, self.n_heads, self.head_dim).transpose(1, 2)

        # K/V from LR — single fused matmul
        xk, xv = self.wkv(lr_wins).chunk(2, dim=-1)              # (M*nW, W_lr, E) each
        xk = self.k_norm(xk).to(xv)
        k = xk.view(Mw, W_lr, self.n_heads, self.head_dim).transpose(1, 2)
        v = xv.view(Mw, W_lr, self.n_heads, self.head_dim).transpose(1, 2)

        # Boundary mask: suppress invalid cross-region pairs in wrapped windows
        if self.cross_mask is not None:
            attn_bias = (self.cross_mask
                         .unsqueeze(0).expand(M, nW, W_hr, W_lr)
                         .reshape(Mw, W_hr, W_lr)
                         .unsqueeze(1))
        else:
            attn_bias = None

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.,
        )
        out = out.transpose(1, 2).reshape(Mw, W_hr, E)

        # Reverse HR windows and undo cyclic shift
        out_3d = window_reverse3D(out, self.hr_ws, dims=(M, dz, dy, dx))
        if self.hr_shift > 0:
            s = self.hr_shift
            out_3d = torch.roll(out_3d, shifts=(s, s, s), dims=(1, 2, 3))

        return self.wo(out_3d.view(M, L_hr, E))


# ── Windowed spatial block ────────────────────────────────────────────────────

class WindowedSpatialBlock(nn.Module):
    """Swin W-MSA/SW-MSA self-attention + windowed cross-attention, per depth slice.

    Identical layout to v3 but cross-attention is windowed (WindowedCrossAttention3D)
    instead of global.  lr_spatial_shape must be provided when use_cross_attn=True.
    """

    def __init__(self, dim, heads, mlp_dim, spatial_shape, window_size, shift_size=0,
                 dropout=0., use_cross_attn=False, lr_spatial_shape=None):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.adaln_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.ln1  = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
        self.attn = WindowSelfAttention3D(dim, heads, spatial_shape, window_size,
                                          shift_size=shift_size, attn_drop=dropout)
        self.ln2  = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
        self.ff   = FeedForward(dim, mlp_dim, dropout=dropout)
        if use_cross_attn:
            assert lr_spatial_shape is not None, (
                "lr_spatial_shape is required for windowed cross-attention"
            )
            self.ln_cross = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
            self.cross_attn = WindowedCrossAttention3D(
                dim, heads,
                hr_shape=spatial_shape,
                lr_shape=lr_spatial_shape,
                hr_window_size=window_size,
                shift_size=shift_size,
                dropout=dropout,
            )

    def forward(self, x, cond, lr_tokens=None):
        """
        x:         (B, L, D, E)
        cond:      (B, E)
        lr_tokens: (B, N_lr, E) or None
        """
        B, L, D, E = x.shape
        x_s = x.permute(0, 2, 1, 3).reshape(B * D, L, E)
        cond_s = cond.unsqueeze(1).expand(B, D, E).reshape(B * D, E)

        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaln_mlp(cond_s).chunk(6, dim=1)
        x_s = x_s + alpha1.unsqueeze(1) * self.attn(modulate(self.ln1(x_s), gamma1, beta1))

        if self.use_cross_attn and lr_tokens is not None:
            # Expand lr_tokens (B, N_lr, E) → (B*D, N_lr, E) to match x_s
            N_lr = lr_tokens.shape[1]
            lr_exp = lr_tokens.unsqueeze(1).expand(B, D, N_lr, E).reshape(B * D, N_lr, E)
            x_s = x_s + self.cross_attn(self.ln_cross(x_s), lr_exp)

        x_s = x_s + alpha2.unsqueeze(1) * self.ff(modulate(self.ln2(x_s), gamma2, beta2))
        return x_s.reshape(B, D, L, E).permute(0, 2, 1, 3)


# ── Axial transformer ─────────────────────────────────────────────────────────

class AxialWindowedTransformer(nn.Module):
    """Interleaves windowed SpatialBlocks (with windowed cross-attention) and DepthBlocks."""

    def __init__(self, dim, spatial_depth, depth_depth, heads, mlp_dim,
                 spatial_shape, window_size, dropout=0., use_checkpoint=False,
                 use_cross_attn=False, lr_spatial_shape=None):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        shift     = window_size // 2
        can_shift = all(s > window_size for s in spatial_shape)

        self.spatial_blocks = nn.ModuleList([
            WindowedSpatialBlock(
                dim, heads, mlp_dim, spatial_shape, window_size,
                shift_size=(shift if (can_shift and i % 2 == 1) else 0),
                dropout=dropout,
                use_cross_attn=use_cross_attn,
                lr_spatial_shape=lr_spatial_shape,
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

class MaskRQTransformer3Dv4(nn.Module):
    """
    Windowed cross-attention RQ + MaskGIT model for 3D volumetric super-resolution.

    Same as v3 (windowed self-attention, global depth attention, AdaLN-Zero) with
    the global LR cross-attention replaced by windowed cross-attention: HR and LR
    token grids are partitioned into matching windows and SDPA is run only within
    spatially corresponding pairs.

    Additional args over v3:
        lr_spatial_shape: (lz, ly, lx) LR token grid. Required when lr_seq_len is set.
                          Constraint: lr_shape[i] * window_size divisible by hr_shape[i].

    Remaining args match v3:
        seq_len, n_rq_depth, spatial_shape, window_size, embed_dim, n_embed,
        body_depth, head_depth, num_heads, mlp_ratio, dropout,
        lr_seq_len, lr_embed_dim, use_checkpoint
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
        lr_spatial_shape=None,
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
        assert spatial_shape[0] * spatial_shape[1] * spatial_shape[2] == seq_len
        for s in spatial_shape:
            assert s % window_size == 0, (
                f"spatial_shape {spatial_shape} not divisible by window_size {window_size}"
            )

        if lr_seq_len is not None:
            assert lr_spatial_shape is not None, (
                "lr_spatial_shape=(lz,ly,lx) is required when lr_seq_len is set"
            )
            lr_spatial_shape = tuple(lr_spatial_shape)
            assert len(lr_spatial_shape) == 3
            assert lr_spatial_shape[0] * lr_spatial_shape[1] * lr_spatial_shape[2] == lr_seq_len, (
                f"prod(lr_spatial_shape)={lr_spatial_shape} != lr_seq_len={lr_seq_len}"
            )

        self.seq_len          = seq_len
        self.n_rq_depth       = n_rq_depth
        self.embed_dim        = embed_dim
        self.n_embed          = n_embed
        self.lr_seq_len       = lr_seq_len
        self.spatial_shape    = spatial_shape
        self.window_size      = window_size
        self.lr_spatial_shape = lr_spatial_shape

        mlp_dim = int(embed_dim * mlp_ratio)

        self.tok_embs  = nn.ModuleList([nn.Embedding(n_embed + 1, embed_dim) for _ in range(n_rq_depth)])
        self.pos_emb   = nn.Embedding(seq_len, embed_dim)
        self.depth_emb = nn.Embedding(n_rq_depth, embed_dim)

        if lr_seq_len is not None:
            lr_in_dim = lr_embed_dim if lr_embed_dim is not None else embed_dim
            self.lr_proj    = nn.Linear(lr_in_dim, embed_dim, bias=False)
            self.lr_pos_emb = nn.Embedding(lr_seq_len, embed_dim)
        else:
            self.uncond_emb = nn.Embedding(1, embed_dim)

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
            lr_spatial_shape=lr_spatial_shape,
        )
        self.out_norm = AdaNorm(x_dim=embed_dim, y_dim=embed_dim)

        self.heads = nn.ModuleList([
            nn.Linear(embed_dim, n_embed + 1, bias=False) for _ in range(n_rq_depth)
        ])
        for d in range(n_rq_depth):
            self.heads[d].weight = self.tok_embs[d].weight

        self._init_weights()

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

        for block in self.axial_transformer.spatial_blocks:
            nn.init.constant_(block.adaln_mlp[1].weight, 0)
            nn.init.constant_(block.adaln_mlp[1].bias,   0)
            if block.use_cross_attn:
                nn.init.constant_(block.cross_attn.wo.weight, 0)

        for block in self.axial_transformer.depth_blocks:
            nn.init.constant_(block.adaln_mlp[1].weight, 0)
            nn.init.constant_(block.adaln_mlp[1].bias,   0)

    def _prepare_lr_context(self, lr_tokens):
        lr_pos = torch.arange(lr_tokens.shape[1], device=lr_tokens.device)
        lr_ctx = self.lr_proj(lr_tokens) + self.lr_pos_emb(lr_pos)
        return lr_ctx, lr_ctx.mean(dim=1)

    def forward(self, codes, lr_tokens=None):
        """
        codes:     (B, dz, dy, dx, D) int64; masked positions = self.n_embed
        lr_tokens: (B, N_lr, lr_embed_dim) or None
        Returns:   list of D tensors, each (B, L, n_embed + 1)
        """
        B, dz, dy, dx, D = codes.shape
        L = dz * dy * dx
        assert (dz, dy, dx) == self.spatial_shape
        assert L == self.seq_len and D == self.n_rq_depth
        codes = codes.reshape(B, L, D)

        tok_stack = torch.stack([self.tok_embs[d](codes[:, :, d]) for d in range(D)], dim=2)
        pos_emb = self.pos_emb(torch.arange(L, device=codes.device))
        depth_emb = self.depth_emb(torch.arange(D, device=codes.device))
        x = tok_stack + pos_emb[None, :, None, :] + depth_emb[None, None, :, :]

        if lr_tokens is not None:
            lr_ctx, cond = self._prepare_lr_context(lr_tokens)
        else:
            cond   = self.uncond_emb(torch.zeros(B, dtype=torch.long, device=codes.device))
            lr_ctx = None

        x = self.axial_transformer(x, cond, lr_tokens=lr_ctx)

        x = self.out_norm(x.reshape(B, L * D, self.embed_dim), cond)
        x = x.reshape(B, L, D, self.embed_dim)

        return [self.heads[d](x[:, :, d, :]) for d in range(D)]


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / 10**9 if torch.cuda.is_available() else 0

    print("Mem Efficient  :", torch.backends.cuda.mem_efficient_sdp_enabled())

    hr_spatial  = 16
    lr_spatial  = 16
    window_size = 16
    L_hr = hr_spatial ** 3
    L_lr = lr_spatial ** 3
    D = 8
    n_embed = 4096
    lr_emb_dim = 256

    configs = {
        "tiny": dict(embed_dim=512, body_depth=4, head_depth=4, num_heads=4),
    }

    for name, cfg in configs.items():
        # conditioned: windowed cross-attention
        model = MaskRQTransformer3Dv4(
            seq_len=L_hr, n_rq_depth=D, n_embed=n_embed,
            spatial_shape=(hr_spatial,) * 3, window_size=window_size,
            lr_seq_len=L_lr, lr_embed_dim=lr_emb_dim,
            lr_spatial_shape=(lr_spatial,) * 3,
            dropout=0.1, use_checkpoint=True, **cfg,
        ).to(device)
        param_count(f"MaskRQTransformer3Dv4-{name} (cond)", model)
        codes  = torch.randint(0, n_embed, (2, hr_spatial, hr_spatial, hr_spatial, D), device=device)
        lr_emb = torch.randn(2, L_lr, lr_emb_dim, device=device)
        logits = model(codes, lr_tokens=lr_emb)
        print(f"  logit shape: {logits[0].shape}")

    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_reserved()
        print("Maximum memory reserved: %0.3f Gb / %0.3f Gb" % (max_mem / 10**9, total_gpu_mem))
