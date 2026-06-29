"""
MaskRQTransformer3Dv2 — Axial Attention RQ + MaskGIT for 3D Volumetric SR

Replaces the body-sum + causal-head factorisation from v1 with axial attention
that keeps per-depth token identity throughout the full transformer stack.

Architecture:
  Input: (B, L, D) codes → embed each (l, d) pair → x: (B, L, D, E)

  AxialTransformer: interleaved spatial and depth blocks
    SpatialBlock: (B,L,D,E) → reshape (B*D, L, E) → bidirectional self-attn
                  + LR cross-attn → reshape back (B,L,D,E)
    DepthBlock:   (B,L,D,E) → reshape (B*L, D, E) → bidirectional self-attn
                  → reshape back (B,L,D,E)

  Output: (B, L, D, E) → per-depth logits list[(B, L, n_embed+1)]

Key difference vs v1:
  v1 sums depths before spatial attention → the body receives one vector per
  position and cannot identify which specific depth is wrong.
  v2 keeps all D depth tokens separate throughout → every spatial block has
  direct, unambiguous access to each depth's residual.

Complexity: O(L²·D) spatial + O(D²·L) depth ≈ O(L²·D)   [linear in D]

References:
  RQTransformer:  https://arxiv.org/abs/2203.01941
  MaskGIT:        https://arxiv.org/abs/2202.04200
  DiT:            https://arxiv.org/abs/2212.09748
  TimeSformer:    https://arxiv.org/abs/2102.05095  (factorised space-time attention)
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from models.MaskTransformer3D import (
    FeedForward, QKNorm, RMSNorm, AdaNorm, CrossAttention, modulate, param_count,
)


# ── Shared attention (unchanged from v1) ─────────────────────────────────────

class AttentionRQ(nn.Module):
    """Self-attention with an optional boolean attention mask."""

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

    def forward(self, x, attn_mask=None):
        B, L, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq, xk = self.qk_norm(xq, xk, xv)
        xq = xq.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.,
        )
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.wo(out)


# ── Axial blocks ──────────────────────────────────────────────────────────────

class SpatialBlock(nn.Module):
    """Bidirectional spatial attention over L positions, applied per depth slice.

    Reshapes (B, L, D, E) → (B*D, L, E) so each depth level attends to its own
    spatial neighbourhood independently. LR cross-attention is broadcast over the
    D depth slices by expanding lr_tokens to (B*D, N_lr, E).
    """

    def __init__(self, dim, heads, mlp_dim, dropout=0., use_cross_attn=False):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.adaln_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.ln1  = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
        self.attn = AttentionRQ(dim, heads, dropout=dropout)
        self.ln2  = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
        self.ff   = FeedForward(dim, mlp_dim, dropout=dropout)
        if use_cross_attn:
            self.ln_cross  = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
            self.cross_attn = CrossAttention(dim, heads, dropout=dropout)

    def forward(self, x, cond, lr_tokens=None):
        """
        x:         (B, L, D, E)
        cond:      (B, E)        — mean-pooled LR for AdaLN
        lr_tokens: (B, N_lr, E) — full LR sequence for cross-attn, or None
        """
        B, L, D, E = x.shape

        # Merge depth into batch; each depth slice is an independent spatial sequence
        x_s = x.permute(0, 2, 1, 3).reshape(B * D, L, E)           # (B*D, L, E)
        cond_s = cond.unsqueeze(1).expand(B, D, E).reshape(B * D, E)   # (B*D, E)

        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaln_mlp(cond_s).chunk(6, dim=1)
        x_s = x_s + alpha1.unsqueeze(1) * self.attn(modulate(self.ln1(x_s), gamma1, beta1))

        if self.use_cross_attn and lr_tokens is not None:
            N_lr = lr_tokens.shape[1]
            lr_s = lr_tokens.unsqueeze(1).expand(B, D, N_lr, E).reshape(B * D, N_lr, E)
            x_s  = x_s + self.cross_attn(self.ln_cross(x_s), lr_s)

        x_s = x_s + alpha2.unsqueeze(1) * self.ff(modulate(self.ln2(x_s), gamma2, beta2))

        return x_s.reshape(B, D, L, E).permute(0, 2, 1, 3)            # (B, L, D, E)


class DepthBlock(nn.Module):
    """Bidirectional depth attention over D depths, applied per spatial position.

    Reshapes (B, L, D, E) → (B*L, D, E) so each position reasons about its own
    full depth stack independently. No LR cross-attention — spatial blocks already
    carry the LR signal into the representation at each depth.
    """

    def __init__(self, dim, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.adaln_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.ln1  = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
        self.attn = AttentionRQ(dim, heads, dropout=dropout)
        self.ln2  = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
        self.ff   = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x, cond):
        """
        x:    (B, L, D, E)
        cond: (B, E)
        """
        B, L, D, E = x.shape

        # Merge spatial positions into batch; each position's depth stack is independent
        x_d = x.reshape(B * L, D, E)  # (B*L, D, E)
        cond_d = cond.unsqueeze(1).expand(B, L, E).reshape(B * L, E)  # (B*L, E)

        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaln_mlp(cond_d).chunk(6, dim=1)
        x_d = x_d + alpha1.unsqueeze(1) * self.attn(modulate(self.ln1(x_d), gamma1, beta1))
        x_d = x_d + alpha2.unsqueeze(1) * self.ff(modulate(self.ln2(x_d), gamma2, beta2))

        return x_d.reshape(B, L, D, E)                                 # (B, L, D, E)


# ── Schedule helper ───────────────────────────────────────────────────────────

def _make_depth_schedule(n_spatial: int, n_depth: int) -> list:
    """Return list of length n_spatial: depth-block index inserted after spatial block i, or -1.

    Distributes n_depth depth blocks evenly across n_spatial spatial blocks.
    E.g. n_spatial=12, n_depth=4 → [−1,−1,0, −1,−1,1, −1,−1,2, −1,−1,3]
         giving the pattern [3S,1D, 3S,1D, 3S,1D, 3S,1D].
    """
    schedule = [-1] * n_spatial
    for d in range(n_depth):
        idx = int(round((d + 1) * n_spatial / n_depth)) - 1
        idx = max(0, min(n_spatial - 1, idx))
        schedule[idx] = d
    return schedule


# ── Axial transformer ─────────────────────────────────────────────────────────

class AxialTransformer(nn.Module):
    """Interleaves SpatialBlocks and DepthBlocks according to a computed schedule."""

    def __init__(self, dim, spatial_depth, depth_depth, heads, mlp_dim,
                 dropout=0., use_checkpoint=False, use_cross_attn=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.spatial_blocks = nn.ModuleList([
            SpatialBlock(dim, heads, mlp_dim, dropout=dropout, use_cross_attn=use_cross_attn)
            for _ in range(spatial_depth)
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

class MaskRQTransformer3Dv2(nn.Module):
    """
    Axial-attention RQ + MaskGIT model for 3D volumetric super-resolution.

    Compared to MaskRQTransformer3D (v1):
      - No depth summation: all D depth tokens per position flow through the
        full transformer stack unchanged, removing the fundamental ambiguity
        where the body could not distinguish which depth was wrong.
      - Causal AR head replaced by bidirectional DepthBlocks interleaved among
        SpatialBlocks — consistent with the MaskGIT non-autoregressive objective.
      - Masking can be applied independently to any (position, depth) pair.

    Args:
        seq_len:        spatial token count L = d' × h' × w'
        n_rq_depth:     number of RQ codebook depths D
        embed_dim:      shared hidden dimension
        n_embed:        codebook size; mask token index = n_embed (same for all depths)
        body_depth:     number of spatial attention blocks
        head_depth:     number of depth attention blocks interleaved among body_depth
        num_heads:      attention heads (shared across all blocks)
        mlp_ratio:      FFN hidden-dim multiplier
        dropout:        dropout rate
        lr_seq_len:     LR spatial token count (None → unconditional)
        lr_embed_dim:   channel dim of incoming LR encoder embeddings (B, N_lr, lr_embed_dim)
        use_checkpoint: gradient checkpointing
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
        lr_seq_len=None,
        lr_embed_dim=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.seq_len    = seq_len
        self.n_rq_depth = n_rq_depth
        self.embed_dim  = embed_dim
        self.n_embed    = n_embed
        self.lr_seq_len = lr_seq_len

        mlp_dim = int(embed_dim * mlp_ratio)

        # Per-depth token embeddings; index n_embed = [MASK] token
        self.tok_embs = nn.ModuleList([
            nn.Embedding(n_embed + 1, embed_dim) for _ in range(n_rq_depth)
        ])

        # Independent spatial and depth positional embeddings
        self.pos_emb   = nn.Embedding(seq_len, embed_dim)
        self.depth_emb = nn.Embedding(n_rq_depth, embed_dim)

        # LR conditioning
        if lr_seq_len is not None:
            lr_in_dim = lr_embed_dim if lr_embed_dim is not None else embed_dim
            self.lr_proj    = nn.Linear(lr_in_dim, embed_dim, bias=False)
            self.lr_pos_emb = nn.Embedding(lr_seq_len, embed_dim)
        else:
            self.uncond_emb = nn.Embedding(1, embed_dim)

        # Axial transformer: body_depth spatial blocks interleaved with head_depth depth blocks
        self.axial_transformer = AxialTransformer(
            dim=embed_dim,
            spatial_depth=body_depth,
            depth_depth=head_depth,
            heads=num_heads,
            mlp_dim=mlp_dim,
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
            codes:     (B, d, h, w, D) int64; masked positions hold self.n_embed
            lr_tokens: (B, N_lr, lr_embed_dim) or None
        Returns:
            logits: list of D tensors, each (B, L, n_embed + 1)
        """
        B, dz, dy, dx, D = codes.shape
        L = dz * dy * dx
        codes = codes.reshape(B, L, D)
        assert L == self.seq_len and D == self.n_rq_depth

        # Embed every (position, depth) pair independently — no summing
        tok_stack = torch.stack(
            [self.tok_embs[d](codes[:, :, d]) for d in range(D)], dim=2
        )                                                                   # (B, L, D, E)

        pos_emb   = self.pos_emb(torch.arange(L, device=codes.device))    # (L, E)
        depth_emb = self.depth_emb(torch.arange(D, device=codes.device))  # (D, E)

        # Additive spatial and depth positional encodings, broadcast over the other axis
        x = tok_stack + pos_emb[None, :, None, :] + depth_emb[None, None, :, :]  # (B, L, D, E)

        # LR conditioning
        if lr_tokens is not None:
            lr_ctx, cond = self._prepare_lr_context(lr_tokens)
        else:
            cond   = self.uncond_emb(torch.zeros(B, dtype=torch.long, device=codes.device))
            lr_ctx = None

        # Axial transformer — representation stays (B, L, D, E) throughout
        x = self.axial_transformer(x, cond, lr_tokens=lr_ctx)              # (B, L, D, E)

        # Final norm: reshape L×D into one sequence dim for AdaNorm, then restore
        x = self.out_norm(x.reshape(B, L * D, self.embed_dim), cond)       # (B, L*D, E)
        x = x.reshape(B, L, D, self.embed_dim)                             # (B, L, D, E)

        # Per-depth logits
        return [self.heads[d](x[:, :, d, :]) for d in range(D)]


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hr_spatial = 8
    lr_spatial = 8
    L_hr = hr_spatial ** 3
    L_lr = lr_spatial ** 3
    D = 4
    n_embed = 512
    lr_embed_dim = 256

    configs = {
        "tiny":  dict(embed_dim=256, body_depth=3,  head_depth=3, num_heads=4),
        "small": dict(embed_dim=384, body_depth=4,  head_depth=4, num_heads=6),
        "base":  dict(embed_dim=512, body_depth=6, head_depth=6, num_heads=8),
    }

    for name, cfg in configs.items():
        model = MaskRQTransformer3Dv2(
            seq_len=L_hr, n_rq_depth=D, n_embed=n_embed,
            lr_seq_len=L_lr, lr_embed_dim=lr_embed_dim, dropout=0.1, **cfg,
        ).to(device)
        param_count(f"MaskRQTransformer3Dv2-{name}", model)

        codes_5d = torch.randint(0, n_embed, (2, hr_spatial, hr_spatial, hr_spatial, D), device=device)
        lr_emb   = torch.randn(2, L_lr, lr_embed_dim, device=device)
        logits   = model(codes_5d, lr_tokens=lr_emb)

        assert len(logits) == D
        assert logits[0].shape == (2, L_hr, n_embed + 1), logits[0].shape
        print(f"[{name}] forward ok — logits[0]: {logits[0].shape}")

        schedule = model.axial_transformer.depth_schedule
        n_placed = sum(1 for v in schedule if v >= 0)
        assert n_placed == cfg["head_depth"], f"Expected {cfg['head_depth']} depth blocks, got {n_placed}"
        print(f"[{name}] depth_schedule: {schedule}\n")
