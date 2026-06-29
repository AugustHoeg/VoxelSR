"""
MaskRQTransformer3D — Two-Transformer RQ + MaskGIT for 3D Volumetric SR

Two factorised transformers following the RQTransformer decomposition:

  Body (spatial): bidirectional transformer over summed depth embeddings.
    Input:  codes summed across D depths → (B, L, E)
    Attn:   fully bidirectional — no mask (MaskGIT style)
    Cond:   LR cross-attention + AdaLN from mean-pooled LR tokens
    Output: spatial_ctx (B, L, E)

  Head (depth): lightweight causal AR transformer per spatial position.
    Input:  [SOS | tok_0 | tok_1 | ... | tok_{D-2}] per position → (B*L, D, E)
    Attn:   causal lower-triangular D×D boolean mask
    Cond:   spatial_ctx (B*L, E) conditions all D depth tokens via AdaLN shift/scale
            — every depth token is modulated equally, no positional hierarchy implied
    Output: (B, L, D, E) → per-depth logits list[(B, L, n_embed+1)]

Complexity: O(L²) body + O(D²·L) head ≈ O(L²) since D << L.

References:
  RQTransformer: https://arxiv.org/abs/2203.01941
  MaskGIT:       https://arxiv.org/abs/2202.04200
  DiT:           https://arxiv.org/abs/2212.09748
"""

import math

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from models.MaskTransformer3D import (
    FeedForward, QKNorm, RMSNorm, AdaNorm, CrossAttention, modulate, param_count,
)


# ── Shared attention module ───────────────────────────────────────────────────

class AttentionRQ(nn.Module):
    """Self-attention with an optional boolean attention mask.

    Serves both the body (no mask → bidirectional) and the head (lower-triangular
    causal mask). SDPA skips computation for False entries in the boolean mask.
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=False):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.n_heads   = num_heads
        self.head_dim  = embed_dim // num_heads
        self.dropout   = dropout
        self.wq = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.wk = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.wv = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.wo = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.qk_norm = QKNorm(embed_dim)

    def forward(self, x, attn_mask=None):
        """
        x:         (B, L, embed_dim)
        attn_mask: (L, L) bool — True where attention is allowed, or None for full attention.
        """
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


# ── Body blocks (DiT-AdaLN + optional LR cross-attention) ────────────────────

class BlockRQ(nn.Module):
    """DiT-AdaLN body block — bidirectional self-attention + optional LR cross-attention."""

    def __init__(self, dim, heads, mlp_dim, dropout=0., use_cross_attn=False):
        super().__init__()
        self.adaln_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.ln1 = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
        self.attn = AttentionRQ(dim, heads, dropout=dropout)
        self.ln2 = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
        self.ff  = FeedForward(dim, mlp_dim, dropout=dropout)
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.ln_cross  = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
            self.cross_attn = CrossAttention(dim, heads, dropout=dropout)

    def forward(self, x, cond, lr_tokens=None, attn_mask=None):
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaln_mlp(cond).chunk(6, dim=1)
        x = x + alpha1.unsqueeze(1) * self.attn(modulate(self.ln1(x), gamma1, beta1), attn_mask=attn_mask)
        if self.use_cross_attn and lr_tokens is not None:
            x = x + self.cross_attn(self.ln_cross(x), lr_tokens)
        x = x + alpha2.unsqueeze(1) * self.ff(modulate(self.ln2(x), gamma2, beta2))
        return x


class BodyTransformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0., use_checkpoint=False, use_cross_attn=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.layers = nn.ModuleList([
            BlockRQ(dim, heads, mlp_dim, dropout=dropout, use_cross_attn=use_cross_attn)
            for _ in range(depth)
        ])

    def forward(self, x, cond, lr_tokens=None):
        for block in self.layers:
            if self.use_checkpoint:
                # attn_mask=None → bidirectional; pass as positional arg for checkpoint compat
                x = checkpoint.checkpoint(block, x, cond, lr_tokens, None, use_reentrant=False)
            else:
                x = block(x, cond, lr_tokens=lr_tokens, attn_mask=None)
        return x


# ── Head blocks (AdaLN-conditioned, causal) ───────────────────────────────────

class HeadBlock(nn.Module):
    """DiT-AdaLN causal block for the depth transformer.

    spatial_ctx (B*L, E) conditions all D depth tokens via AdaLN shift/scale —
    equivalent to what DiT does with class embeddings. Every depth token receives
    the same conditioning signal, with no positional hierarchy implied.
    """

    def __init__(self, dim, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.adaln_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.ln1  = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
        self.attn = AttentionRQ(dim, heads, dropout=dropout)
        self.ln2  = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
        self.ff   = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x, cond, attn_mask=None):
        """
        x:        (B*L, D, E)
        cond:     (B*L, E)   — spatial_ctx for each position, drives AdaLN
        attn_mask: (D, D) causal boolean mask
        """
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaln_mlp(cond).chunk(6, dim=1)
        x = x + alpha1.unsqueeze(1) * self.attn(modulate(self.ln1(x), gamma1, beta1), attn_mask=attn_mask)
        x = x + alpha2.unsqueeze(1) * self.ff(modulate(self.ln2(x), gamma2, beta2))
        return x


class HeadTransformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0., use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.layers = nn.ModuleList([
            HeadBlock(dim, heads, mlp_dim, dropout=dropout)
            for _ in range(depth)
        ])

    def forward(self, x, cond, attn_mask=None):
        for block in self.layers:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x, cond, attn_mask, use_reentrant=False)
            else:
                x = block(x, cond, attn_mask=attn_mask)
        return x


# ── Main model ────────────────────────────────────────────────────────────────

class MaskRQTransformer3D(nn.Module):
    """
    Two-transformer RQ + MaskGIT model for 3D volumetric super-resolution.

    Body transformer: bidirectional over L spatial positions.
      Each position is represented by the sum of its D depth token embeddings,
      so masked positions contribute their [MASK] embedding to the sum.
      LR tokens condition the body via cross-attention and AdaLN.

    Head transformer: causal AR over D depth tokens, run independently for each
      of the B*L spatial positions. The body's spatial_ctx is injected as the
      first (SOS-like) token in the depth sequence.

    Args:
        seq_len:        spatial token count L = d' * h' * w' of the HR feature map
        n_rq_depth:     number of RQ codebook depths D
        embed_dim:      shared hidden dimension for both transformers
        n_embed:        codebook size (uniform across all depths); mask token = n_embed
        body_depth:     number of body transformer layers
        head_depth:     number of head transformer layers (typically much smaller than body_depth)
        num_heads:      attention heads (shared between body and head)
        mlp_ratio:      FFN hidden-dim multiplier
        dropout:        dropout rate
        lr_seq_len:     LR spatial token count (None = unconditional)
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
        self.n_embed    = n_embed       # mask token id (same for all depths)
        self.lr_seq_len = lr_seq_len

        mlp_dim = int(embed_dim * mlp_ratio)

        # Per-depth token embeddings; index n_embed = [MASK] token
        self.tok_embs = nn.ModuleList([
            nn.Embedding(n_embed + 1, embed_dim) for _ in range(n_rq_depth)
        ])

        # Spatial positional embedding for body input
        self.pos_emb = nn.Embedding(seq_len, embed_dim)

        # Depth positional embedding for head input sequence (D positions)
        self.depth_emb = nn.Embedding(n_rq_depth, embed_dim)

        # Learned start-of-sequence token for the head (replaces spatial_ctx as token 0)
        self.head_sos = nn.Embedding(1, embed_dim)

        # LR conditioning
        if lr_seq_len is not None:
            lr_in_dim = lr_embed_dim if lr_embed_dim is not None else embed_dim
            self.lr_proj    = nn.Linear(lr_in_dim, embed_dim, bias=False)
            self.lr_pos_emb = nn.Embedding(lr_seq_len, embed_dim)
        else:
            self.uncond_emb = nn.Embedding(1, embed_dim)

        # Body: bidirectional spatial transformer
        self.body_transformer = BodyTransformer(
            dim=embed_dim, depth=body_depth, heads=num_heads, mlp_dim=mlp_dim,
            dropout=dropout, use_checkpoint=use_checkpoint,
            use_cross_attn=lr_seq_len is not None,
        )
        self.body_norm = AdaNorm(x_dim=embed_dim, y_dim=embed_dim)

        # Head: causal depth transformer
        self.head_transformer = HeadTransformer(
            dim=embed_dim, depth=head_depth, heads=num_heads, mlp_dim=mlp_dim,
            dropout=dropout, use_checkpoint=use_checkpoint,
        )
        self.head_norm = RMSNorm(embed_dim, linear=True, bias=False, eps=1e-5)

        # Per-depth prediction heads, weight-tied to token embeddings
        self.heads = nn.ModuleList([
            nn.Linear(embed_dim, n_embed + 1, bias=False) for _ in range(n_rq_depth)
        ])
        for d in range(n_rq_depth):
            self.heads[d].weight = self.tok_embs[d].weight

        # Causal D×D boolean mask for head — lower-triangular, non-persistent
        self.register_buffer(
            'causal_depth_mask',
            torch.tril(torch.ones(n_rq_depth, n_rq_depth, dtype=torch.bool)),
            persistent=False,
        )

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
        nn.init.normal_(self.pos_emb.weight,    std=0.02)
        nn.init.normal_(self.depth_emb.weight,  std=0.02)
        nn.init.normal_(self.head_sos.weight,   std=0.02)

        if self.lr_seq_len is not None:
            nn.init.normal_(self.lr_pos_emb.weight, std=0.02)
        else:
            nn.init.normal_(self.uncond_emb.weight, std=0.02)

        # DiT-style zero-init for body blocks
        for block in self.body_transformer.layers:
            nn.init.constant_(block.adaln_mlp[1].weight, 0)
            nn.init.constant_(block.adaln_mlp[1].bias,   0)
            if block.use_cross_attn:
                nn.init.constant_(block.cross_attn.wo.weight, 0)

        # DiT-style zero-init for head blocks: AdaLN starts as identity
        for block in self.head_transformer.layers:
            nn.init.constant_(block.adaln_mlp[1].weight, 0)
            nn.init.constant_(block.adaln_mlp[1].bias,   0)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _prepare_lr_context(self, lr_tokens: torch.Tensor):
        """
        lr_tokens: (B, N_lr, lr_embed_dim)
        Returns:
            lr_ctx: (B, N_lr, embed_dim)  full sequence for cross-attention
            cond:   (B, embed_dim)         mean-pooled for AdaLN
        """
        lr_pos = torch.arange(lr_tokens.shape[1], device=lr_tokens.device)
        lr_ctx = self.lr_proj(lr_tokens) + self.lr_pos_emb(lr_pos)
        return lr_ctx, lr_ctx.mean(dim=1)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, codes: torch.Tensor, lr_tokens: torch.Tensor = None):
        """
        Args:
            codes:     (B, d, h, w, D) int64
                       Masked positions hold self.n_embed (the [MASK] token id).
            lr_tokens: (B, N_lr, lr_embed_dim) pre-encoded LR embeddings, or None.
        Returns:
            logits: list of D tensors, each (B, L, n_embed + 1)
                    logits[d] predicts code at depth d given spatial_ctx + codes 0..d-1.
        """
        B, dz, dy, dx, D = codes.shape
        L = dz * dy * dx
        codes = codes.reshape(B, L, D)
        assert L == self.seq_len and D == self.n_rq_depth

        pos_emb   = self.pos_emb(torch.arange(L, device=codes.device))    # (L, E)
        depth_emb = self.depth_emb(torch.arange(D, device=codes.device))  # (D, E)

        # ── Body ──────────────────────────────────────────────────────────────
        # Sum token embeddings across all D depths into a single vector per position
        tok_stack = torch.stack(
            [self.tok_embs[d](codes[:, :, d]) for d in range(D)], dim=2
        )                                               # (B, L, D, E)
        x_body = tok_stack.sum(dim=2) + pos_emb        # (B, L, E)

        if lr_tokens is not None:
            lr_ctx, cond = self._prepare_lr_context(lr_tokens)
        else:
            cond   = self.uncond_emb(torch.zeros(B, dtype=torch.long, device=codes.device))
            lr_ctx = None

        spatial_ctx = self.body_transformer(x_body, cond, lr_tokens=lr_ctx)  # (B, L, E)
        spatial_ctx = self.body_norm(spatial_ctx, cond)                       # (B, L, E)

        # ── Head ──────────────────────────────────────────────────────────────
        # Build causal depth sequence: [SOS | code_0 | ... | code_{D-2}]
        # spatial_ctx conditions all depth tokens equally via AdaLN (no positional hierarchy).
        sos = self.head_sos(torch.zeros(1, dtype=torch.long, device=codes.device))
        sos = sos.view(1, 1, 1, -1).expand(B, L, 1, -1)               # (B, L, 1, E)
        head_input = torch.cat([sos, tok_stack[:, :, :-1, :]], dim=2)  # (B, L, D, E)
        head_input = head_input + depth_emb                             # depth positional encoding
        head_input = head_input.reshape(B * L, D, -1)                  # (B*L, D, E)

        # spatial_ctx as AdaLN conditioning — one vector per spatial position
        cond_head = spatial_ctx.reshape(B * L, -1)                     # (B*L, E)

        # NOTE: each depth is conditioned identically - should we instead let the condition
        # vary by having separate modulations for each depth?
        head_out = self.head_transformer(head_input, cond_head, attn_mask=self.causal_depth_mask)
        head_out = self.head_norm(head_out)                             # (B*L, D, E)
        head_out = head_out.reshape(B, L, D, -1)                       # (B, L, D, E)

        # Per-depth logits: head_out[:, :, d, :] predicts code at depth d
        return [self.heads[d](head_out[:, :, d, :]) for d in range(D)]


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hr_spatial = 8
    lr_spatial = 4
    L_hr = hr_spatial ** 3   # 512
    L_lr = lr_spatial ** 3   # 64
    D = 4
    n_embed = 512
    lr_embed_dim = 256

    configs = {
        "tiny":  dict(embed_dim=256, body_depth=4,  head_depth=2, num_heads=4),
        "small": dict(embed_dim=384, body_depth=6,  head_depth=2, num_heads=6),
        "base":  dict(embed_dim=512, body_depth=12, head_depth=4, num_heads=8),
    }

    for name, cfg in configs.items():
        model = MaskRQTransformer3D(
            seq_len=L_hr, n_rq_depth=D, n_embed=n_embed,
            lr_seq_len=L_lr, lr_embed_dim=lr_embed_dim, dropout=0.1, **cfg,
        ).to(device)
        param_count(f"MaskRQTransformer3D-{name}", model)

        codes_5d = torch.randint(0, n_embed, (2, hr_spatial, hr_spatial, hr_spatial, D), device=device)
        lr_emb = torch.randn(2, L_lr, lr_embed_dim, device=device)
        logits = model(codes_5d, lr_tokens=lr_emb)

        assert len(logits) == D
        assert logits[0].shape == (2, L_hr, n_embed + 1), logits[0].shape
        print(f"[{name}] forward ok — logits[0]: {logits[0].shape}")

        # Verify causal mask shape and content
        mask = model.causal_depth_mask
        assert mask.shape == (D, D) and mask.dtype == torch.bool
        assert mask[0, 0] and not mask[0, 1], "causal mask should be lower-triangular"
        print(f"[{name}] causal_depth_mask ({D}×{D}) ok\n")
