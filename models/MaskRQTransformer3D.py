"""
MaskRQTransformer3D — RQ + MaskGIT Transformer for 3D Volumetric SR

Next-depth prediction (Markovian across depths) combined with masked generation
within each depth (MaskGIT-style iterative refinement).

Token layout:
    [depth-0 tokens (L) | depth-1 tokens (L) | ... | depth-(D-1) tokens (L)]
    Total sequence length: L * D

Markovian attention mask: depth k tokens attend bidirectionally to depth k and
depth k-1 only. All other depths are invisible. Within a depth, masked positions
receive a special mask-token embedding — the attention itself is fully bidirectional.
A boolean mask is used so SDPA can skip computation for blocked entries.

References:
  VAR     (block-causal across scales): https://arxiv.org/abs/2404.02905
  MaskGIT (masked token generation):    https://arxiv.org/abs/2202.04200
  DiT     (AdaLN-zero conditioning):    https://arxiv.org/abs/2212.09748
"""

import math

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from models.MaskTransformer3D import FeedForward, QKNorm, RMSNorm, AdaNorm, CrossAttention, modulate, param_count


class AttentionRQ(nn.Module):
    """Self-attention with a boolean Markovian depth mask."""

    def __init__(self, embed_dim, num_heads, dropout=0., bias=False):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.n_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.wq = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.wk = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.wv = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.wo = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.qk_norm = QKNorm(embed_dim)

    def forward(self, x, attn_mask=None):
        """
        x:        (B, L_total, embed_dim)
        attn_mask: (L_total, L_total) bool — True where attention is allowed.
                   SDPA broadcasts to (B, n_heads, L_total, L_total) and skips
                   computation for False entries.
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


class BlockRQ(nn.Module):
    """DiT-AdaLN block using AttentionRQ (2D bias) and optional LR cross-attention."""

    def __init__(self, dim, heads, mlp_dim, dropout=0., use_cross_attn=False):
        super().__init__()
        self.adaln_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.ln1 = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
        self.attn = AttentionRQ(dim, heads, dropout=dropout)
        self.ln2 = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.ln_cross = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
            self.cross_attn = CrossAttention(dim, heads, dropout=dropout)

    def forward(self, x, cond, lr_tokens=None, attn_mask=None):
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaln_mlp(cond).chunk(6, dim=1)
        x = x + alpha1.unsqueeze(1) * self.attn(modulate(self.ln1(x), gamma1, beta1), attn_mask=attn_mask)
        if self.use_cross_attn and lr_tokens is not None:
            x = x + self.cross_attn(self.ln_cross(x), lr_tokens)
        x = x + alpha2.unsqueeze(1) * self.ff(modulate(self.ln2(x), gamma2, beta2))
        return x


class TransformerEncoderRQ(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0., use_checkpoint=False, use_cross_attn=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.layers = nn.ModuleList([
            BlockRQ(dim, heads, mlp_dim, dropout=dropout, use_cross_attn=use_cross_attn)
            for _ in range(depth)
        ])

    def forward(self, x, cond, lr_tokens=None, attn_mask=None):
        for block in self.layers:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x, cond, lr_tokens, attn_mask, use_reentrant=False)
            else:
                x = block(x, cond, lr_tokens=lr_tokens, attn_mask=attn_mask)
        return x


class MaskRQTransformer3D(nn.Module):
    """
    RQ + MaskGIT Transformer for 3D volumetric super-resolution.

    Generates HR tokens depth-by-depth with block-causal cross-depth attention
    and MaskGIT-style masked generation within each depth.

    All RQ depths share the same codebook size n_embed. Each depth has its own
    token embedding table and prediction head (weight-tied), so the transformer
    can learn depth-specific representations while keeping the vocabulary uniform.

    Args:
        seq_len:        spatial token count L = D' * H' * W' of the HR feature map
        n_rq_depth:     number of RQ codebook depths D
        embed_dim:      transformer hidden dimension
        n_embed:        codebook size (uniform across all depths); mask token = n_embed
        depth:          number of transformer layers
        num_heads:      number of attention heads
        mlp_ratio:      FFN hidden-dim multiplier
        dropout:        attention / FFN dropout rate
        lr_seq_len:     number of LR spatial tokens (None = unconditional mode)
        lr_embed_dim:   channel dim of incoming LR encoder embeddings (B, N_lr, lr_embed_dim)
        use_checkpoint: gradient checkpointing for memory efficiency
    """

    def __init__(
        self,
        seq_len,
        n_rq_depth,
        embed_dim=512,
        n_embed=1024,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.,
        lr_seq_len=None,
        lr_embed_dim=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_rq_depth = n_rq_depth
        self.embed_dim = embed_dim
        self.n_embed = n_embed          # mask token id = n_embed (same for all depths)
        self.lr_seq_len = lr_seq_len

        mlp_dim = int(embed_dim * mlp_ratio)

        # Per-depth token embeddings; slot n_embed is the [MASK] token
        self.tok_embs = nn.ModuleList([
            nn.Embedding(n_embed + 1, embed_dim) for _ in range(n_rq_depth)
        ])

        # Spatial positional embedding — shared across all depths
        self.pos_emb = nn.Embedding(seq_len, embed_dim)

        # Depth indicator — one learned vector per depth, broadcast over L
        self.depth_emb = nn.Embedding(n_rq_depth, embed_dim)

        # LR conditioning
        if lr_seq_len is not None:
            lr_in_dim = lr_embed_dim if lr_embed_dim is not None else embed_dim
            self.lr_proj = nn.Linear(lr_in_dim, embed_dim, bias=False)
            self.lr_pos_emb = nn.Embedding(lr_seq_len, embed_dim)
        else:
            self.uncond_emb = nn.Embedding(1, embed_dim)

        self.transformer = TransformerEncoderRQ(
            dim=embed_dim,
            depth=depth,
            heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
            use_cross_attn=lr_seq_len is not None,
        )

        self.last_norm = AdaNorm(x_dim=embed_dim, y_dim=embed_dim)

        # Per-depth prediction heads, weight-tied to their token embeddings
        self.heads = nn.ModuleList([
            nn.Linear(embed_dim, n_embed + 1, bias=False) for _ in range(n_rq_depth)
        ])
        for d in range(n_rq_depth):
            self.heads[d].weight = self.tok_embs[d].weight

        # Precompute Markovian boolean attention mask — non-persistent (recomputed on load)
        self.register_buffer(
            'attn_mask',
            self._build_markov_mask(seq_len, n_rq_depth),
            persistent=False,
        )

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
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.normal_(self.depth_emb.weight, std=0.02)

        if self.lr_seq_len is not None:
            nn.init.normal_(self.lr_pos_emb.weight, std=0.02)
        else:
            nn.init.normal_(self.uncond_emb.weight, std=0.02)

        # DiT-style zero-init: AdaLN starts as identity, cross-attn output as zero
        for block in self.transformer.layers:
            nn.init.constant_(block.adaln_mlp[1].weight, 0)
            nn.init.constant_(block.adaln_mlp[1].bias, 0)
            if block.use_cross_attn:
                nn.init.constant_(block.cross_attn.wo.weight, 0)


    @staticmethod
    def _build_markov_mask(L: int, D: int) -> torch.Tensor:
        """
        (L*D, L*D) bool mask: True where token i may attend to token j.
        Depth k attends to depth k (bidirectional) and depth k-1 only — Markovian.
        """
        L_total = L * D
        depth = torch.arange(L_total) // L          # depth index for every position
        depth_i = depth.unsqueeze(1)                # (L_total, 1)
        depth_j = depth.unsqueeze(0)                # (1, L_total)
        return (depth_j >= depth_i - 1) & (depth_j <= depth_i)  # (L_total, L_total) bool


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


    def forward(self, codes: torch.Tensor, lr_tokens: torch.Tensor = None):
        """
        Args:
            codes:     (B, d, h, w, D) int64
                       Masked positions hold self.n_embed (the [MASK] token id).
            lr_tokens: (B, N_lr, lr_embed_dim)  pre-encoded LR embeddings, or None.
        Returns:
            logits: list of D tensors, each (B, L, n_embed + 1)
        """
        B, d, h, w, D = codes.shape

        codes = codes.reshape(B, d * h * w, D)
        B, L, D = codes.shape
        assert L == self.seq_len and D == self.n_rq_depth

        pos_emb   = self.pos_emb(torch.arange(L, device=codes.device))    # (L, E)
        depth_embs = self.depth_emb(torch.arange(D, device=codes.device)) # (D, E)

        # Embed each depth and concatenate into a single sequence
        depth_seqs = []
        for d in range(D):
            x_d = self.tok_embs[d](codes[:, :, d])    # (B, L, E)
            x_d = x_d + pos_emb + depth_embs[d]       # spatial + depth position
            depth_seqs.append(x_d)
        x = torch.cat(depth_seqs, dim=1)               # (B, L*D, E)

        # Conditioning
        if lr_tokens is not None:
            lr_ctx, cond = self._prepare_lr_context(lr_tokens)
        else:
            cond = self.uncond_emb(torch.zeros(B, dtype=torch.long, device=x.device))
            lr_ctx = None

        x = self.transformer(x, cond, lr_tokens=lr_ctx, attn_mask=self.attn_mask)
        x = self.last_norm(x, cond)                    # (B, L*D, E)

        x_depths = torch.chunk(x, chunks=D, dim=1)  # split into depths

        out = []
        for i in range(D):
            out.append(self.heads[i](x_depths[i]))

        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hr_spatial   = 8
    lr_spatial   = 4
    L_hr         = hr_spatial ** 3   # 512
    L_lr         = lr_spatial ** 3   # 64
    D            = 4
    n_embed      = 512
    lr_embed_dim = 256

    configs = {
        "tiny":  dict(embed_dim=256, depth=4,  num_heads=4),
        "small": dict(embed_dim=384, depth=6,  num_heads=6),
        "base":  dict(embed_dim=512, depth=12, num_heads=8),
    }

    for name, cfg in configs.items():
        model = MaskRQTransformer3D(
            seq_len=L_hr,
            n_rq_depth=D,
            n_embed=n_embed,
            lr_seq_len=L_lr,
            lr_embed_dim=lr_embed_dim,
            dropout=0.1,
            **cfg,
        ).to(device)
        param_count(f"MaskRQTransformer3D-{name}", model)

        # forward — (B, d, h, w, D) input format
        codes_5d = torch.randint(0, n_embed, (2, hr_spatial, hr_spatial, hr_spatial, D), device=device)
        lr_emb   = torch.randn(2, L_lr, lr_embed_dim, device=device)
        logits   = model(codes_5d, lr_tokens=lr_emb)
        assert len(logits) == D and logits[0].shape == (2, L_hr, n_embed + 1)
        print(f"  [{name}] forward ok  — logits[0]: {logits[0].shape}")
