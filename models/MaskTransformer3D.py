# Transformer Encoder architecture for 3D MaskGIT / MaskGIT-SR
# Adapted from Halton-MaskGIT (2D): https://github.com/valeoai/Halton-MaskGIT/blob/main/Network/transformer.py
# Original 2D influences:
#   - NanoGPT: https://github.com/karpathy/nanoGPT
#   - DiT: https://github.com/facebookresearch/DiT

import math

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


def param_count(archi, model):
    print(f"Size of model {archi}: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 10 ** 6:.3f}M")


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FeedForward(nn.Module):
    def __init__(self, dim, h_dim, multiple_of=256, bias=False, dropout=0.):
        super().__init__()
        self.dropout = dropout
        # SwiGLU: expand to 2/3 of h_dim, rounded up to multiple_of
        h_dim = int(2 * h_dim / 3)
        h_dim = multiple_of * ((h_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, h_dim, bias=bias)
        self.w2 = nn.Linear(h_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, h_dim, bias=bias)

    def forward(self, x):
        x = F.silu(self.w1(x)) * self.w3(x)
        if self.dropout > 0. and self.training:
            x = F.dropout(x, self.dropout)
        return self.w2(x)


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim, linear=False, bias=False)
        self.key_norm = RMSNorm(dim, linear=False, bias=False)

    def forward(self, q, k, v):
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., use_flash=True, bias=False):
        super().__init__()
        self.flash = use_flash
        self.n_local_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.wq = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.wv = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.wo = nn.Linear(num_heads * self.head_dim, embed_dim, bias=bias)
        self.qk_norm = QKNorm(num_heads * self.head_dim)
        self.cache = None

    def forward(self, x, mask=None):
        b, seq, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq, xk = self.qk_norm(xq, xk, xv)
        xq = xq.view(b, seq, self.n_local_heads, self.head_dim)
        xk = xk.view(b, seq, self.n_local_heads, self.head_dim)
        xv = xv.view(b, seq, self.n_local_heads, self.head_dim)
        # renamed loop var to t to avoid shadowing outer x
        xq, xk, xv = (t.transpose(1, 2) for t in (xq, xk, xv))
        if self.flash:
            if mask is not None:
                mask = mask.view(b, 1, 1, seq)
            output = F.scaled_dot_product_attention(xq, xk, xv, mask, dropout_p=self.dropout if self.training else 0.)
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, xv)
        output = output.transpose(1, 2).contiguous().view(b, seq, -1)
        proj = self.wo(output)
        if self.dropout > 0. and self.training:
            proj = F.dropout(proj, self.dropout)
        return proj


class CrossAttention(nn.Module):
    """Cross-attention from HR tokens (queries) to LR tokens (keys/values).

    Each HR position selectively attends to the relevant LR spatial region.
    QK-norm is applied for training stability, matching the self-attention style.
    The output projection is zero-initialised so every block starts as identity
    and gradually learns to incorporate LR context.
    """

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

    def forward(self, x, context):
        """
        x:       (B, L_hr, embed_dim)  HR queries
        context: (B, L_lr, embed_dim)  LR keys/values
        Returns: (B, L_hr, embed_dim)
        """
        B, L_hr, _ = x.shape
        L_lr = context.shape[1]

        xq = self.wq(x)
        xk = self.wk(context)
        xv = self.wv(context)
        xq, xk = self.qk_norm(xq, xk, xv)

        xq = xq.view(B, L_hr, self.n_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(B, L_lr, self.n_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(B, L_lr, self.n_heads, self.head_dim).transpose(1, 2)

        cross_out = F.scaled_dot_product_attention(
            xq, xk, xv,
            dropout_p=self.dropout if self.training else 0.0,
        )

        cross_out = cross_out.transpose(1, 2).contiguous().view(B, L_hr, -1)
        proj = self.wo(cross_out)

        return proj


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, linear=True, bias=True):
        super().__init__()
        self.eps = eps
        self.linear = linear
        self.add_bias = bias
        if self.linear:
            self.weight = nn.Parameter(torch.ones(dim))
        if self.add_bias:
            self.bias = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.linear:
            output = self.weight * output
        if self.add_bias:
            output = output + self.bias
        return output


class AdaNorm(nn.Module):
    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.norm_final = RMSNorm(x_dim, linear=True, bias=True, eps=1e-5)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(y_dim, x_dim * 2))

    def forward(self, x, y):
        shift, scale = self.mlp(y).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return x


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0., use_cross_attn=False):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.ln1 = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
        self.attn = Attention(dim, heads, dropout=dropout)
        self.ln2 = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)

        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.ln_cross = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
            self.cross_attn = CrossAttention(dim, heads, dropout=dropout)

    def forward(self, x, cond, lr_tokens=None, mask=None):
        """
        DiT-style block w. AdaLN and cross-attn module for lr_tokens:
        x = x + scale * self_attn(modulate(norm(x)))
        x = x + cross_attn(norm(x_lr), x)  # Optional
        x = x + scale * feed_forward(modulate(norm(x)))
        """
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.mlp(cond).chunk(6, dim=1)
        x = x + alpha1.unsqueeze(1) * self.attn(modulate(self.ln1(x), gamma1, beta1), mask=mask)
        if self.use_cross_attn and lr_tokens is not None:
            x = x + self.cross_attn(self.ln_cross(x), lr_tokens)
        x = x + alpha2.unsqueeze(1) * self.ff(modulate(self.ln2(x), gamma2, beta2))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0., use_checkpoint=False, use_cross_attn=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.layers = nn.ModuleList([
            Block(dim, heads, mlp_dim, dropout=dropout, use_cross_attn=use_cross_attn)
            for _ in range(depth)
        ])

    def forward(self, x, cond, lr_tokens=None, mask=None):
        for block in self.layers:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x, cond, lr_tokens, mask, use_reentrant=False)
            else:
                x = block(x, cond, lr_tokens=lr_tokens, mask=mask)
        return x


class MaskTransformer3D(nn.Module):
    """DiT-style bidirectional transformer for 3D MaskGIT and MaskGIT-SR.

    Supports two modes controlled by lr_seq_len:

    Unconditional (lr_seq_len=None):
        A single learned embedding drives all AdaLN modulations, matching the
        original MaskGIT setup. No cross-attention blocks are created.

    SR-conditioned (lr_seq_len=N_lr):
        LR codebook embeddings (B, N_lr, lr_embed_dim) are linearly projected
        to embed_dim and combined with separate learned positional embeddings.
        The resulting LR context serves two roles per block:
          - Mean-pooled -> AdaLN cond vector (global LR style)
          - Full sequence -> cross-attention keys/values (local spatial grounding)

    Adapted from Halton-MaskGIT (2D). Key 3D changes vs. the original:
      - Input is a flat (B, L) token sequence where L = D' * H' * W'
      - pos_emb sized to seq_len rather than input_size**2
      - 2D Conv projection layers removed (not needed for flat codebook sequences)
    """

    def __init__(self, seq_len=4096, embed_dim=768, codebook_size=1024,
                 depth=12, num_heads=16, mlp_ratio=4, dropout=0., register=1,
                 lr_seq_len=None, lr_embed_dim=None,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.codebook_size = codebook_size
        self.mlp_dim = int(embed_dim * mlp_ratio)
        self.lr_seq_len = lr_seq_len

        self.tok_emb = nn.Embedding(codebook_size + 1, embed_dim)  # +1 slot for the mask token
        self.pos_emb = nn.Embedding(seq_len, embed_dim)            # flat positional embedding over D'*H'*W'

        if lr_seq_len is not None:
            # SR mode: project LR embeddings + separate positional table for LR tokens
            lr_in_dim = lr_embed_dim if lr_embed_dim is not None else embed_dim
            self.lr_proj = nn.Linear(lr_in_dim, embed_dim, bias=False)
            self.lr_pos_emb = nn.Embedding(lr_seq_len, embed_dim)
        else:
            # Unconditional mode: single learned vector drives AdaLN
            self.uncond_emb = nn.Embedding(1, embed_dim)

        self.transformer = TransformerEncoder(
            dim=embed_dim,
            depth=depth,
            heads=num_heads,
            mlp_dim=self.mlp_dim,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
            use_cross_attn=lr_seq_len is not None,
        )
        self.last_norm = AdaNorm(x_dim=embed_dim, y_dim=embed_dim)

        # Head weight-tied to tok_emb; mask token slot (index codebook_size) is never a prediction target
        self.head = nn.Linear(embed_dim, codebook_size + 1, bias=False)
        self.head.weight = self.tok_emb.weight

        self.register = register
        if self.register > 0:
            self.reg_tokens = nn.Embedding(self.register, embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

        if self.lr_seq_len is not None:
            nn.init.normal_(self.lr_pos_emb.weight, std=0.02)
        else:
            nn.init.normal_(self.uncond_emb.weight, std=0.02)

        # Zero-out AdaNorm modulation MLP in each block (DiT-style zero init)
        for block in self.transformer.layers:
            nn.init.constant_(block.mlp[1].weight, 0)
            nn.init.constant_(block.mlp[1].bias, 0)
            # Zero-init cross-attention output so each block starts as identity
            if block.use_cross_attn:
                nn.init.constant_(block.cross_attn.wo.weight, 0)

        if self.register > 0:
            nn.init.normal_(self.reg_tokens.weight, std=0.02)

    def _prepare_lr_context(self, lr_tokens):
        """Project and positionally embed LR tokens.

        lr_tokens: (B, N_lr, lr_embed_dim)
        Returns:
            lr_ctx: (B, N_lr, embed_dim)  full sequence for cross-attention
            cond:   (B, embed_dim)         mean-pooled for AdaLN
        """
        lr_pos = torch.arange(lr_tokens.shape[1], device=lr_tokens.device)
        lr_ctx = self.lr_proj(lr_tokens) + self.lr_pos_emb(lr_pos)
        return lr_ctx, lr_ctx.mean(dim=1)

    def forward(self, x, lr_tokens=None, mask=None):
        """
        Args:
            x:         (B, L) int64 token indices; masked positions hold codebook_size
            lr_tokens: (B, N_lr, lr_embed_dim) LR codebook embeddings for SR conditioning,
                       or None for unconditional generation
            mask:      optional attention mask (B, L)
        Returns:
            logits: (B, L, codebook_size + 1)
        """
        B, L = x.shape

        pos = torch.arange(L, device=x.device)
        x = self.tok_emb(x) + self.pos_emb(pos)  # (B, L, embed_dim)

        if lr_tokens is not None:
            lr_ctx, cond = self._prepare_lr_context(lr_tokens)
        else:
            lr_ctx = None
            cond = self.uncond_emb(torch.zeros(B, dtype=torch.long, device=x.device))

        if self.register > 0:
            reg = torch.arange(self.register, device=x.device)
            x = torch.cat([x, self.reg_tokens(reg).expand(B, -1, -1)], dim=1)

        x = self.transformer(x, cond, lr_tokens=lr_ctx, mask=mask)

        # Strip register tokens before final norm and head
        x = x[:, :L].contiguous()

        x = self.last_norm(x, cond)
        return self.head(x)  # (B, L, codebook_size + 1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hr_seq_len  = 16 ** 3
    lr_seq_len  = 4 ** 3
    lr_embed_dim = 256  # matches RQVAE latent_dim

    for size in ["tiny", "small", "base"]:
        embed_dim, depth, num_heads = {
            "tiny":  (384,  6,  6),
            "small": (512, 12,  6),
            "base":  (768, 12, 12),
        }[size]

        code = torch.randint(0, 512, (1, hr_seq_len), device=device)
        lr   = torch.randn(1, lr_seq_len, lr_embed_dim, device=device)

        # Unconditional
        m_uncond = MaskTransformer3D(
            seq_len=hr_seq_len, embed_dim=embed_dim, codebook_size=512,
            depth=depth, num_heads=num_heads, mlp_ratio=4, dropout=0.1,
        ).to(device)
        param_count(f"{size} unconditional", m_uncond)
        logits = m_uncond(code)
        assert logits.shape == (1, hr_seq_len, 513), logits.shape
        print(f"  unconditional logits: {logits.shape}")

        # SR-conditioned
        m_sr = MaskTransformer3D(
            seq_len=hr_seq_len, embed_dim=embed_dim, codebook_size=512,
            depth=depth, num_heads=num_heads, mlp_ratio=4, dropout=0.1,
            lr_seq_len=lr_seq_len, lr_embed_dim=lr_embed_dim,
        ).to(device)
        param_count(f"{size} SR-conditioned", m_sr)
        logits_sr = m_sr(code, lr_tokens=lr)
        assert logits_sr.shape == (1, hr_seq_len, 513), logits_sr.shape
        print(f"  SR-conditioned logits: {logits_sr.shape}")
