"""
RQTransformer3D — Two-Transformer RQ + 3D Volumetric SR

Two factorised transformers following the RQTransformer decomposition:

  Body (spatial): bidirectional transformer over summed depth embeddings.
    Input:  codes summed across D depths → (B, L, E)
    Attn:   fully bidirectional
    Cond:   LR cross-attention + AdaLN from mean-pooled LR tokens
    Output: spatial_ctx (B, L, E)

  Head (depth): lightweight causal AR transformer per spatial position.
    Input:  [SOS (spatial_ctx) | tok_0 | tok_1 | ... | tok_{D-2}] per position → (B*L, D, E)
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
from models.models_3D import PixelUnshuffle3D

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

    def forward(self, x, cond, lr_tokens=None, attn_mask=None):
        for block in self.layers:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x, cond, lr_tokens, attn_mask, use_reentrant=False)
            else:
                x = block(x, cond, lr_tokens=lr_tokens, attn_mask=attn_mask)
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

class RQTransformer3D(nn.Module):
    """
    Two-transformer RQ for 3D volumetric super-resolution.

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
        lr_input_len=None,
        lr_input_dim=None,
        lr_down_factor=1,
        use_checkpoint=False,
        head_emb_vqvae=False,
        cumsum_depth_ctx=False,
        input_embed_dim=None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_rq_depth = n_rq_depth
        self.embed_dim = embed_dim
        self.n_embed = n_embed       # mask token id (same for all depths)
        self.lr_input_len = lr_input_len

        # Use frozen RQVAE codebook vectors (cumsum optional) in place of the learned
        # per-depth tok_embs. Requires code_vectors at forward() and code_emb_fn at sample().
        self.head_emb_vqvae = head_emb_vqvae
        self.cumsum_depth_ctx = bool(cumsum_depth_ctx) and head_emb_vqvae
        if head_emb_vqvae:
            assert input_embed_dim is not None, (
                "input_embed_dim (RQVAE codebook vector dim) must be set when head_emb_vqvae=True."
            )
            self.input_embed_dim = input_embed_dim
            self.head_mlp = nn.Linear(input_embed_dim, embed_dim)

        mlp_dim = int(embed_dim * mlp_ratio)

        # Per-depth token embeddings; index n_embed = [MASK] token
        self.tok_embs = nn.ModuleList([
            nn.Embedding(n_embed + 1, embed_dim) for _ in range(n_rq_depth)
        ])

        # Spatial positional embedding for body input
        self.pos_emb = nn.Embedding(seq_len, embed_dim)

        # Depth positional embedding for head input sequence (D positions)
        self.depth_emb = nn.Embedding(n_rq_depth, embed_dim)

        # Learned SOS for the causal body: position 0 sees only sos (+ LR conditioning),
        # positions 1..L-1 see shifted code embeddings 0..L-2 (autoregressive body).
        self.spatial_sos = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # LR conditioning
        if lr_input_len is not None:
            self.lr_seq_len = lr_input_len // lr_down_factor**3
            self.lr_down = PixelUnshuffle3D(lr_down_factor) if lr_down_factor > 1 else nn.Identity()
            lr_in_dim = lr_input_dim * lr_down_factor**3 if lr_input_dim is not None else embed_dim
            self.lr_proj = nn.Conv3d(lr_in_dim, embed_dim, kernel_size=1, padding=0, stride=1, bias=False)
            self.lr_pos_emb = nn.Embedding(self.lr_seq_len, embed_dim)
        else:
            self.uncond_emb = nn.Embedding(1, embed_dim)

        # Body: bidirectional spatial transformer
        self.body_transformer = BodyTransformer(
            dim=embed_dim, depth=body_depth, heads=num_heads, mlp_dim=mlp_dim,
            dropout=dropout, use_checkpoint=use_checkpoint,
            use_cross_attn=lr_input_len is not None,
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
        # Causal L×L boolean mask for body — lower-triangular, non-persistent
        self.register_buffer(
            'causal_spatial_mask',
            torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)),
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
        nn.init.normal_(self.pos_emb.weight,   std=0.02)
        nn.init.normal_(self.depth_emb.weight, std=0.02)
        nn.init.normal_(self.spatial_sos,      std=0.02)

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
        """lr_tokens: (B, N_lr, lr_embed_dim) → (lr_ctx (B,N_lr,E), cond (B,E))"""
        lr_tokens = self.lr_down(lr_tokens)
        b, c, d, h, w = lr_tokens.shape
        lr_pos = torch.arange(d*h*w, device=lr_tokens.device)
        lr_ctx = self.lr_proj(lr_tokens).view(b, self.embed_dim, d*h*w).transpose(1, 2) + self.lr_pos_emb(lr_pos)
        return lr_ctx, lr_ctx.mean(dim=1)

    # ── Forward (split so sampling can re-use the body pass) ─────────────────

    def _body_forward(self, codes_flat: torch.Tensor, lr_tokens: torch.Tensor = None):
        """Causal body transformer over summed depth embeddings, right-shifted by SOS.

        body_input = [spatial_sos, sum_d(tok_embs[d](codes[:, 0, d])), ...,
                                    sum_d(tok_embs[d](codes[:, L-2, d]))] + pos_emb
        so spatial_ctx[:, s, :] attends causally to sos + codes 0..s-1 — exactly the
        context needed to predict the codes at position s.

        Args:
            codes_flat: (B, L, D) int64
            lr_tokens:  (B, C_lr, Dz, Dy, Dx) pre-encoded LR embeddings, or None.
        Returns:
            spatial_ctx: (B, L, E)
            tok_stack:   (B, L, D, E) per-depth token embeddings (re-used by the head)
            cond:        (B, E) AdaLN conditioning vector
            lr_ctx:      (B, N_lr, E) LR cross-attention tokens, or None
        """
        B, L, D = codes_flat.shape
        pos_emb = self.pos_emb(torch.arange(L, device=codes_flat.device))     # (L, E)

        tok_stack = torch.stack(
            [self.tok_embs[d](codes_flat[:, :, d]) for d in range(D)], dim=2  # (B, L, D, E)
        )
        xs_emb = tok_stack.sum(dim=2)                                         # (B, L, E)

        # Right-shift by one: position 0 = SOS, positions 1..L-1 = summed codes 0..L-2
        sos = self.spatial_sos.expand(B, -1, -1)                              # (B, 1, E)
        body_in = torch.cat([sos, xs_emb[:, :-1, :]], dim=1) + pos_emb        # (B, L, E)

        if lr_tokens is not None:
            lr_ctx, cond = self._prepare_lr_context(lr_tokens)
        else:
            cond = self.uncond_emb(torch.zeros(B, dtype=torch.long, device=codes_flat.device))
            lr_ctx = None

        spatial_ctx = self.body_transformer(body_in, cond, lr_tokens=lr_ctx,
                                            attn_mask=self.causal_spatial_mask)
        spatial_ctx = self.body_norm(spatial_ctx, cond)
        return spatial_ctx, tok_stack, cond, lr_ctx

    def _depth_input(self, tok_stack: torch.Tensor, code_vectors: torch.Tensor = None) -> torch.Tensor:
        """Build the per-depth token stream fed to the head.
        Args:
            tok_stack:    (B, L, D, embed_dim) — used when head_emb_vqvae=False
            code_vectors: (B, L, D, input_embed_dim) — required when head_emb_vqvae=True
        Returns:
            (B, L, D, embed_dim) per-(position, depth) token embeddings
        """
        if not self.head_emb_vqvae:
            return tok_stack
        assert code_vectors is not None, (
            "code_vectors must be provided when head_emb_vqvae=True"
        )
        if self.cumsum_depth_ctx:
            code_vectors = code_vectors.cumsum(dim=2)     # partial-recon at each depth
        return self.head_mlp(code_vectors)                # (B, L, D, embed_dim)

    def _head_forward(self, spatial_ctx: torch.Tensor, depth_input: torch.Tensor):
        """Causal depth transformer, teacher-forced with depth_input[:, :, :-1, :].

        Args:
            spatial_ctx: (B, L, embed_dim)      — body output, used as SOS + AdaLN cond
            depth_input: (B, L, D, embed_dim)   — per-depth head input tokens
                                                  (from _depth_input; either tok_stack or
                                                  MLP-projected codebook vectors)
        Returns:
            logits: list[D × (B, L, n_embed + 1)]
        """
        B, L, D, _ = depth_input.shape
        depth_emb = self.depth_emb(torch.arange(D, device=spatial_ctx.device))  # (D, E)

        # [SOS = spatial_ctx | dep_0 | ... | dep_{D-2}]  → predict code_0..code_{D-1}
        sos = spatial_ctx.view(B, L, 1, -1)
        depth_ctx = torch.cat([sos, depth_input[:, :, :-1, :]], dim=2) + depth_emb
        depth_ctx = depth_ctx.reshape(B * L, D, -1)

        cond_head = spatial_ctx.reshape(B * L, -1)

        # NOTE: each depth is conditioned identically — could instead let cond vary per depth.
        head_out = self.head_transformer(depth_ctx, cond_head, attn_mask=self.causal_depth_mask)
        head_out = self.head_norm(head_out).reshape(B, L, D, -1)
        return [self.heads[d](head_out[:, :, d, :]) for d in range(D)]

    def forward(self, codes: torch.Tensor, lr_tokens: torch.Tensor = None,
                code_vectors: torch.Tensor = None):
        """
        Args:
            codes:        (B, d, h, w, D) int64
            lr_tokens:    (B, C_lr, Dz, Dy, Dx) pre-encoded LR embeddings, or None.
            code_vectors: (B, L, D, input_embed_dim) frozen RQVAE codebook vectors —
                          required iff head_emb_vqvae=True; ignored otherwise.
        Returns:
            logits: list of D tensors, each (B, L, n_embed + 1)
                    logits[d] predicts code at depth d given spatial_ctx + codes 0..d-1.
        """
        B, dz, dy, dx, D = codes.shape
        L = dz * dy * dx
        assert L == self.seq_len and D == self.n_rq_depth
        codes_flat = codes.reshape(B, L, D)

        spatial_ctx, tok_stack, _, _ = self._body_forward(codes_flat, lr_tokens)
        depth_input = self._depth_input(tok_stack, code_vectors)
        return self._head_forward(spatial_ctx, depth_input)

    # ── Autoregressive sampling ───────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        lr_tokens: torch.Tensor = None,
        batch_size: int = 1,
        temperature: float = 1.0,
        top_k: int = None,
        code_emb_fn = None,
    ) -> torch.Tensor:
        """Raster-scan spatial AR + causal depth AR generation (kakaobrain style).

        Because the body is causal + shifted by SOS, spatial_ctx[:, s, :] depends
        only on the SOS and codes 0..s-1 — unfilled trailing positions are inert
        and can hold any value (we use 0).

        Complexity: L body passes + L*D head passes per sample. TODO: add KV-caching
        to the body attention to reduce to O(L) total attn work.

        Args:
            lr_tokens:   (B, C_lr, Dz, Dy, Dx) LR embeddings, or None for unconditional.
            batch_size:  used only when lr_tokens is None.
            temperature: softmax temperature (higher = more random).
            top_k:       if set, restrict to top-k logits before sampling.
            code_emb_fn: callable(codes: (..., D) int64) → (..., D, input_embed_dim)
                         yielding frozen RQVAE codebook vectors per depth. Required
                         iff head_emb_vqvae=True; ignored otherwise.
        Returns:
            codes: (B, L, D) int64 — sampled RQ codes in raster order.
        """
        if self.head_emb_vqvae:
            assert code_emb_fn is not None, (
                "code_emb_fn is required when head_emb_vqvae=True "
                "(pass e.g. vq_model.quantizer.embed_code_with_depth wrapped to stack on dim=-2)."
            )
        device = self.pos_emb.weight.device
        B = lr_tokens.shape[0] if lr_tokens is not None else batch_size
        L, D, V = self.seq_len, self.n_rq_depth, self.n_embed
        depth_emb = self.depth_emb(torch.arange(D, device=device))  # (D, E)

        # Trailing positions are inert under the causal body mask, so plain zeros are safe.
        codes_flat = torch.zeros((B, L, D), dtype=torch.long, device=device)

        for s in range(L):
            spatial_ctx, _, _, _ = self._body_forward(codes_flat, lr_tokens)

            # AR over the D depths at position s using the causal head
            cond_head = spatial_ctx[:, s, :]                                   # (B, E)
            head_input = spatial_ctx[:, s:s+1, :] + depth_emb[0:1]             # (B, 1, E)

            for d in range(D):
                attn_mask = self.causal_depth_mask[:d+1, :d+1]
                h = self.head_transformer(head_input, cond_head, attn_mask=attn_mask)
                h = self.head_norm(h)
                logits = self.heads[d](h[:, -1, :])[:, :V]                     # drop mask slot
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = logits.masked_fill(logits < v[:, [-1]], float('-inf'))
                sampled = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(-1)
                codes_flat[:, s, d] = sampled

                if d + 1 < D:
                    if self.head_emb_vqvae:
                        # Look up frozen codebook vectors for position s up to and including depth d,
                        # then take cumulative sum and project with MLP. Matches _depth_input semantics.
                        cv_s = code_emb_fn(codes_flat[:, s:s+1, :])[:, 0, :, :]   # (B, D, input_embed_dim)
                        if self.cumsum_depth_ctx:
                            vec = cv_s[:, :d+1, :].sum(dim=1)                     # (B, input_embed_dim)
                        else:
                            vec = cv_s[:, d, :]                                   # (B, input_embed_dim)
                        new_tok = self.head_mlp(vec).unsqueeze(1) + depth_emb[d+1:d+2]
                    else:
                        new_tok = self.tok_embs[d](sampled).unsqueeze(1) + depth_emb[d+1:d+2]
                    head_input = torch.cat([head_input, new_tok], dim=1)

        return codes_flat


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hr_spatial = 8
    lr_spatial = 8
    L_hr = hr_spatial**3
    L_lr = lr_spatial**3
    D = 8
    n_embed = 4096
    lr_input_dim = 32
    use_checkpoint = True

    configs = {
        "tiny":  dict(embed_dim=768, body_depth=4, head_depth=4, num_heads=4),
        # "small": dict(embed_dim=384, body_depth=4,  head_depth=4, num_heads=6),
        # "base":  dict(embed_dim=512, body_depth=6, head_depth=6, num_heads=8),
    }

    for name, cfg in configs.items():

        model = RQTransformer3D(
            seq_len=L_hr,
            n_rq_depth=D,
            n_embed=n_embed,
            lr_input_len=L_lr,
            lr_input_dim=lr_input_dim,
            dropout=0.1,
            lr_down_factor=2,
            use_checkpoint=use_checkpoint,
            **cfg,
        ).to(device)
        param_count(f"RQTransformer3D-{name}", model)

        codes_5d = torch.randint(0, n_embed, (2, hr_spatial, hr_spatial, hr_spatial, D), device=device)
        lr_emb = torch.randn(2, lr_input_dim, lr_spatial, lr_spatial, lr_spatial, device=device)
        logits = model(codes_5d, lr_tokens=lr_emb)

        assert len(logits) == D
        assert logits[0].shape == (2, L_hr, n_embed + 1), logits[0].shape
        print(f"[{name}] forward ok — logits[0]: {logits[0].shape}")

        # Verify causal mask shape and content
        mask = model.causal_depth_mask
        assert mask.shape == (D, D) and mask.dtype == torch.bool
        assert mask[0, 0] and not mask[0, 1], "causal mask should be lower-triangular"
        print(f"[{name}] causal_depth_mask ({D}×{D}) ok\n")
