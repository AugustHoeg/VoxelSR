"""
RQVARTransformer3D — hybrid RQ + next-scale AR transformer for 3D volumetric SR,
matching the RMSVQVAE3D tokenizer (VAR-style multi-scale residual quantization
with an inner D-depth RQ loop per scale).

It fuses the two transformers already in this repo:
  * InfinityTransformer3D  -> the SPATIAL (outer, next-scale) branch: a prefix-LM,
    block-causal, 3D-RoPE body over [ LR_prefix | SOS | scale_1..scale_{K-1} ],
    consuming the summed-over-depth cumulative f_hat (continuous, Cvae-dim).
  * RQTransformer3DPrefix   -> the DEPTH (inner, next-depth) branch: a shared,
    lightweight causal transformer that, per spatial token, expands the single
    spatial context vector into D residual-codebook predictions.

Two autoregressive axes:
  outer  si = 0..K-1     next-scale (VAR / Infinity)
  inner  d  = 0..D-1     next-depth (RQ)
so sampling takes K * D steps; each inner step predicts one depth for ALL
positions of the current scale in parallel (far cheaper than RQ raster order).

  Spatial branch (body):
    in:   word_embed(summed cumulative f_hat)  (teacher-forced x_BLC_wo_prefix,
          built by ResidualSelfCorrection3D; scale 0 comes from a learned SOS)
    pos:  3D multi-scale axial RoPE + learned pos_start/pos_1LC + lvl_embed
    cond: pure prefix — AdaLN driven by a learned constant (uncond_emb); the only
          LR signal is the prefix tokens. CFG toggles the prefix (null prefix).
    out:  spatial_ctx (B, L_total, C) after an AdaNorm.

  Depth branch (head):
    in:   depth 0 = spatial_ctx (SOS); depth d>=1 = head_mlp(cumulative code
          vectors up to d-1)  (the RMSVQVAE3D codebook vectors, not learned
          token embeddings) + learned depth_emb.
    cond: spatial_ctx per token (RQTransformer3DPrefix convention: spatial_ctx is
          both SOS and AdaLN cond).
    out:  D index logits per token -> list of D tensors (B, L_total, n_embed).

References:
  VAR:      https://github.com/FoundationVision/VAR/blob/main/models/var.py
  Infinity: https://github.com/FoundationVision/Infinity/blob/main/infinity/models/infinity.py
  RQ-VAE:   https://arxiv.org/abs/2203.01941
"""
import math

import torch
from torch import nn

from models.MaskTransformer3D import AdaNorm, RMSNorm, param_count
from models.models_3D import PixelUnshuffle3D
from models.rope import Rope3DMultiScale
from models.InfinityTransformer3D import RoPETransformer3D


class ResBlock(nn.Module):

    def __init__(self, nf: int):
        super().__init__()
        self.conv1 = nn.Conv3d(nf, nf, kernel_size=1)
        self.conv2 = nn.Conv3d(nf, nf, kernel_size=1)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x))) + x


class RQVARTransformer3D(nn.Module):
    """Prefix-conditioned hybrid RQ + next-scale transformer for 3D volumetric SR.

    Args:
        patch_nums:       multi-scale token schedule (== vae.quantizer.v_patch_nums).
        embed_dim:        shared hidden dim C for both transformer branches.
        c_vae:            channel dim of the RMSVQVAE3D latent / codebook vectors.
        n_embed:          codebook size V (per depth codebook).
        n_rq_depth:       number of residual depth codebooks D.
        body_depth:       spatial (next-scale) transformer layers.
        head_depth:       depth (next-depth) transformer layers.
        num_heads:        attention heads (shared).
        mlp_ratio:        FFN hidden-dim multiplier.
        drop_rate:        dropout rate.
        drop_path_rate:   max stochastic-depth rate for the SPATIAL branch (linear
                          per-layer schedule, Infinity-style); the tiny depth branch
                          uses no stochastic depth.
        lr_input_len:     LR token count at encoder resolution (None = unconditional).
        lr_input_dim:     channel dim of incoming LR embeddings.
        lr_down_factor:   extra PixelUnshuffle3D downsample of the LR grid before prefixing.
        rope_theta:       RoPE base frequency.
        rope_norm_coeffs: per-axis (x,y,z) coordinate scale for RoPE frequencies.
        rope_ref_grid:    shared RoPE coordinate frame (default = finest patch_num).
        word_norm:        LayerNorm over c_vae before word_embed (Infinity norm0_ve); default off.
        use_checkpoint:   gradient checkpointing.
    """

    def __init__(
        self,
        patch_nums=(1, 2, 3, 4, 5, 6, 8),
        embed_dim=512,
        c_vae=512,
        n_embed=4096,
        n_rq_depth=4,
        body_depth=16,
        head_depth=4,
        num_heads=16,
        mlp_ratio=4.,
        lr_block_depth=0,
        lr_block_nf=64,
        use_lr_cond=False,
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

        # hyperparameters 
        self.patch_nums = tuple(patch_nums)
        self.K = len(self.patch_nums)
        self.D = self.n_rq_depth = n_rq_depth
        self.pn_dims = [(pn, pn, pn) if isinstance(pn, int) else tuple(pn) for pn in self.patch_nums]
        self.l_k = [pd * ph * pw for (pd, ph, pw) in self.pn_dims]       # tokens per scale
        self.L_total = sum(self.l_k)                                     # predicted sequence length
        self.begins = [sum(self.l_k[:k]) for k in range(self.K)]         # start of scale k in predicted region
        self.l0 = self.l_k[0]

        self.Cvae, self.V = c_vae, n_embed
        self.C = self.embed_dim = embed_dim
        self.body_depth = body_depth
        self.head_depth = head_depth
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.mlp_dim = int(embed_dim * mlp_ratio)
        self.lr_block_depth = lr_block_depth
        self.lr_block_nf = lr_block_nf
        self.use_lr_cond = use_lr_cond
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.use_checkpoint = use_checkpoint
        init_std = math.sqrt(1 / self.C / 3)
        self.init_std = init_std

        #  spatial word (summed-depth vector) embedding 
        self.norm0_ve = nn.LayerNorm(self.Cvae, eps=1e-6) if word_norm else nn.Identity()
        self.word_embed = nn.Linear(self.Cvae, self.C)

        #  depth branch: cumulative code-vector projection + learned depth position 
        self.head_mlp = nn.Linear(self.Cvae, self.C)            # projects Σ_{d'<=d} codebook_{d'}(idx)
        self.depth_emb = nn.Embedding(n_rq_depth, self.C)       # depth is not a spatial axis -> learned pos

        #  conditioning (pure prefix): learned AdaLN constant for the spatial branch 
        self.uncond_emb = nn.Embedding(1, embed_dim)

        #  LR conditioning as a prefix 
        self.lr_input_len = lr_input_len
        if lr_input_len is not None:
            self.n_lr = lr_input_len // lr_down_factor ** 3
            self.lr_shape = tuple(int(round(lr_input_len ** (1. / 3))) for _ in range(3))
            self.prefix_shape = tuple(s // lr_down_factor for s in self.lr_shape)
            self.lr_in_dim = lr_input_dim * lr_down_factor ** 3 if lr_input_dim is not None else embed_dim

            self.lr_down = PixelUnshuffle3D(lr_down_factor) if lr_down_factor > 1 else nn.Identity()
            self.lr_proj = nn.Conv3d(self.lr_in_dim, embed_dim, kernel_size=1, bias=False)

            lr_block = []
            if self.lr_block_depth > 0 and lr_input_dim is not None:
                lr_block.append(nn.Conv3d(lr_input_dim, lr_block_nf, kernel_size=1))
                for i in range(self.lr_block_depth):
                    lr_block.append(ResBlock(lr_block_nf))
                lr_block.append(PixelUnshuffle3D(lr_down_factor) if lr_down_factor > 1 else nn.Identity())
                lr_block.append(nn.Conv3d(lr_block_nf * lr_down_factor ** 3, embed_dim, kernel_size=1, bias=False))
                self.lr_block = nn.Sequential(*lr_block)
            else:
                self.lr_block = None

            # learned null prefix for classifier-free guidance
            self.null_lr_prefix = nn.Parameter(torch.empty(1, self.n_lr, self.C))

            self.lr_pool = nn.Linear(embed_dim, embed_dim) if use_lr_cond else None
        else:
            self.n_lr = 0
            self.lr_shape = self.prefix_shape = None
            self.lr_block = None
            self.lr_pool = None

        self.context_token = self.n_lr + self.l0   # LR + SOS region (level 0)

        #  learned absolute position embeddings (coexist with RoPE, as in VARSR) 
        self.pos_start = nn.Parameter(torch.empty(1, self.l0, self.C))        # SOS block
        self.pos_1LC = nn.Parameter(torch.empty(1, self.L_total, self.C))     # predicted region
        self.lvl_embed = nn.Embedding(self.K, self.C)                         # per-scale level embedding

        #  3D axial RoPE over [ LR_prefix | scale_0 .. scale_{K-1} ] (spatial branch only) 
        self.rope = Rope3DMultiScale(
            self.head_dim, patch_nums=self.pn_dims,
            lr_shape=(self.prefix_shape if self.n_lr > 0 else None),
            ref_grid=rope_ref_grid, theta=rope_theta, norm_coeffs=rope_norm_coeffs,
        )

        #  attention masks (bool, True = attention allowed) 
        self._build_masks()

        #  spatial (next-scale) transformer: block-causal prefix-LM body 
        self.spatial_blocks = RoPETransformer3D(
            dim=self.C, depth=self.body_depth, heads=self.num_heads, mlp_dim=self.mlp_dim,
            dropout=self.drop_rate, drop_path_rate=self.drop_path_rate,
            use_checkpoint=self.use_checkpoint,
        )
        self.body_norm = AdaNorm(x_dim=embed_dim, y_dim=embed_dim)   # normalise spatial_ctx (cond = uncond_emb)

        #  depth (next-depth) transformer: shared causal head, no stochastic depth 
        self.depth_blocks = RoPETransformer3D(
            dim=self.C, depth=self.head_depth, heads=self.num_heads, mlp_dim=self.mlp_dim,
            dropout=self.drop_rate, drop_path_rate=0.0, use_checkpoint=self.use_checkpoint,
        )
        self.head_norm = RMSNorm(embed_dim, linear=True, bias=False, eps=1e-5)

        #  per-depth prediction heads (independent; code vectors replace learned tok embeds) 
        self.heads = nn.ModuleList([nn.Linear(self.C, n_embed, bias=False) for _ in range(n_rq_depth)])

        self._init_weights()

    #  masks 

    def _build_masks(self):
        """Spatial block-causal + prefix-LM mask (Infinity), depth causal mask (RQ)."""
        # spatial: LR_prefix + SOS at level 0 (bidirectional, prefix-LM); scales 1..K-1 at
        # levels 1..K-1. Token i attends j iff level_i >= level_j (VAR's d >= dT).
        T = self.n_lr + self.L_total
        levels = torch.zeros(T, dtype=torch.long)
        ptr = self.context_token                         # n_lr + l0 already at level 0
        for k in range(1, self.K):
            levels[ptr:ptr + self.l_k[k]] = k
            ptr += self.l_k[k]
        d = levels.view(T, 1)
        self.register_buffer("spatial_mask", (d >= d.transpose(0, 1)), persistent=False)

        # per-scale level index for the predicted region (lvl_embed gather)
        lvl_1L = torch.cat([torch.full((l,), k, dtype=torch.long)
                            for k, l in enumerate(self.l_k)])  # (L_total,)
        self.register_buffer("lvl_1L", lvl_1L, persistent=False)

        # depth: causal D x D (each depth attends to itself and shallower depths)
        self.register_buffer(
            "depth_mask", torch.tril(torch.ones(self.D, self.D, dtype=torch.bool)), persistent=False,
        )

    #  initialisation 

    def _init_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=self.init_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic)

        nn.init.trunc_normal_(self.lvl_embed.weight, std=self.init_std)
        nn.init.trunc_normal_(self.depth_emb.weight, std=self.init_std)
        nn.init.trunc_normal_(self.uncond_emb.weight, std=self.init_std)
        nn.init.trunc_normal_(self.pos_start, std=self.init_std)
        nn.init.trunc_normal_(self.pos_1LC, std=self.init_std)
        if self.n_lr > 0:
            nn.init.trunc_normal_(self.null_lr_prefix, std=self.init_std)

        # zero-init the shallow LR enrichment so the prefix starts == lr_block_depth=0
        if self.lr_block is not None:
            nn.init.zeros_(self.lr_block[-1].weight)

        # zero-init pooled-LR conditioning so cond starts == uncond_emb (pure-prefix baseline)
        if self.lr_pool is not None:
            nn.init.zeros_(self.lr_pool.weight)
            nn.init.zeros_(self.lr_pool.bias)

        # DiT-style zero-init: AdaLN starts as identity in both branches + body_norm.
        for block in list(self.spatial_blocks.layers) + list(self.depth_blocks.layers):
            nn.init.constant_(block.adaln_mlp[1].weight, 0)
            nn.init.constant_(block.adaln_mlp[1].bias, 0)
        nn.init.constant_(self.body_norm.mlp[1].weight, 0)
        nn.init.constant_(self.body_norm.mlp[1].bias, 0)

    #  prefix / SOS helpers (spatial branch) 

    def _lr_prefix(self, lr_tokens: torch.Tensor) -> torch.Tensor:
        """(B, C_lr, Dz, Dy, Dx) → (B, n_lr, C). RoPE handles position."""
        feat = self.lr_proj(self.lr_down(lr_tokens))
        if self.lr_block is not None:
            feat = feat + self.lr_block(lr_tokens)
        return feat.view(feat.shape[0], self.C, self.n_lr).transpose(1, 2)

    def _build_prefix(self, lr_tokens, B):
        """Prefix tokens (B, n_lr, C). lr_tokens=None → learned null prefix (CFG uncond)."""
        if self.n_lr == 0:
            return torch.empty(B, 0, self.C, device=self.pos_1LC.device, dtype=self.pos_1LC.dtype)
        if lr_tokens is None:
            return self.null_lr_prefix.expand(B, -1, -1)
        return self._lr_prefix(lr_tokens)

    def _cond(self, prefix: torch.Tensor) -> torch.Tensor:
        """AdaLN cond (B, C) = learned uncond_emb + zero-init pooled-LR global vector.

        prefix: (B, n_lr, C). With lr_pool zero-init this equals uncond_emb; under CFG
        the null branch (null_lr_prefix) yields a consistent unconditional cond.
        """
        B = prefix.shape[0]
        base = self.uncond_emb(prefix.new_zeros(B, dtype=torch.long))
        if self.n_lr == 0 or self.lr_pool is None:
            return base
        return base + self.lr_pool(prefix.mean(dim=1))

    def _add_lvl_pos(self, tok, si):
        """Add scale-`si` level embedding and absolute pos_1LC slice. tok: (B, l_si, C)."""
        b0, l = self.begins[si], self.l_k[si]
        lvl = self.lvl_embed(torch.full((l,), si, device=tok.device, dtype=torch.long))  # (l, C)
        return tok + lvl.unsqueeze(0) + self.pos_1LC[:, b0:b0 + l]

    def _sos_block(self, cond, B):
        """SOS = cond (uncond_emb) expanded to l0 tokens + pos_start + lvl/pos (scale 0)."""
        sos = cond.unsqueeze(1).expand(B, self.l0, -1) + self.pos_start
        return self._add_lvl_pos(sos, 0)

    #  forward (teacher-forced)

    def forward_spatial(self, x_BLC_wo_prefix: torch.Tensor, lr_tokens: torch.Tensor = None) -> torch.Tensor:
        """Next-scale body over [ LR_prefix | SOS | scales 1..K-1 ] → spatial_ctx.

        Args:
            x_BLC_wo_prefix: (B, L_total - l0, Cvae) summed-depth cumulative f_hat
                             (teacher-forced; ResidualSelfCorrection3D.flip_requant).
            lr_tokens:       (B, C_lr, Dz, Dy, Dx) LR embeddings, or None (→ null prefix).
        Returns:
            spatial_ctx: (B, L_total, C)
        """
        x_BLC_wo_prefix = x_BLC_wo_prefix.float()
        B = x_BLC_wo_prefix.shape[0]

        prefix = self._build_prefix(lr_tokens, B)                                        # (B, n_lr, C)
        cond = self._cond(prefix)                                                        # (B, C) uncond_emb + pooled LR
        sos = cond.unsqueeze(1).expand(B, self.l0, -1) + self.pos_start                  # (B, l0, C)

        x_BLC = torch.cat([sos, self.word_embed(self.norm0_ve(x_BLC_wo_prefix))], dim=1)  # (B, L_total, C)
        x_BLC = x_BLC + self.lvl_embed(self.lvl_1L).unsqueeze(0) + self.pos_1LC

        x_BLC = torch.cat([prefix, x_BLC], dim=1)                                        # (B, n_lr + L_total, C)

        x_BLC = self.spatial_blocks(x_BLC, cond, attn_mask=self.spatial_mask, rope=self.rope, rope_offset=0)
        return self.body_norm(x_BLC[:, self.n_lr:, :], cond)                             # (B, L_total, C)

    def forward_depth(self, spatial_ctx: torch.Tensor, code_vectors: torch.Tensor):
        """Causal depth transformer per token, teacher-forced on cumulative code vectors.

        Args:
            spatial_ctx:  (B, L_total, C) spatial branch output (SOS + AdaLN cond).
            code_vectors: (B, L_total, D, Cvae) per-depth (non-cumulative) GT code vectors.
        Returns:
            logits: list of D tensors, each (B, L_total, n_embed).
        """
        B, L, _ = spatial_ctx.shape
        D = self.D
        depth_emb = self.depth_emb(torch.arange(D, device=spatial_ctx.device))           # (D, C)

        depth_input = self.head_mlp(code_vectors.cumsum(dim=2))                           # (B, L, D, C), [d]=cumsum≤d
        sos = spatial_ctx.view(B, L, 1, -1)                                              # depth-0 SOS
        depth_ctx = torch.cat([sos, depth_input[:, :, :-1, :]], dim=2) + depth_emb        # (B, L, D, C)
        depth_ctx = depth_ctx.reshape(B * L, D, -1)

        cond_head = spatial_ctx.reshape(B * L, -1)                                       # per-token AdaLN cond
        h = self.depth_blocks(depth_ctx, cond_head, attn_mask=self.depth_mask)
        h = self.head_norm(h).reshape(B, L, D, -1)
        return [self.heads[d](h[:, :, d, :]) for d in range(D)]

    def forward(self, x_BLC_wo_prefix: torch.Tensor, code_vectors: torch.Tensor,
                lr_tokens: torch.Tensor = None):
        """
        Args:
            x_BLC_wo_prefix: (B, L_total - l0, Cvae)  spatial teacher input.
            code_vectors:    (B, L_total, D, Cvae)    depth teacher stream (per-depth GT vectors).
            lr_tokens:       (B, C_lr, Dz, Dy, Dx)    LR embeddings, or None.
        Returns:
            logits: list of D tensors, each (B, L_total, n_embed).
        """
        spatial_ctx = self.forward_spatial(x_BLC_wo_prefix, lr_tokens)
        return self.forward_depth(spatial_ctx, code_vectors)

    #  autoregressive sampling 

    def _set_kv_cache(self, enable: bool):
        """Toggle the KV cache on the SPATIAL branch only (depth branch is tiny / recomputed)."""
        for block in self.spatial_blocks.layers:
            block.attn.kv_caching(enable)

    def _sample_from_logits(self, logits, temperature, top_k=None, top_p=None):
        """(N, V) logits → (N,) sampled indices with temperature / top-k / top-p."""
        if temperature <= 0:
            return logits.argmax(dim=-1)
        logits = logits / temperature
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            logits = logits.masked_fill(logits < v[:, [-1]], float('-inf'))
        if top_p is not None and 0 < top_p < 1:
            s_logits, s_idx = torch.sort(logits, descending=True, dim=-1)
            cum = s_logits.softmax(dim=-1).cumsum(dim=-1)
            remove = (cum - s_logits.softmax(dim=-1)) > top_p          # keep at least top-1
            s_logits = s_logits.masked_fill(remove, float('-inf'))
            logits = torch.full_like(logits, float('-inf')).scatter(-1, s_idx, s_logits)
        probs = logits.softmax(dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _sample_scale_depths(self, q, spatial_ctx, temperature, top_k, top_p, cfg_t=None):
        """Inner next-depth AR for every position of one scale, in parallel.

        spatial_ctx: (B, l, C) spatial_ctx per position, or (2*B, l, C) [cond; uncond] when
                 cfg_t is not None (logits mixed per depth, sampled code shared).
        Returns:
            idx:     (Bp, D) sampled depth indices
            code_embed_sum: (Bp, Cvae) summed-over-depth sampled codebook embeddings
        """
        B, l, C = spatial_ctx.shape  # (R*B, l, C)
        D = self.D
        depth_emb = self.depth_emb(torch.arange(self.D, device=spatial_ctx.device))      # (D, C)

        depth_ctx = spatial_ctx.reshape(B * l, C).unsqueeze(1) + depth_emb[0:1]  # (N, 1, C)  depth-0 SOS
        cond_head = spatial_ctx.reshape(B * l, C)                                # (N, C) AdaLN cond

        Bp = depth_ctx.shape[0] if cfg_t is None else depth_ctx.shape[0] // 2
        code_embed_sum = torch.zeros((Bp, self.Cvae), device=spatial_ctx.device) # (Bp, Cvae)

        idx_list = []
        for d in range(D):
            h = self.depth_blocks(depth_ctx, cond_head, attn_mask=self.depth_mask[:d + 1, :d + 1])
            logits = self.heads[d](self.head_norm(h)[:, -1, :])                 # (N, V)
            if cfg_t is not None:
                logits = (1 + cfg_t) * logits[:Bp] - cfg_t * logits[Bp:]        # (Bp, V)
            sampled_idx = self._sample_from_logits(logits, temperature, top_k, top_p)  # (Bp,)
            idx_list.append(sampled_idx)
            code_embed_sum = code_embed_sum + q.codebooks[d].embed(sampled_idx)                   # (Bp, Cvae)
            if d + 1 < D:
                next_token_map = (self.head_mlp(code_embed_sum) + depth_emb[d + 1:d + 2]).unsqueeze(1)  # (Bp, 1, C)
                if cfg_t is not None:
                    next_token_map = next_token_map.repeat(2, 1, 1)                                   # share sampled code across branches
                depth_ctx = torch.cat([depth_ctx, next_token_map], dim=1)
        return torch.stack(idx_list, dim=-1), code_embed_sum                           # (Bp, D), (Bp, Cvae)

    @torch.no_grad()
    def autoregressive_infer(self, vae, lr_tokens, temperature=1.0, top_k=None, top_p=None):
        """Next-scale × next-depth AR generation WITHOUT classifier-free guidance.

        Returns:
            ms_idx_Bl: list of K tensors (B, l_k, D) → vae.decode_multiscale(ms_idx_Bl).
        """
        return self._infer(vae, lr_tokens, cfg=0.0, temperature=temperature, top_k=top_k, top_p=top_p, use_cfg=False)

    @torch.no_grad()
    def autoregressive_infer_cfg(self, vae, lr_tokens, cfg=1.5, temperature=1.0, top_k=None, top_p=None):
        """Next-scale × next-depth AR generation with classifier-free guidance.

        VAR schedule t = cfg * si/(K-1) applied to the depth logits at each scale; the
        batch is doubled (real LR prefix vs learned null prefix). Returns the same
        list[(B, l_k, D)] as autoregressive_infer.
        """
        return self._infer(vae, lr_tokens, cfg=cfg, temperature=temperature, top_k=top_k, top_p=top_p, use_cfg=True)

    @torch.no_grad()
    def _infer(self, vae, lr_tokens, cfg, temperature, top_k, top_p, use_cfg):
        assert self.n_lr > 0, "sampling assumes LR tokens are always provided"
        q = vae.quantizer
        B = lr_tokens.shape[0]
        device = lr_tokens.device
        R = 2 if use_cfg else 1  # branch replication

        lvl_pos = self.lvl_embed(self.lvl_1L).unsqueeze(0) + self.pos_1LC  # (1, L_total, C)

        # LR token prefix
        if use_cfg:
            prefix = torch.cat([self._lr_prefix(lr_tokens), self.null_lr_prefix.expand(B, -1, -1)], dim=0)
        else:
            prefix = self._lr_prefix(lr_tokens)                                 # (R*B, n_lr, C)

        # Cond (SOS + AdaLN): uncond_emb + pooled LR, per branch (null branch pools null prefix)
        cond = self._cond(prefix)                                               # (R*B, C)

        # SOS = cond + pos_start + lvl_pos[:, :l0].
        sos = cond.unsqueeze(1).expand(R * B, self.l0, -1) + self.pos_start + lvl_pos[:, :self.l0]

        # First input = [ LR_prefix | SOS ]
        next_token_map = torch.cat([prefix, sos], dim=1)       # (R*B, n_lr + l0, C)

        f_hat = torch.zeros(B, self.Cvae, *self.pn_dims[-1], device=device)
        ms_idx_Bl = []
        cur_L = 0

        self._set_kv_cache(True)
        for si in range(self.K):
            pd, ph, pw = self.pn_dims[si]
            l = self.l_k[si]

            ratio = si / (self.K - 1) if self.K > 1 else 1
            cfg_t = cfg * ratio if use_cfg else None

            # Spatial branch: next-scale prediction for scale si
            attn_mask = self.spatial_mask[:self.context_token, :self.context_token] if si == 0 else None
            rope_offset = 0 if si == 0 else self.n_lr + self.begins[si]
            x = self.spatial_blocks(next_token_map, cond, attn_mask=attn_mask, rope=self.rope, rope_offset=rope_offset)
            spatial_ctx = self.body_norm(x[:, -l:, :], cond)                    # (R*B, l, C)

            # Depth branch: inner AR over D for all l positions in parallel.
            idx_flat, code_embed_sum = self._sample_scale_depths(q, spatial_ctx, temperature, top_k, top_p, cfg_t=cfg_t)
            ms_idx_Bl.append(idx_flat.reshape(B, l, self.D))

            # Summed-depth code (scale-si resolution), accumulate running f_hat
            q_BCDHW = code_embed_sum.reshape(B, pd, ph, pw, self.Cvae).permute(0, 4, 1, 2, 3).contiguous()
            f_hat, next_scale = q.get_next_autoregressive_input(si, f_hat, q_BCDHW)
            cur_L += l
            if si == self.K - 1:
                break

            # Next-scale VAR input: cumulative latent downsampled to scale si+1, with added word embed + lvl_pos
            next_scale = next_scale.reshape(B, self.Cvae, -1).permute(0, 2, 1)          # (B, l_{si+1}, Cvae)
            next_token_map = self.word_embed(self.norm0_ve(next_scale)) + lvl_pos[:, cur_L:cur_L + self.l_k[si + 1]]
            if use_cfg:
                next_token_map = next_token_map.repeat(2, 1, 1)                 # share HR input across branches
        self._set_kv_cache(False)

        return ms_idx_Bl


#  Quick smoke-test 

if __name__ == "__main__":
    import torch.nn.functional as F
    from models.RMSVQVAE3D import RMSVQVAE3D
    from models.residual_self_correction_3d import ResidualSelfCorrection3D

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    patch_size = 64
    patch_nums = (1, 2, 3, 4, 5, 6, 8)           # finest == encoder latent (down_factor 8)
    c_vae = 512
    n_embed = 4096
    n_rq_depth = 4
    lr_spatial = 16
    lr_input_dim = 1
    lr_down_factor = 2

    vae = RMSVQVAE3D(
        in_channels=1,
        latent_dim=c_vae,
        quant_embed_dim=c_vae,
        n_embed=n_embed,
        channels_enc=[64, 64, 256, 512, 512],   # down_factor = 8 -> latent 8^3
        channels_dec=[512, 512, 256, 64, 64],
        resolution=patch_size,
        num_res_blocks_enc=2,
        num_res_blocks_dec=4,
        v_patch_nums=patch_nums,
        n_rq_depth=n_rq_depth,
        quant_resi=0.5,
        use_prog_quant_resi=True,
        using_znorm=True,
        use_stochastic_depth=False,
        keep_last_quant=True,
        skip_attn=False,
        use_checkpoint=False,
    ).to(device).eval()

    model = RQVARTransformer3D(
        patch_nums=patch_nums,
        embed_dim=768,
        c_vae=c_vae,
        n_embed=n_embed,
        n_rq_depth=n_rq_depth,
        body_depth=4,
        head_depth=2,
        num_heads=8,
        mlp_ratio=4.0,
        lr_block_depth=2,  # exercise the 1x1x1 ResBlock LR enrichment branch in the smoke test
        lr_block_nf=32,    # ResBlock width
        use_lr_cond=True,  # exercise the pooled-LR AdaLN cond path in the smoke test
        drop_rate=0.0,
        drop_path_rate=0.1,
        lr_input_len=lr_spatial ** 3,
        lr_input_dim=lr_input_dim,
        lr_down_factor=lr_down_factor,
        rope_theta=10000,
        use_checkpoint=False,
    ).to(device)

    param_count("RQVARTransformer3D", model)
    print(f"n_lr={model.n_lr}, L_total={model.L_total}, D={model.D}, "
          f"spatial_mask={tuple(model.spatial_mask.shape)}, depth_mask={tuple(model.depth_mask.shape)}, "
          f"rope rot_dim={model.rope.rot_dim}/{model.head_dim}")

    B = 2
    x_hr = torch.randn(B, 1, patch_size, patch_size, patch_size, device=device)
    lr_emb = torch.randn(B, lr_input_dim, lr_spatial, lr_spatial, lr_spatial, device=device)

    #  teacher-forcing inputs + GT depth targets via Residual Self-Correction 
    rsc = ResidualSelfCorrection3D(vae, noise_apply_layers=0, noise_apply_strength=0.0)
    raw = vae.encode(x_hr)
    x_BLC_wo_prefix, code_vectors, gt_ms_idx_Bl = rsc.flip_requant(raw, device)
    assert x_BLC_wo_prefix.shape == (B, model.L_total - model.l0, c_vae), x_BLC_wo_prefix.shape
    assert code_vectors.shape == (B, model.L_total, n_rq_depth, c_vae), code_vectors.shape

    #  forward + per-depth CE loss 
    logits = model(x_BLC_wo_prefix, code_vectors, lr_tokens=lr_emb)            # list D of (B, L_total, V)
    assert len(logits) == n_rq_depth and logits[0].shape == (B, model.L_total, n_embed), logits[0].shape
    targets = torch.cat(gt_ms_idx_Bl, dim=1)                                   # (B, L_total, D)
    loss = sum(F.cross_entropy(logits[d].reshape(-1, n_embed), targets[:, :, d].reshape(-1))
               for d in range(n_rq_depth)) / n_rq_depth
    print(f"forward ok - {len(logits)} depth heads, logits[0] {tuple(logits[0].shape)}, CE loss {loss.item():.4f}")

    #  clean teacher-forcing parity: RSC(noise=0) == VAE clean helpers 
    gt_clean = vae.encode_multiscale(x_hr)                                     # list K of (B, l_k, D)
    xin_clean = vae.quantizer.idxBl_to_var_input(gt_clean)
    cv_clean = vae.quantizer.idx_to_depth_code_vectors(gt_clean)
    assert torch.allclose(xin_clean, x_BLC_wo_prefix, atol=1e-4), "RSC(noise=0) spatial input != clean helper"
    assert torch.allclose(cv_clean, code_vectors, atol=1e-4), "RSC(noise=0) code vectors != clean helper"
    assert all(torch.equal(a, b) for a, b in zip(gt_clean, gt_ms_idx_Bl)), "RSC(noise=0) GT != clean helper"
    print("clean teacher-forcing parity ok (RSC noise=0 matches VAE helpers)")

    model.eval()
    with torch.inference_mode():
        #  spatial KV-cache equivalence (cached per-scale == single full forward) 
        full_ctx = model.forward_spatial(x_BLC_wo_prefix, lr_tokens=lr_emb)    # (B, L_total, C)

        model._set_kv_cache(True)
        lvl_pos = model.lvl_embed(model.lvl_1L).unsqueeze(0) + model.pos_1LC
        prefix = model._lr_prefix(lr_emb)
        condB = model._cond(prefix)
        sos = condB.unsqueeze(1).expand(B, model.l0, -1) + model.pos_start + lvl_pos[:, :model.l0]
        m = model.context_token
        out = model.spatial_blocks(torch.cat([prefix, sos], dim=1), condB,
                                   attn_mask=model.spatial_mask[:m, :m], rope=model.rope, rope_offset=0)
        cached = [model.body_norm(out[:, -model.l0:, :], condB)]
        tf = model.norm0_ve(x_BLC_wo_prefix.float())
        for si in range(model.K - 1):
            b0 = model.begins[si + 1] - model.l0
            blk = model.word_embed(tf[:, b0:b0 + model.l_k[si + 1], :]) + lvl_pos[:, model.begins[si + 1]:model.begins[si + 1] + model.l_k[si + 1]]
            out = model.spatial_blocks(blk, condB, attn_mask=None, rope=model.rope,
                                       rope_offset=model.n_lr + model.begins[si + 1])
            cached.append(model.body_norm(out, condB))
        model._set_kv_cache(False)
        cached = torch.cat(cached, dim=1)
        max_err = (cached - full_ctx).abs().max().item()
        print(f"spatial kv-cache max abs-diff vs full forward: {max_err:.2e}")
        assert torch.allclose(cached, full_ctx, atol=1e-4), f"spatial KV-cache diverged ({max_err:.2e})"
        print("spatial kv-cache equivalence ok")

        #  depth AR equivalence: incremental (GT-fed) == teacher-forced forward_depth 
        tf_logits = model.forward_depth(full_ctx, code_vectors)                # list D of (B, L_total, V)
        Bp = B * model.L_total
        spatial_ctx_flat = full_ctx.reshape(Bp, model.C)
        cv_flat = code_vectors.reshape(Bp, model.D, c_vae)
        depth_emb = model.depth_emb(torch.arange(model.D, device=device))
        head_input = spatial_ctx_flat.unsqueeze(1) + depth_emb[0:1]
        inc = []
        cum = torch.zeros(Bp, c_vae, device=device)
        for d in range(model.D):
            h = model.depth_blocks(head_input, spatial_ctx_flat, attn_mask=model.depth_mask[:d + 1, :d + 1])
            inc.append(model.heads[d](model.head_norm(h)[:, -1, :]).reshape(B, model.L_total, n_embed))
            cum = cum + cv_flat[:, d, :]                                        # feed GT cumulative vectors
            if d + 1 < model.D:
                head_input = torch.cat([head_input, (model.head_mlp(cum) + depth_emb[d + 1:d + 2]).unsqueeze(1)], dim=1)
        derr = max((inc[d] - tf_logits[d]).abs().max().item() for d in range(model.D))
        print(f"depth AR max abs-diff vs teacher-forced: {derr:.2e}")
        assert all(torch.allclose(inc[d], tf_logits[d], atol=1e-4) for d in range(model.D)), f"depth AR diverged ({derr:.2e})"
        print("depth AR equivalence ok")

        #  sampling (cfg + no-cfg) → decode 
        for tag, fn in [("cfg", lambda: model.autoregressive_infer_cfg(vae, lr_emb, cfg=2.0, temperature=1.0, top_k=100)),
                        ("no-cfg", lambda: model.autoregressive_infer(vae, lr_emb, temperature=1.0, top_k=100))]:
            ms_idx = fn()
            assert len(ms_idx) == model.K
            for k, c in enumerate(ms_idx):
                assert c.shape == (B, model.l_k[k], n_rq_depth), (k, c.shape)
            x_rec = vae.decode_multiscale(ms_idx)
            print(f"sample ({tag}) ok - {len(ms_idx)} scales, decoded {tuple(x_rec.shape)}")
