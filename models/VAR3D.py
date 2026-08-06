"""
VAR3D — 3D Volumetric Visual AutoRegressive Transformer.

Predicts multi-scale VQ tokens produced by MSVQVAE3D in coarse-to-fine scale
order. All tokens within a scale are predicted in parallel; a block-causal
attention mask enforces "scale k attends to scales 0..k". Inference is K
transformer passes (one per scale) rather than L.

For SR conditioning, vanilla VAR's class embedding is replaced by mean-pooled
LR features, following the same pattern used in RQTransformer3D:
    - SOS block = mean-pooled LR, tiled to l_0 positions
    - AdaLN cond = mean-pooled LR
    - Optional per-block cross-attention over the full LR token grid

Reference: https://github.com/FoundationVision/VAR/blob/main/models/var.py
"""

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.MaskTransformer3D import RMSNorm, AdaNorm, param_count
from models.RQTransformer3D import BodyTransformer
from models.models_3D import PixelUnshuffle3D


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def _as_dhw(pn) -> Tuple[int, int, int]:
    return (pn, pn, pn) if isinstance(pn, int) else tuple(pn)


def _make_block_causal_mask(patch_nums: List[Tuple[int, int, int]]) -> torch.Tensor:
    """(L, L) bool. mask[i, j] = True iff scale(j) <= scale(i).

    Within a scale, positions can attend bidirectionally (all in the same
    block). Across scales, position at scale k cannot attend to any position
    at scale > k.
    """
    lengths = [pd * ph * pw for (pd, ph, pw) in patch_nums]
    scale_ids = torch.repeat_interleave(
        torch.arange(len(lengths)),
        torch.tensor(lengths),
    )
    q = scale_ids.view(-1, 1)
    k = scale_ids.view(1, -1)
    return k <= q  # (L, L) bool


# ─────────────────────────────────────────────────────────────────────────────
# main model
# ─────────────────────────────────────────────────────────────────────────────
class VAR3D(nn.Module):
    """
    Args:
        Cvae:            codebook vector dim (== MSVQVAE3D.quant_embed_dim)
        vocab_size:      codebook size V (== MSVQVAE3D.n_embed)
        v_patch_nums:    scale schedule matching MSVQVAE3D. int (isotropic) or (D,H,W).
        embed_dim:       transformer hidden dim
        depth:           number of transformer blocks
        num_heads:       attention heads
        mlp_ratio:       FFN hidden multiplier
        dropout:         dropout rate
        lr_input_dim:    channels of the raw LR feature volume (encoder output). None => unconditional.
        lr_input_len:    D*H*W of the raw LR volume (before lr_down). None => unconditional.
        lr_down_factor:  PixelUnshuffle3D factor applied to LR before projection
        use_cross_attn:  add LR cross-attention in every block (in addition to AdaLN)
        use_checkpoint:  gradient checkpointing over trunk blocks
    """
    def __init__(
        self,
        Cvae: int,
        vocab_size: int,
        v_patch_nums: Sequence[Union[int, Tuple[int, int, int]]],
        embed_dim: int = 768,
        depth: int = 16,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        lr_input_dim: Optional[int] = None,
        lr_input_len: Optional[int] = None,
        lr_down_factor: int = 1,
        use_cross_attn: bool = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.Cvae = Cvae
        self.V = vocab_size
        self.embed_dim = embed_dim

        # ---- scales ----
        self.patch_nums: List[Tuple[int, int, int]] = [_as_dhw(pn) for pn in v_patch_nums]
        self.K = len(self.patch_nums)
        self.scale_lens = [pd * ph * pw for (pd, ph, pw) in self.patch_nums]
        self.L = sum(self.scale_lens)
        offs = [0]
        for l in self.scale_lens:
            offs.append(offs[-1] + l)
        self.begin_ends = list(zip(offs[:-1], offs[1:]))  # [(0, l_0), (l_0, l_0+l_1), ...]

        # ---- word embedding for continuous next-scale features ----
        # Fed with x_BLCv_wo_first_l from MSVQVAE3D.quantizer.idxBl_to_var_input.
        self.word_embed = nn.Linear(Cvae, embed_dim, bias=False)

        # ---- pos + level embeddings ----
        self.pos_1LC = nn.Parameter(torch.zeros(1, self.L, embed_dim))
        self.lvl_embed = nn.Embedding(self.K, embed_dim)
        scale_ids = torch.repeat_interleave(
            torch.arange(self.K), torch.tensor(self.scale_lens),
        )
        self.register_buffer('scale_ids', scale_ids, persistent=False)

        # ---- block-causal attention mask (L, L) ----
        self.register_buffer(
            'block_causal_mask',
            _make_block_causal_mask(self.patch_nums),
            persistent=False,
        )

        # ---- LR conditioning ----
        self.has_lr = (lr_input_dim is not None) and (lr_input_len is not None)
        if self.has_lr:
            assert lr_input_len % (lr_down_factor ** 3) == 0, \
                'lr_input_len must be divisible by lr_down_factor**3'
            self.lr_seq_len = lr_input_len // (lr_down_factor ** 3)
            self.lr_down = PixelUnshuffle3D(lr_down_factor) if lr_down_factor > 1 else nn.Identity()
            lr_in_dim = lr_input_dim * (lr_down_factor ** 3)
            self.lr_proj = nn.Conv3d(lr_in_dim, embed_dim, kernel_size=1, bias=False)
            self.lr_pos_emb = nn.Embedding(self.lr_seq_len, embed_dim)
        else:
            self.uncond_emb = nn.Embedding(1, embed_dim)

        # ---- transformer trunk (reuses DiT-AdaLN body blocks) ----
        self.trunk = BodyTransformer(
            dim=embed_dim,
            depth=depth,
            heads=num_heads,
            mlp_dim=int(embed_dim * mlp_ratio),
            dropout=dropout,
            use_checkpoint=use_checkpoint,
            use_cross_attn=(use_cross_attn and self.has_lr),
        )

        # ---- output ----
        self.head_norm = AdaNorm(x_dim=embed_dim, y_dim=embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic)

        nn.init.trunc_normal_(self.pos_1LC, std=0.02)
        nn.init.normal_(self.lvl_embed.weight, std=0.02)
        if self.has_lr:
            nn.init.normal_(self.lr_pos_emb.weight, std=0.02)
        else:
            nn.init.normal_(self.uncond_emb.weight, std=0.02)

        # DiT-style zero-init: AdaLN starts as identity, cross-attn output zero
        for block in self.trunk.layers:
            nn.init.constant_(block.adaln_mlp[1].weight, 0)
            nn.init.constant_(block.adaln_mlp[1].bias, 0)
            if block.use_cross_attn:
                nn.init.constant_(block.cross_attn.wo.weight, 0)
        nn.init.zeros_(self.head.weight)

    # ─────────────────────────── helpers ───────────────────────────

    def _prepare_lr(self, lr_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(B, C_lr, D, H, W) -> (lr_ctx (B, L_lr, E), lr_cond (B, E))."""
        lr = self.lr_down(lr_tokens)
        B, C, D, H, W = lr.shape
        pos = torch.arange(D * H * W, device=lr.device)
        lr_ctx = (
            self.lr_proj(lr).view(B, self.embed_dim, D * H * W).transpose(1, 2)
            + self.lr_pos_emb(pos)
        )
        return lr_ctx, lr_ctx.mean(dim=1)

    def _get_sos_cond(self, lr_tokens: Optional[torch.Tensor], B: int, device):
        """Return (sos_block (B, l_0, E), cond (B, E), lr_ctx (B, L_lr, E) or None)."""
        l_0 = self.scale_lens[0]
        if self.has_lr and lr_tokens is not None:
            lr_ctx, cond = self._prepare_lr(lr_tokens)
        else:
            cond = self.uncond_emb(torch.zeros(B, dtype=torch.long, device=device))
            lr_ctx = None
        sos_block = cond[:, None, :].expand(-1, l_0, -1)
        return sos_block, cond, lr_ctx

    # ─────────────────────────── training forward ───────────────────────────

    def forward(
        self,
        x_BLCv_wo_first_l: Optional[torch.Tensor],
        lr_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x_BLCv_wo_first_l: (B, L - l_0, Cvae) continuous next-scale conditioning,
                               from MSVQVAE3D.quantizer.idxBl_to_var_input(gt_ms_idx_Bl).
                               May be None only if K == 1.
            lr_tokens:         (B, C_lr, D, H, W) or None (unconditional).
        Returns:
            logits: (B, L, V)
        """
        if x_BLCv_wo_first_l is not None:
            B = x_BLCv_wo_first_l.shape[0]
        elif lr_tokens is not None:
            B = lr_tokens.shape[0]
        else:
            raise ValueError('Provide at least one of x_BLCv_wo_first_l or lr_tokens.')
        device = self.pos_1LC.device

        sos_block, cond, lr_ctx = self._get_sos_cond(lr_tokens, B, device)

        if x_BLCv_wo_first_l is not None:
            scale_blocks = self.word_embed(x_BLCv_wo_first_l)   # (B, L - l_0, E)
            x = torch.cat([sos_block, scale_blocks], dim=1)     # (B, L, E)
        else:
            x = sos_block                                        # (B, l_0, E)

        cur_len = x.shape[1]
        x = x + self.pos_1LC[:, :cur_len, :] + self.lvl_embed(self.scale_ids[:cur_len]).unsqueeze(0)

        mask = self.block_causal_mask[:cur_len, :cur_len]
        h = self.trunk(x, cond=cond, lr_tokens=lr_ctx, attn_mask=mask)
        h = self.head_norm(h, cond)
        return self.head(h)                                      # (B, L, V)

    # ─────────────────────────── autoregressive sampling ───────────────────────────

    @torch.no_grad()
    def sample(
        self,
        quantizer,                                   # MSVQVAE3D.quantizer
        lr_tokens: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> List[torch.Tensor]:
        """
        Scale-by-scale AR sampling. K full-sequence transformer passes.
        (KV-cache to reduce this is a TODO.)

        Returns list[LongTensor(B, l_k)] suitable for MSVQVAE3D.decode_multiscale.
        """
        device = self.pos_1LC.device
        B = lr_tokens.shape[0] if lr_tokens is not None else batch_size

        # sanity check that the quantizer schedule matches ours
        assert [_as_dhw(pn) for pn in quantizer.v_patch_nums] == self.patch_nums, \
            'quantizer.v_patch_nums must match VAR3D.patch_nums'

        sos_block, cond, lr_ctx = self._get_sos_cond(lr_tokens, B, device)
        x = sos_block                                              # (B, l_0, E)

        D, H, W = self.patch_nums[-1]
        C = self.Cvae
        f_hat = torch.zeros(B, C, D, H, W, device=device, dtype=torch.float32)

        ms_idx: List[torch.Tensor] = []

        for si in range(self.K):
            cur_len = x.shape[1]
            x_in = (
                x
                + self.pos_1LC[:, :cur_len, :]
                + self.lvl_embed(self.scale_ids[:cur_len]).unsqueeze(0)
            )
            mask = self.block_causal_mask[:cur_len, :cur_len]

            h = self.trunk(x_in, cond=cond, lr_tokens=lr_ctx, attn_mask=mask)
            h = self.head_norm(h, cond)
            logits = self.head(h)                                  # (B, cur_len, V)

            b_si, e_si = self.begin_ends[si]
            logits_si = logits[:, b_si:e_si, :]                    # (B, l_si, V)

            # ---- sampling: temperature, top_k, top_p ----
            logits_si = logits_si / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits_si, min(top_k, logits_si.size(-1)), dim=-1)
                logits_si = torch.where(
                    logits_si < v[..., -1:],
                    torch.full_like(logits_si, float('-inf')),
                    logits_si,
                )
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits_si, descending=True, dim=-1)
                cumprobs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                remove = cumprobs > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                sorted_logits = sorted_logits.masked_fill(remove, float('-inf'))
                logits_si = torch.zeros_like(logits_si).scatter_(-1, sorted_idx, sorted_logits)

            probs = F.softmax(logits_si, dim=-1)                   # (B, l_si, V)
            l_si = e_si - b_si
            idx_si = torch.multinomial(probs.reshape(-1, self.V), num_samples=1).view(B, l_si)
            ms_idx.append(idx_si)

            if si < self.K - 1:
                # decode this scale's tokens, roll into f_hat, downsample to next scale
                pd, ph, pw = self.patch_nums[si]
                h_BCDHW = (
                    quantizer.codebook.embed(idx_si.view(B, pd, ph, pw))
                    .permute(0, 4, 1, 2, 3).contiguous()           # (B, C, pd, ph, pw)
                )
                f_hat, next_cond = quantizer.get_next_autoregressive_input(si, f_hat, h_BCDHW)
                # (B, C, pd_n, ph_n, pw_n) -> (B, l_{si+1}, C)
                next_flat = next_cond.reshape(B, C, -1).transpose(1, 2)
                new_block = self.word_embed(next_flat)             # (B, l_{si+1}, E)
                x = torch.cat([x, new_block], dim=1)

        return ms_idx


# ─────────────────────────────────────────────────────────────────────────────
# smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Match MSVQVAE3D defaults for a 64^3 input -> 8^3 latent config
    v_patch_nums = (1, 2, 3, 4, 6, 8)         # 1+8+27+64+216+512 = 828 tokens
    Cvae = 768
    vocab_size = 4096

    # LR input as raw 1-channel volume at 8^3 (matches RQTransformer3D smoke test style)
    lr_spatial     = 8
    lr_input_dim   = 1
    lr_input_len   = lr_spatial ** 3
    lr_down_factor = 2

    model = VAR3D(
        Cvae=Cvae,
        vocab_size=vocab_size,
        v_patch_nums=v_patch_nums,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
        lr_input_dim=lr_input_dim,
        lr_input_len=lr_input_len,
        lr_down_factor=lr_down_factor,
        use_cross_attn=True,
        use_checkpoint=True,
    ).to(device)
    param_count('VAR3D-base', model)

    # ─── training forward (teacher-forced with dummy features) ───
    B = 2
    L = model.L
    l_0 = model.scale_lens[0]
    x_wo_first = torch.randn(B, L - l_0, Cvae, device=device)     # would come from quantizer.idxBl_to_var_input
    lr = torch.randn(B, lr_input_dim, lr_spatial, lr_spatial, lr_spatial, device=device)

    logits = model(x_wo_first, lr_tokens=lr)
    assert logits.shape == (B, L, vocab_size), logits.shape
    print(f'[VAR3D] forward ok — logits: {tuple(logits.shape)}')

    if torch.cuda.is_available():
        print('Max memory reserved: %0.3f Gb'
              % (torch.cuda.max_memory_reserved() / 1e9))
