"""
Multi-Scale VQ-VAE for 3D volumes.

Port of VAR's 2D VectorQuantizer2 (https://github.com/FoundationVision/VAR)
to 3D. Residual quantization is applied across *spatial scales* (from a small
coarse grid up to the full latent grid) using a single shared codebook -
contrast to RQVAE3D, which does residual quantization across *codebook depth*
at a single spatial scale.

The codebook implementation (VQEmbedding, EMA + dead-code restart) is reused
from RQVAE3D so the SR comparison isolates the residual-axis choice.

Sequence length per volume for the transformer = sum(pd_k * ph_k * pw_k).
Defaults picked for a 64^3 input -> 8^3 latent (down_factor=8).
"""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.basic_vae import EncoderV3 as Encoder
from models.basic_vae import DecoderV3 as Decoder
from models.RQVAE3D import VQEmbedding
from utils.utils_3D_image import numel


def _as_dhw(pn) -> Tuple[int, int, int]:
    """Accept int (isotropic) or a 3-tuple (D, H, W)."""
    return (pn, pn, pn) if isinstance(pn, int) else tuple(pn)


# ---------------------------------------------------------------------------
# Phi refinement convs (per-scale 3x3x3 residual convs)
# ---------------------------------------------------------------------------
class Phi3D(nn.Conv3d):
    """(1 - r) * x + r * conv(x). Small refine applied AFTER upsampling each scale."""
    def __init__(self, embed_dim, quant_resi):
        super().__init__(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        self.resi_ratio = abs(quant_resi)

    def forward(self, x):
        return x.mul(1 - self.resi_ratio) + super().forward(x).mul_(self.resi_ratio)


class PhiShared3D(nn.Module):
    """One Phi for every scale."""
    def __init__(self, qresi: Phi3D):
        super().__init__()
        self.qresi = qresi

    def __getitem__(self, _):
        return self.qresi


class PhiPartiallyShared3D(nn.Module):
    """K Phi's, resolved to the nearest scale by tick on [0, 1]."""
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = (np.linspace(1/3/K, 1 - 1/3/K, K) if K == 4
                      else np.linspace(1/2/K, 1 - 1/2/K, K))

    def __getitem__(self, at_from_0_to_1: float) -> Phi3D:
        return self.qresi_ls[int(np.argmin(np.abs(self.ticks - at_from_0_to_1)))]

    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'


# ---------------------------------------------------------------------------
# Multi-scale residual quantization bottleneck
# ---------------------------------------------------------------------------
class MultiScaleBottleneck3D(nn.Module):
    """
    Args:
        vocab_size:          codebook size V (shared across all scales)
        Cvae:                codebook embedding dim C
        v_patch_nums:        list of scales, small -> large. Each entry is an int
                             (isotropic) or a (D, H, W) tuple. The LAST scale must
                             equal the encoder output resolution.
        beta:                commitment loss weight
        using_znorm:         if True, use cosine similarity in nearest-neighbour lookup
        quant_resi:          residual ratio for Phi. 0.5 => 0.5*conv(x) + 0.5*x.
                             Set to 0 to disable Phi (Phi becomes Identity).
        share_quant_resi:    1 = single shared Phi; N>1 = N Phi's mapped by tick;
                             0 = one Phi per scale (heavy)
        ema, decay, restart_unused_codes:  passed to VQEmbedding
    """
    def __init__(
        self,
        vocab_size: int,
        Cvae: int,
        v_patch_nums: Sequence[Union[int, Tuple[int, int, int]]],
        beta: float = 0.25,
        using_znorm: bool = False,
        quant_resi: float = 0.5,
        share_quant_resi: int = 4,
        ema: bool = True,
        decay: float = 0.99,
        restart_unused_codes: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.Cvae = Cvae
        self.beta = beta
        self.using_znorm = using_znorm
        self.v_patch_nums: List[Tuple[int, int, int]] = [_as_dhw(pn) for pn in v_patch_nums]
        self.K = len(self.v_patch_nums)
        self.quant_resi_ratio = quant_resi

        # ---- Phi refinement convs ----
        def _mk_phi():
            return Phi3D(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()

        if share_quant_resi == 0:      # non-shared: one per scale
            self.quant_resi = PhiPartiallyShared3D(nn.ModuleList([_mk_phi() for _ in range(self.K)]))
        elif share_quant_resi == 1:    # fully shared
            self.quant_resi = PhiShared3D(_mk_phi())
        else:                          # partially shared
            self.quant_resi = PhiPartiallyShared3D(
                nn.ModuleList([_mk_phi() for _ in range(share_quant_resi)])
            )

        # ---- Shared codebook (same class as RQVAE3D uses) ----
        self.codebook = VQEmbedding(
            n_embed=vocab_size,
            embed_dim=Cvae,
            ema=ema,
            decay=decay,
            restart_unused_codes=restart_unused_codes,
        )

    # ---------------------------------------------------------------
    # Training forward: encode -> residual quantize -> STE
    # ---------------------------------------------------------------
    def forward(self, f_BCDHW: torch.Tensor):
        """
        f_BCDHW: (B, C, D, H, W) encoder output.

        Returns:
            f_hat:            (B, C, D, H, W) straight-through quantized latent
            commitment_loss:  scalar
            frac_unique:      list[float] len K, fraction of codebook used at each scale
        """
        B, C, D, H, W = f_BCDHW.shape

        f_no_grad = f_BCDHW.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        mean_vq_loss = f_BCDHW.new_zeros(())
        frac_unique_list: List[float] = []

        for si, (pd, ph, pw) in enumerate(self.v_patch_nums):
            is_last = (si == self.K - 1)

            # 1) downsample residual to this scale
            rest_ds = f_rest if is_last else F.interpolate(f_rest, size=(pd, ph, pw), mode='area')

            # 2) shared-codebook nearest-neighbour lookup (channel-last)
            rest_dhwc = rest_ds.permute(0, 2, 3, 4, 1).contiguous()   # (B, pd, ph, pw, C)
            if self.using_znorm:
                # cosine variant: normalize inputs and codebook rows
                normed = F.normalize(rest_dhwc, dim=-1)
                E = F.normalize(self.codebook.weight[:-1, :], dim=-1)  # exclude padding row
                logits = normed.reshape(-1, C) @ E.t()                 # (N, V)
                idx = logits.argmax(dim=-1).view(B, pd, ph, pw)
                embeds = self.codebook.embed(idx)                      # (B, pd, ph, pw, C)
                # EMA update path when training
                if self.training and self.codebook.ema:
                    self.codebook._update_buffers(rest_dhwc, idx)
                    self.codebook._update_embedding()
            else:
                embeds, idx, _ = self.codebook(rest_dhwc)              # L2 lookup + EMA inside

            # 3) upsample chosen embeddings back to full (D, H, W)
            h = embeds.permute(0, 4, 1, 2, 3).contiguous()             # (B, C, pd, ph, pw)
            if not is_last:
                h = F.interpolate(h, size=(D, H, W), mode='trilinear', align_corners=False)

            # 4) scale-indexed Phi refine
            h = self.quant_resi[si / max(self.K - 1, 1)](h)

            # 5) accumulate + peel residual
            f_hat = f_hat + h
            f_rest = f_rest - h

            # per-scale commitment loss (both directions, as in VAR)
            mean_vq_loss = (mean_vq_loss
                            + F.mse_loss(f_hat.data, f_BCDHW).mul_(self.beta)
                            + F.mse_loss(f_hat, f_no_grad))

            # diagnostic: codebook usage this scale (0-dim tensor, matches RQVAE3D)
            frac_unique_list.append(
                torch.bincount(idx.reshape(-1), minlength=self.vocab_size).count_nonzero() / self.vocab_size
            )

        mean_vq_loss = mean_vq_loss / self.K

        # straight-through
        f_hat = (f_hat.data - f_no_grad).add_(f_BCDHW)
        return f_hat, mean_vq_loss, frac_unique_list

    # ---------------------------------------------------------------
    # Analysis / VAR data-prep utilities
    # ---------------------------------------------------------------
    @torch.no_grad()
    def f_to_idxBl_or_fhat(
        self,
        f_BCDHW: torch.Tensor,
        to_fhat: bool,
        v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int, int]]]] = None,
    ) -> List[torch.Tensor]:
        """
        Encode a latent into either:
          - to_fhat=True  : list[Tensor(B, C, D, H, W)]  cumulative reconstructions
          - to_fhat=False : list[LongTensor(B, pd*ph*pw)]  flat token maps per scale
        """
        B, C, D, H, W = f_BCDHW.shape
        f_rest = f_BCDHW.detach().clone()
        f_hat = torch.zeros_like(f_rest)
        patches = [_as_dhw(pn) for pn in (v_patch_nums or self.v_patch_nums)]
        assert patches[-1] == (D, H, W)

        out: List[torch.Tensor] = []
        for si, (pd, ph, pw) in enumerate(patches):
            is_last = (si == len(patches) - 1)
            rest_ds = f_rest if is_last else F.interpolate(f_rest, size=(pd, ph, pw), mode='area')
            rest_dhwc = rest_ds.permute(0, 2, 3, 4, 1).contiguous()
            idx, _ = self.codebook.find_nearest_embedding(rest_dhwc)   # (B, pd, ph, pw)
            embeds = self.codebook.embed(idx)                          # (B, pd, ph, pw, C)
            h = embeds.permute(0, 4, 1, 2, 3).contiguous()
            if not is_last:
                h = F.interpolate(h, size=(D, H, W), mode='trilinear', align_corners=False)
            h = self.quant_resi[si / max(len(patches) - 1, 1)](h)
            f_hat = f_hat + h
            f_rest = f_rest - h
            out.append(f_hat.clone() if to_fhat else idx.reshape(B, pd * ph * pw))
        return out

    @torch.no_grad()
    def idxBl_to_var_input(self, gt_ms_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
        """
        Teacher-forcing input for a VAR-style transformer.
        Returns (B, L - l_0, C) with L = sum_k pd_k*ph_k*pw_k, i.e. every scale
        except the first, as area-downsampled running f_hat.
        """
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.Cvae
        D, H, W = self.v_patch_nums[-1]
        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, D, H, W, dtype=torch.float32)
        next_scales = []
        for si in range(self.K - 1):
            pd, ph, pw = self.v_patch_nums[si]
            idx = gt_ms_idx_Bl[si].view(B, pd, ph, pw)
            h = self.codebook.embed(idx).permute(0, 4, 1, 2, 3).contiguous()
            h = F.interpolate(h, size=(D, H, W), mode='trilinear', align_corners=False)
            f_hat = f_hat + self.quant_resi[si / max(self.K - 1, 1)](h)

            pd_n, ph_n, pw_n = self.v_patch_nums[si + 1]
            next_scales.append(
                F.interpolate(f_hat, size=(pd_n, ph_n, pw_n), mode='area')
                 .reshape(B, C, -1).transpose(1, 2)                    # (B, l_{s+1}, C)
            )
        return torch.cat(next_scales, dim=1) if next_scales else None

    @torch.no_grad()
    def get_next_autoregressive_input(
        self, si: int, f_hat: torch.Tensor, h_BCDHW: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference-time: refine sampled h, add to running f_hat, return next-scale conditioning."""
        D, H, W = self.v_patch_nums[-1]
        is_last = (si == self.K - 1)
        if not is_last:
            h = self.quant_resi[si / max(self.K - 1, 1)](
                F.interpolate(h_BCDHW, size=(D, H, W), mode='trilinear', align_corners=False)
            )
            f_hat = f_hat + h
            pd_n, ph_n, pw_n = self.v_patch_nums[si + 1]
            return f_hat, F.interpolate(f_hat, size=(pd_n, ph_n, pw_n), mode='area')
        else:
            h = self.quant_resi[si / max(self.K - 1, 1)](h_BCDHW)
            f_hat = f_hat + h
            return f_hat, f_hat

    def extra_repr(self) -> str:
        return (f'v_patch_nums={self.v_patch_nums}, K={self.K}, znorm={self.using_znorm}, '
                f'beta={self.beta}, quant_resi={self.quant_resi_ratio}')


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class MSVQVAE3D(nn.Module):
    """
    Multi-scale VQ-VAE for 3D volumes. Mirrors RQVAE3D's init signature so the
    two are drop-in swappable in your trainer.

    Returns from forward:
        x_hat:           (B, C, D, H, W)
        commitment_loss: scalar
        codes:           None during VQVAE training (list-of-token-maps only
                         needed for VAR-training; call encode_multiscale() for it)
        z_e:             pre-quant latent
        frac_unique:     list[float] len K, per-scale codebook usage this batch
    """
    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 768,
        quant_embed_dim: int = 768,
        n_embed: int = 4096,
        resolution: int = 64,
        num_res_blocks_enc: int = 2,
        num_res_blocks_dec: int = 4,
        channels_enc=[64, 64, 256, 512, 512],
        channels_dec=[512, 512, 256, 64, 64],
        # multi-scale specific
        v_patch_nums=(1, 2, 4, 8),
        quant_resi: float = 0.5,
        share_quant_resi: int = 4,
        using_znorm: bool = False,
        beta: float = 0.25,
        # codebook (kept in sync with RQVAE3D defaults for fair comparison)
        ema: bool = True,
        decay: float = 0.99,
        restart_unused_codes: bool = True,
        # encoder/decoder
        skip_attn: bool = True,
        attn_resolutions=[16],
        use_checkpoint: bool = False,
    ):
        super().__init__()
        down_factor = 2 ** (len(channels_enc) - 2)
        latent_res = resolution // down_factor

        # Sanity: the last scale of v_patch_nums must match the latent grid
        last = _as_dhw(v_patch_nums[-1])
        assert last == (latent_res, latent_res, latent_res), (
            f'v_patch_nums[-1]={last} must equal the encoder output resolution '
            f'({latent_res},)*3. Adjust v_patch_nums or channels_enc.'
        )

        self.encoder = Encoder(
            image_channels=in_channels,
            latent_dim=latent_dim,
            num_res_blocks=num_res_blocks_enc,
            resolution=resolution,
            attn_resolutions=attn_resolutions,
            channels=channels_enc,
            skip_attn=skip_attn,
            use_checkpoint=use_checkpoint,
        )
        self.decoder = Decoder(
            image_channels=in_channels,
            latent_dim=latent_dim,
            num_res_blocks=num_res_blocks_dec,
            resolution=latent_res,
            attn_resolutions=attn_resolutions,
            channels=channels_dec,
            skip_attn=skip_attn,
            use_checkpoint=use_checkpoint,
        )

        self.pre_quant_conv = nn.Conv3d(latent_dim, quant_embed_dim, 1)
        self.post_quant_conv = nn.Conv3d(quant_embed_dim, latent_dim, 1)

        self.quantizer = MultiScaleBottleneck3D(
            vocab_size=n_embed,
            Cvae=quant_embed_dim,
            v_patch_nums=v_patch_nums,
            beta=beta,
            using_znorm=using_znorm,
            quant_resi=quant_resi,
            share_quant_resi=share_quant_resi,
            ema=ema,
            decay=decay,
            restart_unused_codes=restart_unused_codes,
        )

    def encode(self, x):
        return self.pre_quant_conv(self.encoder(x))

    def decode(self, z):
        return self.decoder(self.post_quant_conv(z))

    def forward(self, x):
        z_e = self.encode(x)
        f_hat, vq_loss, frac_unique = self.quantizer(z_e)
        x_hat = self.decode(f_hat)
        return x_hat, vq_loss, None, z_e, frac_unique

    # -------- helpers for VAR training later --------
    @torch.no_grad()
    def encode_multiscale(self, x) -> List[torch.Tensor]:
        """Returns list[LongTensor(B, l_k)] token maps per scale (VAR training data)."""
        z_e = self.encode(x)
        return self.quantizer.f_to_idxBl_or_fhat(z_e, to_fhat=False)

    @torch.no_grad()
    def decode_multiscale(self, ms_idx_Bl: List[torch.Tensor]):
        """Decode list of per-scale token maps back to a volume."""
        B = ms_idx_Bl[0].shape[0]
        D, H, W = self.quantizer.v_patch_nums[-1]
        f_hat = ms_idx_Bl[0].new_zeros(B, self.quantizer.Cvae, D, H, W, dtype=torch.float32)
        for si, idx_Bl in enumerate(ms_idx_Bl):
            pd, ph, pw = self.quantizer.v_patch_nums[si]
            idx = idx_Bl.view(B, pd, ph, pw)
            h = self.quantizer.codebook.embed(idx).permute(0, 4, 1, 2, 3).contiguous()
            if (pd, ph, pw) != (D, H, W):
                h = F.interpolate(h, size=(D, H, W), mode='trilinear', align_corners=False)
            f_hat = f_hat + self.quantizer.quant_resi[si / max(self.quantizer.K - 1, 1)](h)
        return self.decode(f_hat).clamp_(-1, 1)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    total_gpu_mem = (torch.cuda.get_device_properties(0).total_memory / 1e9
                     if torch.cuda.is_available() else 0)

    patch_size = 64
    latent_dim = 768
    quant_embed_dim = 768
    channels_enc = [64, 64, 256, 512, 512]   # down_factor = 8 -> latent 8^3
    channels_dec = [512, 512, 256, 64, 64]

    model = MSVQVAE3D(
        in_channels=1,
        latent_dim=latent_dim,
        quant_embed_dim=quant_embed_dim,
        channels_enc=channels_enc,
        channels_dec=channels_dec,
        n_embed=4096,
        resolution=patch_size,
        num_res_blocks_enc=2,
        num_res_blocks_dec=4,
        v_patch_nums=(1, 2, 3, 4, 6, 8),           # -> 828 tokens per volume
        quant_resi=0.5,
        share_quant_resi=4,
        skip_attn=True,
        use_checkpoint=True,
    ).to(device)

    print('Number of parameters, G', numel(model, only_trainable=True))
    model.train()

    x = torch.randn(1, 1, patch_size, patch_size, patch_size, device=device)
    x_hat, loss, codes, z_e, frac_unique = model(x)

    print(f'Input:            {tuple(x.shape)}')
    print(f'Output:           {tuple(x_hat.shape)}')
    print(f'Commitment loss:  {loss.item():.4f}')
    print(f'Frac unique / scale: {[f"{u:.2f}" for u in frac_unique]}')

    ms_idx = model.encode_multiscale(x)
    print(f'Per-scale token maps: {[tuple(t.shape) for t in ms_idx]} '
          f'(total tokens = {sum(t.shape[1] for t in ms_idx)})')

    x_rec = model.decode_multiscale(ms_idx)
    print(f'Round-trip decode: {tuple(x_rec.shape)}')

    if torch.cuda.is_available():
        max_memory_reserved = torch.cuda.max_memory_reserved()
        print('Max memory reserved: %0.3f Gb / %0.3f Gb'
              % (max_memory_reserved / 1e9, total_gpu_mem))
