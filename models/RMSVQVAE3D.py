"""
Residual Multi-Scale VQ-VAE for 3D volumes (RMSVQVAE3D).

A hybrid of MSVQVAE3D and RQVAE3D. Like MSVQVAE3D, residual quantization is
applied across *spatial scales* (coarse grid -> full latent grid). Unlike
MSVQVAE3D - which does a single codebook lookup per scale - each scale here runs
an *inner residual loop over D codebooks* (RQVAE-style depth). The D depth
codebooks are distinct instances but each is shared across all resolution
scales, so depth `d` always uses `codebook_d` regardless of scale.

Per scale (in downsampled scale space):
    1. area-downsample the running residual to the scale grid once
    2. run D residual codebook lookups at that low resolution, peeling the
       residual and summing the D chosen embeddings
    3. trilinear-upsample the summed embedding back to full latent grid once
    4. apply the per-scale Phi refine conv (shared exactly as in MSVQVAE3D)
    5. accumulate into f_hat and peel from the full-res residual

Commitment loss is VAR-style: a both-direction MSE on the cumulative f_hat
after each scale completes its full D-depth quantization (depth is internal to
the scale). Straight-through estimator is applied once at the end.

The codebook implementation (VQEmbedding, EMA + dead-code restart) is reused
from RQVAE3D so the SR comparison isolates the residual-axis choice.

NOTE: Only the VQVAE reconstruction path (forward / encode / decode) is
implemented. The VAR-training helpers (multi-scale token export, teacher
forcing, autoregressive stepping) are stubbed with NotImplementedError until
the depth axis is wired into the transformer data path.

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
# Multi-scale residual quantization bottleneck with per-scale codebook depth
# ---------------------------------------------------------------------------
class MultiScaleResidualBottleneck3D(nn.Module):
    """
    Args:
        vocab_size:          codebook size V (per depth codebook)
        Cvae:                codebook embedding dim C
        v_patch_nums:        list of scales, small -> large. Each entry is an int
                             (isotropic) or a (D, H, W) tuple. The LAST scale must
                             equal the encoder output resolution.
        n_rq_depth:          number of residual codebooks D applied *within* each
                             scale. D distinct codebook instances, each shared
                             across all scales (depth d always uses codebook_d).
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
        n_rq_depth: int = 4,
        beta: float = 0.25,
        using_znorm: bool = False,
        quant_resi: float = 0.5,
        share_quant_resi: int = 4,
        ema: bool = True,
        decay: float = 0.99,
        restart_unused_codes: bool = True,
        restart_clamp_factor: float = 1.0,
        skip_update_over: Optional[float] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.Cvae = Cvae
        self.beta = beta
        self.using_znorm = using_znorm
        self.v_patch_nums: List[Tuple[int, int, int]] = [_as_dhw(pn) for pn in v_patch_nums]
        self.K = len(self.v_patch_nums)
        self.n_rq_depth = n_rq_depth
        self.quant_resi_ratio = quant_resi

        # ---- Phi refinement convs (applied once per scale on the summed depth output) ----
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

        # ---- D depth codebooks (each shared across all scales) ----
        self.codebooks = nn.ModuleList([
            VQEmbedding(
                n_embed=vocab_size,
                embed_dim=Cvae,
                ema=ema,
                decay=decay,
                restart_unused_codes=restart_unused_codes,
                restart_clamp_factor=restart_clamp_factor,
                skip_update_over=skip_update_over,
                using_znorm=using_znorm,
            )
            for _ in range(n_rq_depth)
        ])

    # ---------------------------------------------------------------
    # Inner depth RQ at a single scale (channel-last, low resolution)
    # ---------------------------------------------------------------
    def _quantize_depth(self, rest_dhwc: torch.Tensor):
        """
        Run the D-step residual loop at one scale.

        rest_dhwc: (B, pd, ph, pw, C) residual, channel-last.

        Returns:
            agg:         (B, pd, ph, pw, C) sum of the D chosen embeddings
            depth_fracs: list[Tensor] len D, per-depth codebook usage this scale
        """
        depth_rest = rest_dhwc
        agg = torch.zeros_like(rest_dhwc)
        depth_fracs: List[torch.Tensor] = []

        for codebook in self.codebooks:
            embeds, idx, _ = codebook(depth_rest)
            depth_rest = depth_rest - embeds     # peel residual for next depth
            agg = agg + embeds
            depth_fracs.append(
                torch.bincount(idx.reshape(-1), minlength=self.vocab_size).count_nonzero()
                / self.vocab_size
            )

        return agg, depth_fracs

    # ---------------------------------------------------------------
    # Training forward: encode -> per-scale depth-residual quantize -> STE
    # ---------------------------------------------------------------
    def forward(self, f_BCDHW: torch.Tensor):
        """
        f_BCDHW: (B, C, D, H, W) encoder output.

        Returns:
            f_hat:            (B, C, D, H, W) straight-through quantized latent
            commitment_loss:  scalar
            frac_unique:      list[float] len K, mean over the D depth codebooks of
                              their usage at each scale (diagnostic only)
        """
        B, C, D, H, W = f_BCDHW.shape

        f_BCDHW = f_BCDHW.float()
        f_no_grad = f_BCDHW.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        with torch.amp.autocast("cuda", enabled=False):
            mean_vq_loss = f_BCDHW.new_zeros(())
            frac_unique_list: List[float] = []

            for si, (pd, ph, pw) in enumerate(self.v_patch_nums):
                is_last = (si == self.K - 1)

                # 1) downsample residual to this scale
                rest_ds = f_rest if is_last else F.interpolate(f_rest, size=(pd, ph, pw), mode='area')

                # 2) inner D-depth residual quantization at this scale (channel-last)
                rest_dhwc = rest_ds.permute(0, 2, 3, 4, 1).contiguous()   # (B, pd, ph, pw, C)
                agg_dhwc, depth_fracs = self._quantize_depth(rest_dhwc)

                # 3) upsample summed depth embedding back to full (D, H, W)
                h = agg_dhwc.permute(0, 4, 1, 2, 3).contiguous()           # (B, C, pd, ph, pw)
                if not is_last:
                    h = F.interpolate(h, size=(D, H, W), mode='trilinear', align_corners=False)

                # 4) scale-indexed Phi refine (once, on the summed depth output)
                h = self.quant_resi[si / max(self.K - 1, 1)](h)

                # 5) accumulate + peel residual
                f_hat = f_hat + h
                f_rest = f_rest - h

                # per-scale commitment loss (both directions, as in VAR)
                mean_vq_loss = (mean_vq_loss
                                + F.mse_loss(f_hat.data, f_BCDHW).mul_(self.beta)
                                + F.mse_loss(f_hat, f_no_grad))

                # diagnostic: mean codebook usage over depths this scale (0-dim tensor)
                frac_unique_list.append(torch.stack(depth_fracs).mean())

            mean_vq_loss = mean_vq_loss / self.K

            # straight-through
            f_hat = (f_hat.data - f_no_grad).add_(f_BCDHW)

        return f_hat, mean_vq_loss, frac_unique_list

    # ---------------------------------------------------------------
    # Analysis / VAR data-prep utilities (not yet implemented for depth)
    # ---------------------------------------------------------------
    @torch.no_grad()
    def f_to_idxBl_or_fhat(self, *args, **kwargs):
        raise NotImplementedError(
            'RMSVQVAE3D: VAR-training helpers are not implemented yet. The depth '
            'axis (D codes per scale position) must be wired into the transformer '
            'data path first. Only the VQVAE reconstruction path is available.'
        )

    @torch.no_grad()
    def idxBl_to_var_input(self, *args, **kwargs):
        raise NotImplementedError(
            'RMSVQVAE3D: idxBl_to_var_input is not implemented for the depth axis yet.'
        )

    @torch.no_grad()
    def get_next_autoregressive_input(self, *args, **kwargs):
        raise NotImplementedError(
            'RMSVQVAE3D: get_next_autoregressive_input is not implemented for the depth axis yet.'
        )

    def extra_repr(self) -> str:
        return (f'v_patch_nums={self.v_patch_nums}, K={self.K}, n_rq_depth={self.n_rq_depth}, '
                f'znorm={self.using_znorm}, beta={self.beta}, quant_resi={self.quant_resi_ratio}')


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class RMSVQVAE3D(nn.Module):
    """
    Residual Multi-Scale VQ-VAE for 3D volumes. Mirrors MSVQVAE3D's init
    signature (plus n_rq_depth) so the two are drop-in swappable in your trainer.

    Returns from forward:
        x_hat:           (B, C, D, H, W)
        commitment_loss: scalar
        codes:           None during VQVAE training
        z_e:             pre-quant latent
        frac_unique:     list[float] len K, per-scale (depth-averaged) codebook usage
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
        n_rq_depth: int = 4,
        quant_resi: float = 0.5,
        share_quant_resi: int = 4,
        using_znorm: bool = False,
        beta: float = 0.25,
        # codebook (kept in sync with RQVAE3D defaults for fair comparison)
        ema: bool = True,
        decay: float = 0.99,
        restart_unused_codes: bool = True,
        restart_clamp_factor: float = 1.0,
        skip_update_over: Optional[float] = None,
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

        self.quantizer = MultiScaleResidualBottleneck3D(
            vocab_size=n_embed,
            Cvae=quant_embed_dim,
            v_patch_nums=v_patch_nums,
            n_rq_depth=n_rq_depth,
            beta=beta,
            using_znorm=using_znorm,
            quant_resi=quant_resi,
            share_quant_resi=share_quant_resi,
            ema=ema,
            decay=decay,
            restart_unused_codes=restart_unused_codes,
            restart_clamp_factor=restart_clamp_factor,
            skip_update_over=skip_update_over,
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

    # -------- helpers for VAR training later (not yet implemented) --------
    @torch.no_grad()
    def encode_multiscale(self, x):
        raise NotImplementedError(
            'RMSVQVAE3D: encode_multiscale is not implemented yet. The per-scale '
            'depth axis must be wired into the VAR transformer data path first.'
        )

    @torch.no_grad()
    def decode_multiscale(self, ms_idx_Bl):
        raise NotImplementedError(
            'RMSVQVAE3D: decode_multiscale is not implemented yet.'
        )


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

    model = RMSVQVAE3D(
        in_channels=1,
        latent_dim=latent_dim,
        quant_embed_dim=quant_embed_dim,
        channels_enc=channels_enc,
        channels_dec=channels_dec,
        n_embed=4096,
        resolution=patch_size,
        num_res_blocks_enc=2,
        num_res_blocks_dec=4,
        v_patch_nums=(1, 2, 3, 4, 6, 8),           # -> 828 spatial tokens per volume
        n_rq_depth=4,                              # D shared codebooks per scale
        quant_resi=0.5,
        share_quant_resi=4,
        using_znorm=True,
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
    print(f'Frac unique / scale (depth-avg): {[f"{u:.2f}" for u in frac_unique]}')

    if torch.cuda.is_available():
        max_memory_reserved = torch.cuda.max_memory_reserved()
        print('Max memory reserved: %0.3f Gb / %0.3f Gb'
              % (max_memory_reserved / 1e9, total_gpu_mem))
