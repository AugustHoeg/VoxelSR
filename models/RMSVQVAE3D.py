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
    3. Progressive trilinear-upsample to full latent grid and apply Phi refine convs (if enabled)
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

NOTE 2: Renamed old RMSVQVAE3D to RMSVQVAE3D_old and did following changes:
- Added scale dropout (stochastic depth), so a given sample may skip an entire scale during training.
This is independent of the depth dropout (DAC-style) which is still supported.
- Copied progressive upsampling/refinement module (PhiProgressive3D) from BSQVAE3D
- Renaming of variables to be more consistent with BSQVAE3D

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
# Phi refinement convs (per-scale 3x3x3 residual convs with optional activation)
# ---------------------------------------------------------------------------


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Phi3D(nn.Module):
    """(1 - r) * x + r * conv(x). Small learned refine applied after upsampling."""

    def __init__(self, embed_dim, quant_resi, act: nn.Module = nn.Identity()):
        super().__init__()
        self.conv = nn.Conv3d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        self.resi_ratio = abs(quant_resi)
        self.act = act

    def forward(self, x):
        return x.mul(1 - self.resi_ratio) + self.act(self.conv(x)).mul_(self.resi_ratio)


class PhiProgressive3D(nn.Module):
    """Progressive version of Phi refinement mechanism from the VAR paper.
    Instead of upsampling the quantized code to the full latent grid and then applying Phi,
    We upsample progressively K times using trilinear, and apply Phi conv after each upsampling.
    """

    def __init__(self, embed_dim, quant_resi, v_patch_nums, z_up="trilinear", act: nn.Module = nn.Identity()):
        super().__init__()
        self.patches = [_as_dhw(pn) for pn in v_patch_nums]
        self.phi_convs = nn.ModuleList([Phi3D(embed_dim, quant_resi, act) for _ in range(len(self.patches) - 1)])
        self.z_up = z_up

    def forward(self, x: torch.Tensor, scale_start=1) -> torch.Tensor:
        for abs_si in range(scale_start, len(self.patches)):
            pd, ph, pw = self.patches[abs_si]
            x = F.interpolate(x, size=(pd, ph, pw), mode=self.z_up).contiguous()  # interpolate to next scale
            x = self.phi_convs[abs_si - 1](x)  # refine scale: x = (1 - quant_resi) * x + quant_resi * conv(x)

        # for si, (pd, ph, pw) in enumerate(self.patches[scale_start:]):
        #     x = F.interpolate(x, size=(pd, ph, pw), mode=self.z_up)  # interpolate to next scale
        #     x = self.phi_convs[si](x)  # refine scale: x = (1 - quant_resi) * x + quant_resi * conv(x)

        return x


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
        depth_drop_rate:     DAC-style depth dropout in [0, 1]. Probability that a
                             given sample is trained with a randomly reduced number
                             of active depth codebooks (uniform in [1, D]). 0.0
                             disables the feature entirely (byte-identical to the
                             all-depth path). Training only; eval uses full depth.
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
        use_prog_quant_resi: bool = False,
        z_down: str = "area",
        z_up: str = "trilinear",
        ema: bool = True,
        decay: float = 0.99,
        restart_unused_codes: bool = True,
        restart_clamp_factor: float = 1.0,
        skip_update_over: Optional[float] = None,
        use_stochastic_depth: bool = True,
        scale_drop_rate: float = 0.25,
        depth_drop_rate: float = 0.0,
        keep_last_quant: bool = True,
        keep_first_quant: bool = False,
        lambda_vq_depth: float = 1.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.Cvae = Cvae
        self.beta = beta
        self.using_znorm = using_znorm
        self.v_patch_nums: List[Tuple[int, int, int]] = [_as_dhw(pn) for pn in v_patch_nums]
        self.K = len(self.v_patch_nums)
        self.n_rq_depth = n_rq_depth
        self.depth_drop_rate = depth_drop_rate
        self.quant_resi_ratio = quant_resi
        self.z_down = z_down
        self.z_up = z_up

        self.lambda_vq_depth = lambda_vq_depth  # Controls RVQ-style commitment loss on top of VAR-style commitment loss

        self.use_stochastic_depth = use_stochastic_depth
        self.scale_drop_rate = scale_drop_rate
        self.keep_first_quant = keep_first_quant
        self.keep_last_quant = keep_last_quant

        # ---- Phi refinement convs (learned progressive upsampler) ----
        # use_prog_quant_resi = False -> disabled: fall back to no Phi refine.
        self.quant_resi_ratio = quant_resi
        if use_prog_quant_resi:
            self.quant_resi = PhiProgressive3D(
                embed_dim=Cvae, quant_resi=quant_resi, v_patch_nums=v_patch_nums, z_up=z_up, act=Swish()
            )
        else:
            self.quant_resi = None  # Fall back to no Phi refine (plain trilinear upsample)

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

        # RNG for scale dropout (for training only)
        self.scale_rng = torch.Generator().manual_seed(999)
        self.depth_rng = torch.Generator().manual_seed(888)

    # ---------------------------------------------------------------
    # Inner depth RQ at a single scale (channel-last, low resolution)
    # ---------------------------------------------------------------
    def _quantize_depth(self, r: torch.Tensor, n_active: Optional[torch.Tensor] = None):
        """
        Run the D-step residual loop at one scale.

        r: (B, C, pd, ph, pw) residual
        n_active:  optional (B,) LongTensor giving how many depth codebooks are
                   active per sample (quantizer/depth dropout). None -> all D
                   depths active for every sample. The residual is ALWAYS peeled
                   with the full embedding, so deeper codebooks keep seeing the
                   true residual and their EMA stays healthy; dropout only masks
                   which depths are *added* into the summed output (DAC-style).

        Returns:
            r:         (B, pd, ph, pw, C) sum of the active depth embeddings
            depth_fracs: list[Tensor] len D, per-depth codebook usage this scale
        """
        B = r.shape[0]
        r_dhwc = r.permute(0, 2, 3, 4, 1).contiguous()  # (B, pd, ph, pw, C)
        depth_rest = r_dhwc
        quant_out = torch.zeros_like(r_dhwc)
        quant_out_list: List[torch.Tensor] = []
        depth_fracs: List[torch.Tensor] = []

        for d, codebook in enumerate(self.codebooks):
            embeds, idx, _ = codebook(depth_rest)  # quantize
            depth_rest = depth_rest - embeds  # peel residual for next depth

            mask = (d < n_active).view(B, 1, 1, 1, 1).to(embeds.dtype) if n_active is not None else 1.0 # depth dropout
            quant_out = quant_out + embeds * mask  # accumulate the embeddings
            quant_out_list.append(quant_out.permute(0, 4, 1, 2, 3).contiguous())  # list of aggregated quants

            # record codebook usage across depths
            depth_fracs.append(torch.bincount(idx.reshape(-1), minlength=self.vocab_size).count_nonzero() / self.vocab_size)

        quant_out = quant_out.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, pd, ph, pw)

        return quant_out, quant_out_list, depth_fracs

    def depth_commitment_loss(self, x, quant_list):

        loss_list = []

        for idx, quant in enumerate(quant_list):
            partial_loss = F.mse_loss(x, quant.detach())
            loss_list.append(partial_loss)

        commitment_loss = torch.mean(torch.stack(loss_list))
        return commitment_loss

    def _sample_active_depths(self, B: int, device) -> torch.Tensor:
        """Per-sample active depth count for quantizer dropout (DAC-style).

        Samples n ~ U{1, ..., D} for every sample, then reverts a fraction
        (1 - depth_drop_rate) back to full depth D. So depth_drop_rate is the
        probability that a given sample is trained at a reduced depth. The same
        count is used across all scales for a sample, so it stays a coherent
        reduced-depth reconstruction rather than a per-scale mix.
        """
        D = self.n_rq_depth
        n = torch.randint(1, D + 1, (B,), generator=self.depth_rng).to(device)
        keep_full = torch.rand(B, generator=self.depth_rng).to(device) >= self.depth_drop_rate
        return torch.where(keep_full, torch.full_like(n, D), n)

    def _upsample_and_refine(self, q: torch.Tensor, si: int) -> torch.Tensor:
        D, H, W = self.v_patch_nums[-1]
        is_last = si == self.K - 1

        if self.quant_resi is not None:
            # Phi refine (upsamples to full grid internally)
            if not is_last:
                q = self.quant_resi(q, si + 1)
        else:
            # plain trilinear upsample to the full grid
            if not is_last:
                q = F.interpolate(q, size=(D, H, W), mode=self.z_up)

        return q.contiguous()

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

        # per-sample active depth count for quantizer (depth) dropout; None disables
        n_active = None
        if self.training and self.depth_drop_rate > 0:
            n_active = self._sample_active_depths(B, f_BCDHW.device)

        with torch.amp.autocast("cuda", enabled=False):
            f_BCDHW = f_BCDHW.float()
            f_no_grad = f_BCDHW.detach()
            residual = f_no_grad.clone()
            f_hat = torch.zeros_like(residual)

            mean_vq_loss = f_BCDHW.new_zeros(())
            frac_unique_list: List[torch.Tensor] = [torch.tensor(torch.nan, device=f_BCDHW.device)] * self.K

            for si, (pd, ph, pw) in enumerate(self.v_patch_nums):
                is_last = (si == self.K - 1)

                # 1) downsample running residual to this scale
                r = residual if is_last else F.interpolate(residual, size=(pd, ph, pw), mode=self.z_down)

                # 2) quantize this scale's residual
                keep_first = si == 0 and self.keep_first_quant
                keep_last = is_last and self.keep_last_quant
                keep_scale = torch.rand(1, generator=self.scale_rng).item() > self.scale_drop_rate if self.use_stochastic_depth else True
                if keep_scale or keep_first or keep_last or (not self.training):
                    # quantize RVQ-style
                    q, q_list, depth_fracs = self._quantize_depth(r, n_active)
                    frac_unique_list[si] = torch.stack(depth_fracs).mean()
                else:
                    # scale dropout: skip quantization at this scale
                    q = torch.zeros_like(r)
                    q_list = [torch.zeros_like(r) for _ in range(self.n_rq_depth)]

                # 3) upsample to full scale and optionally refine (if enabled)
                q = self._upsample_and_refine(q, si)

                # 4) accumulate + peel residual
                f_hat = f_hat + q
                residual = residual - q

                # add RVQ-style commitment loss enforcing all depths to be utilized
                if keep_scale or keep_first or keep_last or (not self.training):
                    if self.lambda_vq_depth > 0:
                        mean_vq_loss = mean_vq_loss + self.lambda_vq_depth * self.depth_commitment_loss(r, q_list)

                # per-scale commitment loss (both directions, as in VAR)
                mean_vq_loss = mean_vq_loss + F.mse_loss(f_hat.data, f_BCDHW).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad)

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
                f'qdrop={self.depth_drop_rate}, znorm={self.using_znorm}, beta={self.beta}, '
                f'quant_resi={self.quant_resi_ratio}')


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
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8),
        n_rq_depth: int = 4,
        quant_resi: float = 0.5,
        use_prog_quant_resi: bool = False,
        using_znorm: bool = False,
        beta: float = 0.25,
        # codebook (kept in sync with RQVAE3D defaults for fair comparison)
        ema: bool = True,
        decay: float = 0.99,
        restart_unused_codes: bool = True,
        restart_clamp_factor: float = 1.0,
        skip_update_over: Optional[float] = None,
        use_stochastic_depth: bool = True,
        scale_drop_rate: float = 0.1,
        depth_drop_rate: float = 0.1,
        keep_last_quant: bool = True,
        keep_first_quant: bool = False,
        lambda_vq_depth: float = 1.0,
        # encoder/decoder
        skip_attn: bool = True,
        attn_resolutions=[16],
        use_checkpoint: bool = False,
    ):
        super().__init__()
        down_factor = 2 ** (len(channels_enc) - 2)
        latent_res = resolution // down_factor

        last = _as_dhw(v_patch_nums[-1])
        assert last == (latent_res, latent_res, latent_res), (
            f"v_patch_nums[-1]={last} must equal the encoder output resolution "
            f"({latent_res},)*3. Adjust v_patch_nums or channels_enc."
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
            use_prog_quant_resi=use_prog_quant_resi,
            ema=ema,
            decay=decay,
            restart_unused_codes=restart_unused_codes,
            restart_clamp_factor=restart_clamp_factor,
            skip_update_over=skip_update_over,
            use_stochastic_depth=use_stochastic_depth,
            scale_drop_rate=scale_drop_rate,
            depth_drop_rate=depth_drop_rate,
            keep_last_quant=keep_last_quant,
            keep_first_quant=keep_first_quant,
            lambda_vq_depth=lambda_vq_depth,
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
    model = RMSVQVAE3D(
        in_channels=1,
        latent_dim=512,
        quant_embed_dim=512,
        channels_enc= [64, 64, 256, 512, 512],
        channels_dec=[512, 512, 256, 64, 64],
        n_embed=4096,
        resolution=patch_size,
        num_res_blocks_enc=2,
        num_res_blocks_dec=4,
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8),        # -> 953 tokens per volume
        n_rq_depth=4,                              # D shared codebooks per scale
        quant_resi=0.5,
        use_prog_quant_resi=True,
        using_znorm=True,
        use_stochastic_depth=True,
        scale_drop_rate=0.10,
        depth_drop_rate=0.0,
        keep_last_quant=True,
        keep_first_quant=False,
        lambda_vq_depth=0.0,
        skip_attn=False,
        use_checkpoint=True,
    ).to(device)

    print("Number of parameters, G", numel(model, only_trainable=True))
    model.train()

    x = torch.randn(1, 1, patch_size, patch_size, patch_size, device=device)
    x_hat, loss, codes, z_e, frac_unique = model(x)

    print(f"Input:            {tuple(x.shape)}")
    print(f"Output:           {tuple(x_hat.shape)}")
    print(f"VQ loss:          {loss.item():.4f}")
    print(f'Frac unique / scale (depth-avg): {[f"{u:.2f}" for u in frac_unique]}')

    # exercise the quantizer (depth) dropout path on the same model
    model.quantizer.depth_drop_rate = 0.5
    x_hat_d, loss_d, _, _, frac_unique_d = model(torch.randn(2, 1, patch_size, patch_size, patch_size, device=device))
    print(f'[qdrop=0.5] Output: {tuple(x_hat_d.shape)}  loss: {loss_d.item():.4f}  '
          f'frac unique/scale: {[f"{u:.2f}" for u in frac_unique_d]}')
    model.quantizer.depth_drop_rate = 0.0

    if torch.cuda.is_available():
        max_memory_reserved = torch.cuda.max_memory_reserved()
        print("Max memory reserved: %0.3f Gb / %0.3f Gb"
              % (max_memory_reserved / 1e9, total_gpu_mem))
