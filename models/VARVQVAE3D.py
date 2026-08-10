"""
Faithful 3D port of VAR's VectorQuantizer2 (models/quant.py).

Reference: https://github.com/FoundationVision/VAR/blob/main/models/quant.py

This is deliberately kept as close to the reference as possible so it can be
used to isolate training-stability issues in the 3D multi-scale VQ-VAE. In
particular, unlike `MSVQVAE3D.MultiScaleBottleneck3D` (which reuses the
EMA + dead-code-restart `VQEmbedding` from RQVAE3D), this version follows VAR
exactly:

  * The codebook is a *plain* ``nn.Embedding``. It is NOT updated with EMA and
    there is NO dead-code restart. The codebook is trained only by back-prop
    through the commitment / codebook MSE term and the straight-through
    estimator, exactly as in the reference.
  * The residual-quantization loop, the L2/cosine nearest-neighbour lookup, the
    per-scale ``Phi`` refinement, the loss formulation, and the straight-through
    trick are line-for-line equivalents of the 2D reference.

3D-specific deviations from the 2D reference (unavoidable / documented):
  * ``F.interpolate(..., mode='bicubic')`` (2D only) becomes
    ``mode='trilinear', align_corners=False`` for the upsampling step.
  * ``pn`` scale descriptors may be an int (isotropic) or a ``(D, H, W)`` tuple.

Progressive training (``prog_si``) is intentionally NOT ported.
"""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.basic_vae import EncoderV3 as Encoder
from models.basic_vae import DecoderV3 as Decoder
from utils.utils_3D_image import numel


def _as_dhw(pn) -> Tuple[int, int, int]:
    """Accept int (isotropic) or a 3-tuple (D, H, W)."""
    return (pn, pn, pn) if isinstance(pn, int) else tuple(pn)


# ---------------------------------------------------------------------------
# Phi refinement convs  (VAR: Phi / PhiShared / PhiPartiallyShared / PhiNonShared)
# ---------------------------------------------------------------------------
class Phi3D(nn.Conv3d):
    """(1 - r) * x + r * conv(x). 3x3x3 residual refine applied after upsampling."""
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(embed_dim, embed_dim, kernel_size=ks, stride=1, padding=ks // 2)
        self.resi_ratio = abs(quant_resi)

    def forward(self, h_BCDHW):
        return h_BCDHW.mul(1 - self.resi_ratio) + super().forward(h_BCDHW).mul_(self.resi_ratio)


class PhiShared3D(nn.Module):
    """Fully shared: one Phi for every scale."""
    def __init__(self, qresi: Phi3D):
        super().__init__()
        self.qresi: Phi3D = qresi

    def __getitem__(self, _: float) -> Phi3D:
        return self.qresi


class PhiPartiallyShared3D(nn.Module):
    """K Phi's resolved to the nearest scale by a tick on [0, 1]."""
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = (np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4
                      else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K))

    def __getitem__(self, at_from_0_to_1: float) -> Phi3D:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]

    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'


class PhiNonShared3D(nn.ModuleList):
    """Non-shared: one Phi per scale, resolved by tick (identical mapping to VAR)."""
    def __init__(self, qresi: list):
        super().__init__(qresi)
        K = len(qresi)
        self.ticks = (np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4
                      else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K))

    def __getitem__(self, at_from_0_to_1: float) -> Phi3D:
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())

    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'


# ---------------------------------------------------------------------------
# VectorQuantizer2 (3D)
# ---------------------------------------------------------------------------
class VectorQuantizer2_3D(nn.Module):
    """
    Faithful 3D port of VAR's ``VectorQuantizer2``.

    Args:
        vocab_size:        codebook size V (shared across all scales)
        Cvae:              codebook embedding dim C
        v_patch_nums:      scales small -> large. Each entry is an int (isotropic)
                           or a (D, H, W) tuple. The LAST scale must equal the
                           encoder output resolution.
        using_znorm:       if True, use cosine similarity for the NN lookup
                           (L2 distance otherwise) - matches VAR.
        beta:              commitment loss weight.
        quant_resi:        residual ratio for Phi. 0.5 => 0.5*conv(x) + 0.5*x.
                           Set to 0 to disable Phi (Phi becomes Identity).
        share_quant_resi:  1 = single shared Phi; N>1 = N Phi's mapped by tick;
                           0 = one Phi per scale.
        eini:              codebook init. 0 -> default nn.Embedding init (VAR's
                           behaviour when training from the released checkpoint).
                           >0 -> trunc_normal_ with std=eini. <0 -> uniform in
                           [-|eini|/V, |eini|/V] (the classic VQ init; often more
                           stable when training from scratch).
    """
    def __init__(
        self,
        vocab_size: int,
        Cvae: int,
        v_patch_nums: Sequence[Union[int, Tuple[int, int, int]]],
        using_znorm: bool = False,
        beta: float = 0.25,
        quant_resi: float = 0.5,
        share_quant_resi: int = 4,
        eini: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.Cvae = Cvae
        self.using_znorm = using_znorm
        self.v_patch_nums: List[Tuple[int, int, int]] = [_as_dhw(pn) for pn in v_patch_nums]
        self.beta: float = beta

        # ---- Phi refinement convs ----
        self.quant_resi_ratio = quant_resi

        def _mk_phi():
            return Phi3D(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()

        if share_quant_resi == 0:      # non-shared: one per scale
            self.quant_resi = PhiNonShared3D([_mk_phi() for _ in range(len(self.v_patch_nums))])
        elif share_quant_resi == 1:    # fully shared
            self.quant_resi = PhiShared3D(_mk_phi())
        else:                          # partially shared
            self.quant_resi = PhiPartiallyShared3D(
                nn.ModuleList([_mk_phi() for _ in range(share_quant_resi)])
            )

        # ---- Plain codebook (trained by back-prop, NO EMA / restart) ----
        self.embedding = nn.Embedding(self.vocab_size, self.Cvae)
        self.eini(eini)

    def eini(self, eini: float):
        if eini > 0:
            nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0:
            self.embedding.weight.data.uniform_(-abs(eini) / self.vocab_size, abs(eini) / self.vocab_size)
        # eini == 0: leave default nn.Embedding init untouched

    def extra_repr(self) -> str:
        return (f'{self.v_patch_nums}, znorm={self.using_znorm}, beta={self.beta}  |  '
                f'S={len(self.v_patch_nums)}, quant_resi={self.quant_resi_ratio}')

    # ---- shared nearest-neighbour lookup for a residual at one scale ----
    def _find_idx(self, rest_NC: torch.Tensor) -> torch.Tensor:
        """rest_NC: (N, C) -> idx_N: (N,) long. Uses .data so no grad flows here."""
        if self.using_znorm:
            rest_NC = F.normalize(rest_NC, dim=-1)
            idx_N = torch.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
        else:
            d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) \
                + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
            d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (N, V)
            idx_N = torch.argmin(d_no_grad, dim=1)
        return idx_N

    # ===================== forward: only used in VAE training =====================
    def forward(self, f_BCDHW: torch.Tensor):
        """
        f_BCDHW: (B, C, D, H, W) encoder output.

        Returns (matches MSVQVAE3D.MultiScaleBottleneck3D so it is drop-in):
            f_hat:        (B, C, D, H, W) straight-through quantized latent
            vq_loss:      scalar
            frac_unique:  list[Tensor] len S, fraction of codebook used per scale
        """
        dtype = f_BCDHW.dtype
        if dtype != torch.float32:
            f_BCDHW = f_BCDHW.float()
        B, C, D, H, W = f_BCDHW.shape
        f_no_grad = f_BCDHW.detach()

        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        with torch.amp.autocast('cuda', enabled=False):
            mean_vq_loss: torch.Tensor = f_BCDHW.new_zeros(())
            frac_unique: List[torch.Tensor] = []
            SN = len(self.v_patch_nums)

            for si, (pd, ph, pw) in enumerate(self.v_patch_nums):   # small -> large
                is_last = (si == SN - 1)

                # 1) downsample residual to this scale, flatten to (N, C)
                rest_ds = f_rest if is_last else F.interpolate(f_rest, size=(pd, ph, pw), mode='area')
                rest_NC = rest_ds.permute(0, 2, 3, 4, 1).reshape(-1, C)

                # 2) nearest-neighbour lookup in the shared codebook
                idx_N = self._find_idx(rest_NC)
                hit_V = idx_N.bincount(minlength=self.vocab_size)

                # 3) embed (WITH grad -> trains the codebook), upsample, Phi refine
                idx_Bdhw = idx_N.view(B, pd, ph, pw)
                h_BCDHW = self.embedding(idx_Bdhw).permute(0, 4, 1, 2, 3)      # (B, C, pd, ph, pw)
                if not is_last:
                    h_BCDHW = F.interpolate(h_BCDHW, size=(D, H, W), mode='trilinear', align_corners=False)
                h_BCDHW = h_BCDHW.contiguous()
                h_BCDHW = self.quant_resi[si / max(SN - 1, 1)](h_BCDHW)

                # 4) accumulate + peel residual
                f_hat = f_hat + h_BCDHW
                f_rest = f_rest - h_BCDHW

                # 5) per-scale loss (commitment via f_hat.data, codebook via f_hat)
                mean_vq_loss = mean_vq_loss \
                    + F.mse_loss(f_hat.data, f_BCDHW).mul_(self.beta) \
                    + F.mse_loss(f_hat, f_no_grad)

                frac_unique.append((hit_V > 0).float().mean())

            mean_vq_loss = mean_vq_loss * (1.0 / SN)

            # straight-through estimator
            f_hat = (f_hat.data - f_no_grad).add_(f_BCDHW)

        return f_hat, mean_vq_loss, frac_unique

    # ---------------------------------------------------------------
    # Analysis / VAR data-prep utilities (faithful 3D ports)
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
          - to_fhat=True  : list[Tensor(B, C, D, H, W)]   cumulative reconstructions
          - to_fhat=False : list[LongTensor(B, pd*ph*pw)] flat token maps per scale
        """
        B, C, D, H, W = f_BCDHW.shape
        f_rest = f_BCDHW.detach().clone()
        f_hat = torch.zeros_like(f_rest)

        patches = [_as_dhw(pn) for pn in (v_patch_nums or self.v_patch_nums)]
        assert patches[-1] == (D, H, W), \
            f'last scale {patches[-1]} must equal latent grid {(D, H, W)}'

        SN = len(patches)
        out: List[torch.Tensor] = []
        for si, (pd, ph, pw) in enumerate(patches):
            is_last = (si == SN - 1)
            rest_ds = f_rest if is_last else F.interpolate(f_rest, size=(pd, ph, pw), mode='area')
            idx_N = self._find_idx(rest_ds.permute(0, 2, 3, 4, 1).reshape(-1, C))
            idx_Bdhw = idx_N.view(B, pd, ph, pw)
            h_BCDHW = self.embedding(idx_Bdhw).permute(0, 4, 1, 2, 3)
            if not is_last:
                h_BCDHW = F.interpolate(h_BCDHW, size=(D, H, W), mode='trilinear', align_corners=False)
            h_BCDHW = self.quant_resi[si / max(SN - 1, 1)](h_BCDHW.contiguous())
            f_hat.add_(h_BCDHW)
            f_rest.sub_(h_BCDHW)
            out.append(f_hat.clone() if to_fhat else idx_N.reshape(B, pd * ph * pw))
        return out

    @torch.no_grad()
    def idxBl_to_var_input(self, gt_ms_idx_Bl: List[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Teacher-forcing input for a VAR-style transformer.
        Returns (B, L - l_0, C) with L = sum_k pd_k*ph_k*pw_k, i.e. every scale
        except the first, as area-downsampled running f_hat.
        """
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.Cvae
        D, H, W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)

        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, D, H, W, dtype=torch.float32)
        next_scales: List[torch.Tensor] = []
        for si in range(SN - 1):
            pd, ph, pw = self.v_patch_nums[si]
            idx = gt_ms_idx_Bl[si].view(B, pd, ph, pw)
            h_BCDHW = self.embedding(idx).permute(0, 4, 1, 2, 3)
            h_BCDHW = F.interpolate(h_BCDHW, size=(D, H, W), mode='trilinear', align_corners=False)
            f_hat.add_(self.quant_resi[si / max(SN - 1, 1)](h_BCDHW.contiguous()))

            pd_n, ph_n, pw_n = self.v_patch_nums[si + 1]
            next_scales.append(
                F.interpolate(f_hat, size=(pd_n, ph_n, pw_n), mode='area')
                 .reshape(B, C, -1).transpose(1, 2)                       # (B, l_{s+1}, C)
            )
        return torch.cat(next_scales, dim=1) if len(next_scales) else None

    @torch.no_grad()
    def get_next_autoregressive_input(
        self, si: int, SN: int, f_hat: torch.Tensor, h_BCDHW: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference-time: refine sampled h, add to running f_hat, return next conditioning."""
        D, H, W = self.v_patch_nums[-1]
        if si != SN - 1:
            h = self.quant_resi[si / max(SN - 1, 1)](
                F.interpolate(h_BCDHW, size=(D, H, W), mode='trilinear', align_corners=False)
            )
            f_hat.add_(h)
            pd_n, ph_n, pw_n = self.v_patch_nums[si + 1]
            return f_hat, F.interpolate(f_hat, size=(pd_n, ph_n, pw_n), mode='area')
        else:
            h = self.quant_resi[si / max(SN - 1, 1)](h_BCDHW)
            f_hat.add_(h)
            return f_hat, f_hat

    @torch.no_grad()
    def embed_to_fhat(
        self, ms_h_BCDHW: List[torch.Tensor], all_to_max_scale: bool = True, last_one: bool = False
    ):
        """Accumulate a list of per-scale h maps into (a list of) f_hat."""
        B = ms_h_BCDHW[0].shape[0]
        D, H, W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)

        ls_f_hat: List[torch.Tensor] = []
        f_hat = ms_h_BCDHW[0].new_zeros(B, self.Cvae, D, H, W, dtype=torch.float32)
        for si in range(SN):
            h_BCDHW = ms_h_BCDHW[si]
            if si < SN - 1:
                h_BCDHW = F.interpolate(h_BCDHW, size=(D, H, W), mode='trilinear', align_corners=False)
            h_BCDHW = self.quant_resi[si / max(SN - 1, 1)](h_BCDHW.contiguous())
            f_hat.add_(h_BCDHW)
            if last_one:
                ls_f_hat = f_hat
            else:
                ls_f_hat.append(f_hat.clone())
        return ls_f_hat


# ---------------------------------------------------------------------------
# Thin VAE wrapper (mirrors MSVQVAE3D so the two are drop-in swappable)
# ---------------------------------------------------------------------------
class VARVQVAE3D(nn.Module):
    """
    3D multi-scale VQ-VAE using the faithful VectorQuantizer2_3D bottleneck.
    Same init signature / forward return as MSVQVAE3D for a clean A/B swap.

    forward returns:
        x_hat:       (B, C, D, H, W)
        vq_loss:     scalar
        codes:       None during VQVAE training (use encode_multiscale() for tokens)
        z_e:         pre-quant latent
        frac_unique: list[Tensor] len S, per-scale codebook usage this batch
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
        quant_resi: float = 0.5,
        share_quant_resi: int = 4,
        using_znorm: bool = False,
        beta: float = 0.25,
        eini: float = 0.0,
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

        self.quantizer = VectorQuantizer2_3D(
            vocab_size=n_embed,
            Cvae=quant_embed_dim,
            v_patch_nums=v_patch_nums,
            using_znorm=using_znorm,
            beta=beta,
            quant_resi=quant_resi,
            share_quant_resi=share_quant_resi,
            eini=eini,
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
        z_e = self.encode(x)
        return self.quantizer.f_to_idxBl_or_fhat(z_e, to_fhat=False)

    @torch.no_grad()
    def decode_multiscale(self, ms_idx_Bl: List[torch.Tensor]):
        B = ms_idx_Bl[0].shape[0]
        D, H, W = self.quantizer.v_patch_nums[-1]
        SN = len(self.quantizer.v_patch_nums)
        with torch.amp.autocast('cuda', enabled=False):
            f_hat = ms_idx_Bl[0].new_zeros(B, self.quantizer.Cvae, D, H, W, dtype=torch.float32)
            for si, idx_Bl in enumerate(ms_idx_Bl):
                pd, ph, pw = self.quantizer.v_patch_nums[si]
                idx = idx_Bl.view(B, pd, ph, pw)
                h = self.quantizer.embedding(idx).float().permute(0, 4, 1, 2, 3)
                if (pd, ph, pw) != (D, H, W):
                    h = F.interpolate(h, size=(D, H, W), mode='trilinear', align_corners=False)
                f_hat = f_hat + self.quantizer.quant_resi[si / max(SN - 1, 1)](h.contiguous())
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

    model = VARVQVAE3D(
        in_channels=1,
        latent_dim=latent_dim,
        quant_embed_dim=quant_embed_dim,
        channels_enc=channels_enc,
        channels_dec=channels_dec,
        n_embed=4096,
        resolution=patch_size,
        num_res_blocks_enc=2,
        num_res_blocks_dec=4,
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8),           # -> 828 tokens per volume
        quant_resi=0.5,
        share_quant_resi=4,
        using_znorm=False,
        beta=0.25,
        eini=-1.0,                                  # uniform init (often stabler from scratch)
        skip_attn=True,
        use_checkpoint=True,
    ).to(device)

    print('Number of parameters, G', numel(model, only_trainable=True))
    model.train()

    x = torch.randn(1, 1, patch_size, patch_size, patch_size, device=device)
    x_hat, loss, codes, z_e, frac_unique = model(x)

    print(f"Input:            {tuple(x.shape)}")
    print(f"Output:           {tuple(x_hat.shape)}")
    print(f"Commitment loss:  {loss.item():.4f}")
    print(f"Frac unique / scale: {[f'{u:.2f}' for u in frac_unique]}")

    ms_idx = model.encode_multiscale(x)
    print(f"Per-scale token maps: {[tuple(t.shape) for t in ms_idx]} "
          f"(total tokens = {sum(t.shape[1] for t in ms_idx)})")

    x_rec = model.decode_multiscale(ms_idx)
    print(f"Round-trip decode: {tuple(x_rec.shape)}")

    if torch.cuda.is_available():
        max_memory_reserved = torch.cuda.max_memory_reserved()
        print("Max memory reserved: %0.3f Gb / %0.3f Gb" % (max_memory_reserved / 1e9, total_gpu_mem))