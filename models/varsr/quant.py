"""
Multi-scale residual VectorQuantizer for the VARSR baseline (2D).

Faithful port of the official VARSR ``models/quant.py``
(https://github.com/quyp2000/VARSR/blob/master/models/quant.py), which is the
VAR multi-scale quantizer (``VectorQuantizer2``).

Deviations from the official file — all deliberate and documented:

  * The official file does ``import dist`` (VARSR's own distributed helper) and
    unconditionally computes ``margin`` from ``tdist.get_world_size()`` in
    ``forward``. That crashes outside a launched distributed job. Here every
    distributed op is guarded by ``_dist_ok()`` (``torch.distributed`` availability
    + initialization), so the module runs identically single-GPU or under DDP.
  * ``forward`` returns ``(f_hat, vq_loss, frac_unique)`` where ``frac_unique`` is
    a ``list[Tensor]`` (per-scale fraction of the codebook touched this batch).
    This matches the contract ``models/model_vqvae.py::ModelVQVAE`` already uses
    for the 3D quantizers (it maintains its own EMA of these values), instead of
    the official ``usages`` percentages that require the EMA-hit buffer + dist.
    The ``ema_vocab_hit_SV`` buffer is still kept and updated under DDP so that
    official RGB checkpoints load with strict key matching.
  * ``f_to_idxBl_or_fhat`` drops the trailing quantization-residual element and the
    second ``idx_N_list`` return that the official file appends for its diffusion
    refiner (which this baseline does not port). It now returns a plain
    ``list`` of per-scale token maps / cumulative ``f_hat`` maps, matching the VAR
    reference semantics used by the transformer stage.

Progressive training (``prog_si``) is not supported (always ``-1``), same as the
official file.
"""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import distributed as tdist, nn as nn
from torch.nn import functional as F


# this file only provides the VectorQuantizer2 used in VQVAE
__all__ = ['VectorQuantizer2']


def _dist_ok() -> bool:
    """True only when a real distributed process group is up (safe single-GPU)."""
    return tdist.is_available() and tdist.is_initialized()


class VectorQuantizer2(nn.Module):
    # VQGAN originally use beta=1.0, never tried 0.25; SD seems using 0.25
    def __init__(
        self, vocab_size, Cvae, using_znorm, beta: float = 0.25,
        default_qresi_counts=0, v_patch_nums=None, quant_resi=0.5, share_quant_resi=4,  # share_quant_resi: args.qsr
    ):
        super().__init__()
        self.vocab_size: int = vocab_size
        self.Cvae: int = Cvae
        self.using_znorm: bool = using_znorm
        self.v_patch_nums: Tuple[int] = tuple(v_patch_nums)

        self.quant_resi_ratio = quant_resi
        if share_quant_resi == 0:   # non-shared: \phi_{1 to K} for K scales
            self.quant_resi = PhiNonShared([(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(default_qresi_counts or len(self.v_patch_nums))])
        elif share_quant_resi == 1:  # fully shared: only a single \phi for K scales
            self.quant_resi = PhiShared(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
        else:                        # partially shared: \phi_{1 to share_quant_resi} for K scales
            self.quant_resi = PhiPartiallyShared(nn.ModuleList([(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(share_quant_resi)]))

        self.register_buffer('ema_vocab_hit_SV', torch.full((len(self.v_patch_nums), self.vocab_size), fill_value=0.0))
        self.record_hit = 0

        self.beta: float = beta
        self.embedding = nn.Embedding(self.vocab_size, self.Cvae)

        # only used for progressive training of VAR (not supported)
        self.prog_si = -1   # progressive training: not supported, prog_si always -1

    def eini(self, eini):
        if eini > 0:
            nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0:
            self.embedding.weight.data.uniform_(-abs(eini) / self.vocab_size, abs(eini) / self.vocab_size)

    def extra_repr(self) -> str:
        return f'{self.v_patch_nums}, znorm={self.using_znorm}, beta={self.beta}  |  S={len(self.v_patch_nums)}, quant_resi={self.quant_resi_ratio}'

    # ===================== `forward` is only used in VAE training =====================
    def forward(self, f_BChw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        dtype = f_BChw.dtype
        if dtype != torch.float32:
            f_BChw = f_BChw.float()
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()

        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        frac_unique: List[torch.Tensor] = []
        with torch.autocast('cuda', enabled=False):
            mean_vq_loss: torch.Tensor = f_BChw.new_zeros(())
            SN = len(self.v_patch_nums)
            for si, pn in enumerate(self.v_patch_nums):  # from small to large
                # find the nearest embedding
                rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN - 1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                if self.using_znorm:
                    rest_NC = F.normalize(rest_NC, dim=-1)
                    idx_N = torch.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
                else:
                    d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                    d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
                    idx_N = torch.argmin(d_no_grad, dim=1)

                hit_V = idx_N.bincount(minlength=self.vocab_size).float()
                handler = None
                if self.training and _dist_ok():
                    handler = tdist.all_reduce(hit_V, async_op=True)

                # calc loss
                idx_Bhw = idx_N.view(B, pn, pn)
                h_BChw = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (si != SN - 1) else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
                h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
                f_hat = f_hat + h_BChw
                f_rest -= h_BChw

                if self.training and handler is not None:
                    handler.wait()
                    if self.record_hit == 0:
                        self.ema_vocab_hit_SV[si].copy_(hit_V)
                    elif self.record_hit < 100:
                        self.ema_vocab_hit_SV[si].mul_(0.9).add_(hit_V.mul(0.1))
                    else:
                        self.ema_vocab_hit_SV[si].mul_(0.99).add_(hit_V.mul(0.01))
                    self.record_hit += 1

                frac_unique.append((hit_V > 0).float().mean())
                mean_vq_loss = mean_vq_loss + F.mse_loss(f_hat.data, f_BChw).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad)

            mean_vq_loss *= 1. / SN
            f_hat = (f_hat.data - f_no_grad).add_(f_BChw)

        return f_hat, mean_vq_loss, frac_unique
    # ===================== `forward` is only used in VAE training =====================

    def embed_to_fhat(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale=True, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        ls_f_hat_BChw = []
        B = ms_h_BChw[0].shape[0]
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        if all_to_max_scale:
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, H, W, dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums):  # from small to large
                h_BChw = ms_h_BChw[si]
                if si < len(self.v_patch_nums) - 1:
                    h_BChw = F.interpolate(h_BChw, size=(H, W), mode='bicubic')
                h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
                f_hat.add_(h_BChw)
                if last_one:
                    ls_f_hat_BChw = f_hat
                else:
                    ls_f_hat_BChw.append(f_hat.clone())
        else:
            # WARNING: only for experimental purposes; not used in VQVAE training or inference.
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, self.v_patch_nums[0], self.v_patch_nums[0], dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums):  # from small to large
                f_hat = F.interpolate(f_hat, size=(pn, pn), mode='bicubic')
                h_BChw = self.quant_resi[si / (SN - 1)](ms_h_BChw[si])
                f_hat.add_(h_BChw)
                if last_one:
                    ls_f_hat_BChw = f_hat
                else:
                    ls_f_hat_BChw.append(f_hat)

        return ls_f_hat_BChw

    def f_to_idxBl_or_fhat(self, f_BChw: torch.Tensor, to_fhat: bool, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[Union[torch.Tensor, torch.LongTensor]]:
        # NOTE: unlike the official file this returns a plain list (no trailing
        # quantization-residual element, no idx_N_list) because the diffusion
        # refiner is not ported. Semantics now match the VAR reference.
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        f_hat_or_idx_Bl: List[torch.Tensor] = []

        patch_hws = [(pn, pn) if isinstance(pn, int) else (pn[0], pn[1]) for pn in (v_patch_nums or self.v_patch_nums)]  # from small to large
        assert patch_hws[-1][0] == H and patch_hws[-1][1] == W, f'{patch_hws[-1]=} != ({H=}, {W=})'

        SN = len(patch_hws)
        for si, (ph, pw) in enumerate(patch_hws):  # from small to large
            z_NC = F.interpolate(f_rest, size=(ph, pw), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN - 1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
            if self.using_znorm:
                z_NC = F.normalize(z_NC, dim=-1)
                idx_N = torch.argmax(z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
            else:
                d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                d_no_grad.addmm_(z_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
                idx_N = torch.argmin(d_no_grad, dim=1)

            idx_Bhw = idx_N.view(B, ph, pw)
            h_BChw = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (si != SN - 1) else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
            h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
            f_hat.add_(h_BChw)
            f_rest.sub_(h_BChw)
            f_hat_or_idx_Bl.append(f_hat.clone() if to_fhat else idx_N.reshape(B, ph * pw))

        return f_hat_or_idx_Bl

    # ===================== idxBl_to_var_input: teacher-forcing input for VAR training =====================
    def idxBl_to_var_input(self, gt_ms_idx_Bl: List[torch.Tensor], v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> torch.Tensor:
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.Cvae
        if v_patch_nums is None:
            v_patch_nums = self.v_patch_nums
        H = W = v_patch_nums[-1]
        SN = len(v_patch_nums)

        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, H, W, dtype=torch.float32)
        pn_next: int = v_patch_nums[0]
        for si in range(SN - 1):
            h_BChw = F.interpolate(self.embedding(gt_ms_idx_Bl[si]).transpose_(1, 2).view(B, C, pn_next, pn_next), size=(H, W), mode='bicubic')
            f_hat.add_(self.quant_resi[si / (SN - 1)](h_BChw))
            pn_next = v_patch_nums[si + 1]
            next_scales.append(F.interpolate(f_hat, size=(pn_next, pn_next), mode='area').view(B, C, -1).transpose(1, 2))
        return torch.cat(next_scales, dim=1) if len(next_scales) else None    # cat BlCs to BLC, float32

    # ===================== get_next_autoregressive_input: next-step input during VAR inference =====================
    def get_next_autoregressive_input(self, si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        HW = self.v_patch_nums[-1]
        if si != SN - 1:
            h = self.quant_resi[si / (SN - 1)](F.interpolate(h_BChw, size=(HW, HW), mode='bicubic'))     # conv after upsample
            f_hat.add_(h)
            return f_hat, F.interpolate(f_hat, size=(self.v_patch_nums[si + 1], self.v_patch_nums[si + 1]), mode='area')
        else:
            h = h_BChw
            f_hat.add_(h)
            return f_hat, f_hat


class Phi(nn.Conv2d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks // 2)
        self.resi_ratio = abs(quant_resi)

    def forward(self, h_BChw):
        return h_BChw.mul(1 - self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)


class PhiShared(nn.Module):
    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi: Phi = qresi

    def __getitem__(self, _) -> Phi:
        return self.qresi


class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4 else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)

    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]

    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'


class PhiNonShared(nn.ModuleList):
    def __init__(self, qresi: List):
        super().__init__(qresi)
        K = len(qresi)
        self.ticks = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4 else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)

    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())

    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'
