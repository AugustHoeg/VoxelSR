"""
Bitwise Self-Correction (BSC) for the 3D BSQ tokenizer.

Faithful 3D port of Infinity's `BitwiseSelfCorrection.flip_requant`
(https://github.com/FoundationVision/Infinity/blob/main/infinity/models/bitwise_self_correction.py).

Infinity does NOT build the transformer's teacher-forcing input inside the VAE's
`MultiScaleBSQ.forward`; it uses this separate object, driven from the trainer:

    bsc = BitwiseSelfCorrection3D(vae, noise_apply_layers, noise_apply_strength, ...)
    raw_features = vae.encode(x)                         # == Infinity encode_for_raw_features
    x_BLC_wo_prefix, gt_ms_idx_Bl = bsc.flip_requant(raw_features, device)

Returns (exactly like Infinity):
    x_BLC_wo_prefix: (B, sum_{k=1..K-1} l_k, L) teacher-forcing input. Scale 0's
                     region is excluded ("wo_prefix") because the transformer
                     conditions scale 0 on the text/BOS prefix, not on a var input.
    gt_ms_idx_Bl:    list length K of (B, l_k, L) ground-truth bit targets, taken
                     from the *un-flipped* quantization of each scale's residual.

Faithfulness notes:
    * Next-scale resample uses the quantizer's `z_up` (trilinear) mode, matching
      Infinity's `this_scale_input = F.interpolate(cum_var_input, size=next, mode=z_up)`.
      (Infinity uses the *up* mode here even though it downsamples to a smaller grid.)
    * The per-scale code is upsampled+accumulated via the quantizer's own
      `_upsample_and_refine`, so `cum_var_input` is byte-for-byte the VAE's decodable
      `f_hat` space (whatever out_fact / Phi config the VAE runs). To reproduce
      Infinity's plain-trilinear accumulate exactly, run the VAE with
      `use_decay_factor=False, use_prog_quant_resi=False`.
    * GT targets are re-derived against the *corrupted* history (because the residual
      is peeled with the flipped+requantized code), which is the whole point of BSC.
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class BitwiseSelfCorrection3D:
    """Standalone BSC driver (not an nn.Module, matching Infinity)."""

    def __init__(
        self,
        vae,
        noise_apply_layers: int,
        noise_apply_strength: float,
        noise_apply_requant: bool = True,
    ):
        self.vae = vae
        self.quantizer = vae.quantizer
        self.noise_apply_layers = noise_apply_layers
        self.noise_apply_strength = noise_apply_strength
        self.noise_apply_requant = noise_apply_requant

    @torch.no_grad()
    def flip_requant(
        self,
        raw_features: torch.Tensor,
        device,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        raw_features: (B, L, D, H, W) == vae.encode(x) (pre-quant L-channel latent).

        Returns (x_BLC_wo_prefix, gt_ms_idx_Bl); see module docstring.
        """
        q = self.quantizer
        schedule = q.v_patch_nums
        L = q.L

        with torch.amp.autocast("cuda", enabled=False):
            codes_out = raw_features.float()                 # (B, L, D, H, W)
            B = codes_out.shape[0]
            cum_var_input = torch.zeros_like(codes_out)      # == Infinity cum_var_input (full res)

            gt_all_bit_indices: List[torch.Tensor] = []
            x_BLC_wo_prefix: List[torch.Tensor] = []

            for si, (pd, ph, pw) in enumerate(schedule):
                is_last = (si == q.K - 1)

                # residual against the running reconstruction, then down to this scale
                residual = codes_out - cum_var_input
                if not is_last:
                    residual = F.interpolate(
                        residual, size=(pd, ph, pw), mode=q.z_down
                    ).contiguous()

                # quantize this scale's residual (gt bits = UN-flipped)
                quantized, _idx, bit_indices, _aux = q.bsq(residual)   # (B,L,pd,ph,pw), (B,pd,ph,pw,L)
                gt_all_bit_indices.append(bit_indices)

                # bitwise self-correction: flip a random fraction of bits on early scales
                if si < self.noise_apply_layers:
                    strength = np.random.randint(0, int(100 * self.noise_apply_strength) + 1) * 0.01
                    mask = torch.rand(*bit_indices.shape, device=device) < strength
                    pred_bits = bit_indices.clone()
                    pred_bits[mask] = 1 - pred_bits[mask]
                    if self.noise_apply_requant:
                        code = q.bsq.indices_to_code(pred_bits)        # (B,pd,ph,pw,L)
                        quantized = code.permute(0, 4, 1, 2, 3).contiguous()
                # else / requant=False: keep the un-flipped `quantized` (flip is a no-op on the input)

                # upsample+refine to full res and accumulate (== VAE f_hat space)
                quantized = q._upsample_and_refine(quantized, si)
                cum_var_input = cum_var_input + quantized

                # teacher-forcing input for the NEXT scale: cumulative ↓ next scale (trilinear)
                if not is_last:
                    pd_n, ph_n, pw_n = schedule[si + 1]
                    this_scale = F.interpolate(
                        cum_var_input, size=(pd_n, ph_n, pw_n), mode=q.z_up
                    ).contiguous()
                    x_BLC_wo_prefix.append(this_scale.reshape(B, L, -1).permute(0, 2, 1))  # (B, l_{si+1}, L)

            gt_ms_idx_Bl = [b.reshape(B, -1, L) for b in gt_all_bit_indices]  # list of (B, l_k, L)
            x_BLC_wo_prefix = torch.cat(x_BLC_wo_prefix, dim=1)               # (B, sum l_{1..K-1}, L)

        return x_BLC_wo_prefix, gt_ms_idx_Bl
