"""
Residual Self-Correction (RSC) for the 3D hybrid RQ + next-scale tokenizer
(RMSVQVAE3D), the VQ-code analog of BitwiseSelfCorrection3D.

Infinity's BitwiseSelfCorrection flips a random fraction of *bits* on early
scales, requantizes, and feeds the corrupted cumulative latent forward so the
transformer learns to correct its own next-scale rollout. RMSVQVAE3D has no
bits: each position holds D residual VQ indices. The analog of a bit flip is a
**random-code replacement** — with a per-scale probability, a chosen code index
is swapped for a random codebook entry, and the summed-over-depth vector that
accumulates into the running f_hat is rebuilt from the swapped codes.

Scope (matching the chosen design): self-correction is applied on the **scale
axis only**. The inner D-depth RQ chain is quantized cleanly; its indices are the
ground-truth targets and its (non-cumulative) code vectors are the depth
transformer's teacher-forcing stream. Only the cross-scale cumulative latent
(the spatial branch's teacher input) is corrupted.

Driven from the trainer, mirroring Infinity / BitwiseSelfCorrection3D:

    rsc = ResidualSelfCorrection3D(vae, noise_apply_layers, noise_apply_strength)
    raw = vae.encode(x)                                   # (B, C, D, H, W) pre-quant latent
    x_BLC_wo_prefix, code_vectors, gt_ms_idx_Bl = rsc.flip_requant(raw, device)

Returns:
    x_BLC_wo_prefix: (B, sum_{k=1..K-1} l_k, C) spatial teacher input — cumulative
                     (corrupted) f_hat resampled (z_up) to scales 1..K-1. Scale 0's
                     region is excluded ("wo_prefix"): the transformer conditions
                     scale 0 on the SOS/LR prefix, not on a var input.
    code_vectors:    (B, L_total, D, C) per-depth (non-cumulative) GT code vectors
                     for the depth branch stream (cumsum done inside the transformer).
    gt_ms_idx_Bl:    list length K of (B, l_k, D) long GT depth indices, taken from
                     the clean quantization of each scale's residual (re-derived
                     against the corrupted cross-scale history, as in BSC).

Faithfulness notes:
    * Next-scale resample uses the quantizer's `z_up` (trilinear), matching
      RMSVQVAE3D.get_next_autoregressive_input and idxBl_to_var_input.
    * The residual is peeled with the *clean* summed code, so GT indices and the
      depth stream are the true quantization; only `cum_var_input` accumulates the
      swapped code. With noise_apply_layers=0 this is byte-identical to clean
      teacher forcing (== idxBl_to_var_input + idx_to_depth_code_vectors).
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class ResidualSelfCorrection3D:
    """Standalone RSC driver (not an nn.Module, matching BitwiseSelfCorrection3D)."""

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
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        raw_features: (B, C, D, H, W) == vae.encode(x) (pre-quant C-channel latent).

        Returns (x_BLC_wo_prefix, code_vectors, gt_ms_idx_Bl); see module docstring.
        """
        q = self.quantizer
        schedule = q.v_patch_nums
        C = q.Cvae
        D = q.n_rq_depth
        V = q.vocab_size

        with torch.amp.autocast("cuda", enabled=False):
            codes_out = raw_features.float()                 # (B, C, D, H, W)
            B = codes_out.shape[0]
            cum_var_input = torch.zeros_like(codes_out)       # corrupted running f_hat (full res)

            gt_ms_idx_Bl: List[torch.Tensor] = []
            depth_vecs: List[torch.Tensor] = []
            x_BLC_wo_prefix: List[torch.Tensor] = []

            for si, (pd, ph, pw) in enumerate(schedule):
                is_last = (si == q.K - 1)

                # residual against the (corrupted) running reconstruction, down to this scale
                residual = codes_out - cum_var_input
                if not is_last:
                    residual = F.interpolate(residual, size=(pd, ph, pw), mode=q.z_down).contiguous()
                r_dhwc = residual.permute(0, 2, 3, 4, 1).contiguous()   # (B, pd, ph, pw, C)

                # clean D-depth RQ: GT indices + per-depth vectors + swapped summed code
                depth_rest = r_dhwc
                clean_summed = torch.zeros_like(r_dhwc)
                swap_summed = torch.zeros_like(r_dhwc)
                idx_list, emb_list = [], []
                for d, cb in enumerate(q.codebooks):
                    idx, _ = cb.find_nearest_embedding(depth_rest)       # (B, pd, ph, pw) clean GT
                    emb = cb.embed(idx)                                  # (B, pd, ph, pw, C)
                    depth_rest = depth_rest - emb                        # peel with clean code (GT chain)
                    idx_list.append(idx)
                    emb_list.append(emb)
                    clean_summed = clean_summed + emb

                    # random-code replacement on early scales -> corrupted summed code
                    if self.noise_apply_requant and si < self.noise_apply_layers:
                        strength = np.random.randint(0, int(100 * self.noise_apply_strength) + 1) * 0.01
                        swap = torch.rand(idx.shape, device=device) < strength
                        sidx = torch.where(swap, torch.randint_like(idx, 0, V), idx)
                        swap_summed = swap_summed + cb.embed(sidx)
                    else:
                        swap_summed = swap_summed + emb

                gt_ms_idx_Bl.append(torch.stack(idx_list, dim=-1).reshape(B, pd * ph * pw, D))   # (B, l_k, D)
                depth_vecs.append(torch.stack(emb_list, dim=-2).reshape(B, pd * ph * pw, D, C))  # (B, l_k, D, C)

                # accumulate the (corrupted) summed code into the running f_hat
                q_scale = q._upsample_and_refine(swap_summed.permute(0, 4, 1, 2, 3).contiguous(), si)
                cum_var_input = cum_var_input + q_scale

                # spatial teacher input for the NEXT scale: cumulative f_hat down (z_up) to scale si+1
                if not is_last:
                    pd_n, ph_n, pw_n = schedule[si + 1]
                    this_scale = F.interpolate(cum_var_input, size=(pd_n, ph_n, pw_n), mode=q.z_up).contiguous()
                    x_BLC_wo_prefix.append(this_scale.reshape(B, C, -1).permute(0, 2, 1))          # (B, l_{si+1}, C)

            x_BLC_wo_prefix = torch.cat(x_BLC_wo_prefix, dim=1)   # (B, sum l_{1..K-1}, C)
            code_vectors = torch.cat(depth_vecs, dim=1)           # (B, L_total, D, C)

        return x_BLC_wo_prefix, code_vectors, gt_ms_idx_Bl
