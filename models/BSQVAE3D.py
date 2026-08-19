"""
Multi-Scale Binary Spherical Quantization (BSQ) VAE for 3D volumes.

Port of BitVAE / Infinity's MultiScaleBSQ quantizer
(https://github.com/FoundationVision/BitVAE, https://arxiv.org/abs/2406.07548)
to 3D, kept drop-in swappable with MSVQVAE3D / RQVAE3D so the tokenizer-family
SR comparison isolates the *quantizer* choice.

Relationship to MSVQVAE3D:
    * Same multi-scale residual scaffolding (coarse grid -> full latent grid),
      the same area-downsample / trilinear-upsample residual loop.
    * MSVQVAE3D quantizes each residual against a *learned* codebook (VQEmbedding,
      EMA + dead-code restart, commitment loss). BSQ is **lookup-free**: it
      projects the residual to an L-dim space and binarizes each dim to +-1 on the
      unit sphere. The implicit vocabulary is 2**L; there is no codebook to learn,
      no EMA, no dead-code restart.
    * Because codes are always unit-norm (direction only), BitVAE reconciles them
      with residual refinement via a hand-set magnitude decay `out_fact`
      (1.0 -> 0.1, step -0.1 per scale) instead of learned Phi refine convs.
    * Codebook usage is regularized by a factorized entropy loss (BSQ paper eq.),
      O(L) rather than O(2**L), so it scales to large L.

Token representation (Infinity-style, bitwise):
    Each spatial position emits an L-bit code. `forward` returns the quantized
    latent for the decoder; `encode_multiscale` / `f_to_bits_or_fhat` return the
    per-scale bit maps (B, l_k, L) that an Infinity-style transformer predicts
    bit-by-bit. Integer indices are materialized as a convenience only when
    L <= 62 (int64-safe); for larger L use the bits directly.

Defaults picked for a 64^3 input -> 8^3 latent (down_factor=8).
"""

from typing import Callable, List, Optional, Sequence, Tuple, Union

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


def _binary_entropy(p: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Elementwise binary (Bernoulli) entropy H_b(p), in nats.

    eps must survive float32: 1 - 1e-8 rounds to 1.0, which would leave
    log(1-p) = -inf and 0 * -inf = NaN when the sigmoid saturates.
    """
    p = p.clamp(eps, 1.0 - eps)
    return -(p * p.log() + (1.0 - p) * (1.0 - p).log())

def get_entropy(count, dim=-1):
    """Reference implementation of get_entropy assuming normalize=False"""
    H = -(count * torch.log(count + 1e-8)).sum(dim=dim)
    return H


# ---------------------------------------------------------------------------
# Binary Spherical Quantizer (single scale, lookup-free)
# ---------------------------------------------------------------------------
class BSQ3D(nn.Module):
    """
    Binary Spherical Quantization for a single (already-projected) latent.

    Operates on the *channel* dim: the incoming (B, L, D, H, W) tensor is treated
    as L-dim vectors per voxel, binarized to +-1 and scaled by 1/sqrt(L) so every
    code lands on the unit hypersphere. Implicit codebook size = 2**L.

    Args:
        codebook_bits (L):     number of bits / spherical dimensions.
        entropy_loss_weight:   weight on the factorized entropy penalty.
        commitment_loss_weight:weight on ||z - q.detach()||^2 (encoder-side only).
        inv_temperature:       sigmoid sharpness for the soft-bit distribution.
        diversity_gamma:       weight of the batch (diversity) entropy term.
    """

    def __init__(
        self,
        codebook_bits: int,
        entropy_loss_weight: float = 0.1,     # reference `entropy_weight`
        commitment_loss_weight: float = 0.25,
        inv_temperature: float = 100.0,
        diversity_gamma: float = 1.0,         # reference `gamma` (batch/diversity term)
        gamma0: float = 1.0,                  # per-sample entropy weight
        zeta: float = 1.0,                    # overall entropy-penalty scale
        return_loss_breakdown: bool = False,
    ):
        super().__init__()
        self.L = codebook_bits
        self.q_scale = 1.0 / (codebook_bits ** 0.5)
        self.entropy_loss_weight = entropy_loss_weight
        self.commitment_loss_weight = commitment_loss_weight
        self.inv_temperature = inv_temperature
        self.diversity_gamma = diversity_gamma
        self.gamma0 = gamma0
        self.zeta = zeta
        self.return_loss_breakdown = return_loss_breakdown
        # 2**L overflows int64 past 62 bits; only then materialize integer indices
        self.can_make_indices = codebook_bits <= 62
        if self.can_make_indices:
            self.register_buffer(
                "_bit_weights",
                (2 ** torch.arange(codebook_bits, dtype=torch.long)),
                persistent=False,
            )

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        # reference keeps quantize unscaled (zhat in {-1, +1}); q_scale is applied
        # AFTER the straight-through so the STE gradient to z is q_scale, not 1.
        zhat = torch.where(z > 0, torch.ones_like(z), -torch.ones_like(z))
        return z + (zhat - z).detach()

    def soft_entropy_loss(self, z):
        p = torch.sigmoid(-4 * z / (self.L ** 0.5) * self.inv_temperature)
        prob = torch.stack([p, 1 - p], dim=-1)  # (..., L, 2)
        # per-sample: sum per-bit binary entropy over L, mean over samples
        per_sample_entropy = get_entropy(prob, dim=-1).sum(dim=-1).mean()

        # batch/macro average of the *stacked* [p, 1-p] over all samples -> (L, 2),
        # then per-bit binary entropy summed over L (reference reduce '... g d -> g d')
        avg_prob = prob.reshape(-1, self.L, 2).mean(dim=0)  # (L, 2)
        codebook_entropy = get_entropy(avg_prob, dim=-1)    # (L,)

        # factorized approx: total entropy is the sum of per-bit (subgroup) entropies
        return per_sample_entropy, codebook_entropy.sum(), avg_prob

    def forward(self, z_BLDHW: torch.Tensor):
        """
        z_BLDHW: (B, L, D, H, W) projected residual for this scale.

        Returns:
            quantized:   (B, L, D, H, W) unit-sphere code, STE-connected to z.
            indices:     (B, D, H, W) long, or None if L > 62.
            bit_indices: (B, D, H, W, L) bool  (the transformer targets).
            aux_loss:    scalar reference aux_loss (commit + entropy penalty).
            breakdown:   only if return_loss_breakdown -> (per_sample_H, cb_H, commit).
        """
        # channel-last: treat each voxel as an L-vector
        z = z_BLDHW.permute(0, 2, 3, 4, 1).contiguous()          # (B, D, H, W, L)

        # Normalize
        z = F.normalize(z, dim=-1)

        # Force F32 precision
        z = z.float()

        with torch.amp.autocast('cuda', enabled=False):

            # binarize with straight-through estimator, then scale onto unit sphere
            code = self.quantize(z)
            code = self.q_scale * code       # reference applies q_scale after the STE

            # calculate indices
            bit_indices = (code > 0).int()  # (B,D,H,W,L) bool

            # entropy penalty (BSQ paper factorized form; reference `entropy_penalty`)
            per_sample_entropy, codebook_entropy, avg_p = self.soft_entropy_loss(z)
            entropy_penalty = self.gamma0 * per_sample_entropy - self.diversity_gamma * codebook_entropy

            # commitment (lookup-free: only the encoder is pulled toward the code)
            commit_loss = F.mse_loss(z, code.detach())

            # reference aux_loss: commit + temperature-normalized entropy penalty
            aux_loss = (commit_loss * self.commitment_loss_weight
                        + (self.zeta * entropy_penalty / self.inv_temperature) * self.entropy_loss_weight)

            # --- discrete outputs ---
            indices = None
            if self.can_make_indices:
                indices = (bit_indices.long() * self._bit_weights).sum(dim=-1)  # (B,D,H,W)

            quantized = code.permute(0, 4, 1, 2, 3).contiguous()      # (B, L, D, H, W)

            if self.return_loss_breakdown:
                breakdown = (per_sample_entropy.detach(),
                             codebook_entropy.detach(),
                             commit_loss.detach())
                return quantized, indices, bit_indices, aux_loss, breakdown
            return quantized, indices, bit_indices, aux_loss

    def indices_to_code(self, indices: torch.Tensor) -> torch.Tensor:
        """ Added to match functionality of reference implementation.
        https://github.com/FoundationVision/BitVAE/blob/main/bitvae/modules/quantizer/multiscale_bsq.py#L505
        """

        # indices to codes, which are bits of either -1 or 1
        codes = self.bits_to_code(indices) * self.q_scale
        return codes

    # -- bits <-> code (for decoding / teacher forcing, no gradient path) --
    def bits_to_code(self, bits_BL: torch.Tensor) -> torch.Tensor:
        """(..., L) {0,1} bits -> (..., L) {-1, +1} code"""
        zhat = bits_BL.to(torch.float32) * 2.0 - 1.0
        return zhat

    def normalized_bit_usage(self, bit_indices: torch.Tensor) -> torch.Tensor:
        """Diagnostic in [0, 1]: mean per-bit entropy / log(2). ~1 => bits balanced."""
        p = bit_indices.float().reshape(-1, self.L).mean(dim=0)   # (L,) P(bit=1)
        return _binary_entropy(p).mean() / np.log(2.0)


# ---------------------------------------------------------------------------
# Multi-scale residual BSQ bottleneck
# ---------------------------------------------------------------------------
class MultiScaleBSQ3D(nn.Module):
    """
    Coarse-to-fine residual quantization on the unit hypersphere.

    Args:
        codebook_bits:       L, spherical dimensions / bits per token.
        v_patch_nums:        list of scales, small -> large. int (isotropic) or
                             (D, H, W). The LAST scale must equal the encoder
                             output resolution.
        use_decay_factor:    if True, scale-s code magnitude = max(0.1, 1-0.1*s);
                             the BitVAE mechanism that lets unit-norm codes act as
                             shrinking residual corrections. If False, out_fact=1.
        z_down / z_up:       interpolation modes for residual down / code up.
        entropy_loss_weight, commitment_loss_weight, inv_temperature,
        diversity_gamma:     passed to BSQ3D.
    """

    def __init__(
        self,
        codebook_bits: int,
        v_patch_nums: Sequence[Union[int, Tuple[int, int, int]]],
        use_decay_factor: bool = True,
        z_down: str = "area",
        z_up: str = "trilinear",
        entropy_loss_weight: float = 0.1,
        commitment_loss_weight: float = 0.25,
        inv_temperature: float = 100.0,
        diversity_gamma: float = 1.0,
        gamma0: float = 1.0,
        zeta: float = 1.0,
        lfq_weight: float = 1.0,
    ):
        super().__init__()
        self.L = codebook_bits
        self.v_patch_nums: List[Tuple[int, int, int]] = [_as_dhw(pn) for pn in v_patch_nums]
        self.K = len(self.v_patch_nums)
        self.use_decay_factor = use_decay_factor
        self.z_down = z_down
        self.z_up = z_up
        self.lfq_weight = lfq_weight

        self.bsq = BSQ3D(
            codebook_bits=codebook_bits,
            entropy_loss_weight=entropy_loss_weight,
            commitment_loss_weight=commitment_loss_weight,
            inv_temperature=inv_temperature,
            diversity_gamma=diversity_gamma,
            gamma0=gamma0,
            zeta=zeta,
        )

    def _out_fact(self, si: int) -> float:
        """BitVAE decay schedule: 1.0, 0.9, ... clamped at 0.1 (or constant 1.0)."""
        return max(0.1, 1.0 - 0.1 * si) if self.use_decay_factor else 1.0

    # ---------------------------------------------------------------
    # Training forward
    # ---------------------------------------------------------------
    def forward(self, f_BLDHW: torch.Tensor):
        """
        f_BLDHW: (B, L, D, H, W) encoder output projected to L channels.

        Returns:
            f_hat:        (B, L, D, H, W) quantized latent (STE-connected to input).
            vq_loss:      scalar (entropy + commitment, scale-averaged & weighted).
            frac_unique:  list[Tensor] len K, per-scale normalized bit usage in[0,1].
        """
        B, L, D, H, W = f_BLDHW.shape
        assert L == self.L, f"encoder produced {L} channels, expected L={self.L}"

        with torch.amp.autocast("cuda", enabled=False):
            f = f_BLDHW.float()
            residual = f
            f_hat = torch.zeros_like(f)

            all_losses: List[torch.Tensor] = []
            frac_unique: List[torch.Tensor] = []

            for si, (pd, ph, pw) in enumerate(self.v_patch_nums):
                is_last = (si == self.K - 1)

                # 1) downsample running residual to this scale
                r = residual if is_last else F.interpolate(residual, size=(pd, ph, pw), mode=self.z_down)

                # 2) binary spherical quantize (per-scale reference aux_loss)
                q, _idx, bit_idx, aux_loss = self.bsq(r)

                # 3) decay-scaled magnitude for this scale
                q = q * self._out_fact(si)

                # 4) upsample code back to full latent grid
                if not is_last:
                    q = F.interpolate(q, size=(D, H, W), mode=self.z_up)
                q = q.contiguous()

                # 5) accumulate (grad path) + peel residual (detached, like BitVAE)
                f_hat = f_hat + q
                residual = residual - q.detach()

                all_losses.append(aux_loss)
                frac_unique.append(self.bsq.normalized_bit_usage(bit_idx))

            # reference: stack per-scale aux_loss; d_vae reduces with mean * lfq_weight
            vq_loss = torch.stack(all_losses).mean() * self.lfq_weight

        return f_hat, vq_loss, frac_unique

    # ---------------------------------------------------------------
    # Analysis / transformer data-prep
    # ---------------------------------------------------------------
    @torch.no_grad()
    def f_to_bits_or_fhat(
        self,
        f_BLDHW: torch.Tensor,
        to_fhat: bool,
        v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int, int]]]] = None,
        bit_noise_fn: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        """
        Encode a latent into either:
          - to_fhat=True  : list[Tensor(B, L, D, H, W)]   cumulative reconstructions
          - to_fhat=False : list[IntTensor(B, pd*ph*pw, L)] per-scale *ground-truth* bit maps

        Bitwise Self-Correction hook (Infinity). If `bit_noise_fn` is given it is
        called per scale as `bit_noise_fn(si, gt_bit_idx) -> used_bit_idx`, where
        `gt_bit_idx` is the true quantization of the current residual, shape
        (B, pd, ph, pw, L). The returned (possibly bit-flipped) `used_bit_idx` is
        requantized and used for the residual peel + f_hat accumulation, so the
        downstream scales see the *corrupted* history (train/test-discrepancy
        reduction). The emitted maps stay ground-truth: `to_fhat=False` returns the
        true `gt_bit_idx` (transformer targets), `to_fhat=True` returns the corrupted
        cumulative f_hat (teacher-forcing input). A BSC module can therefore make one
        `to_fhat=True` pass and capture the gt bits inside its own `bit_noise_fn`
        closure under the same random-flip realization. `bit_noise_fn=None` is an
        exact no-op (identity requantization reproduces the plain code).
        """
        B, L, D, H, W = f_BLDHW.shape
        with torch.amp.autocast("cuda", enabled=False):
            patches = [_as_dhw(pn) for pn in (v_patch_nums or self.v_patch_nums)]
            assert patches[-1] == (D, H, W)

            residual = f_BLDHW.float()
            f_hat = torch.zeros_like(residual)
            out: List[torch.Tensor] = []
            for si, (pd, ph, pw) in enumerate(patches):
                is_last = (si == len(patches) - 1)
                r = residual if is_last else F.interpolate(residual, size=(pd, ph, pw), mode=self.z_down)
                q, _idx, bit_idx, _aux = self.bsq(r)

                # BSC: requantize (possibly flipped) bits for the accumulated history
                if bit_noise_fn is not None:
                    used_bit_idx = bit_noise_fn(si, bit_idx)
                    code = self.bsq.indices_to_code(used_bit_idx)          # (B,pd,ph,pw,L)
                    q = code.permute(0, 4, 1, 2, 3).contiguous()           # (B,L,pd,ph,pw)

                q = q * self._out_fact(si)
                if not is_last:
                    q = F.interpolate(q, size=(D, H, W), mode=self.z_up)
                q = q.contiguous()
                f_hat = f_hat + q
                residual = residual - q
                out.append(f_hat.clone() if to_fhat else bit_idx.reshape(B, pd * ph * pw, L))
        return out

    @torch.no_grad()
    def f_to_var_input(
        self,
        f_BLDHW: torch.Tensor,
        bit_noise_fn: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
        return_gt_bits: bool = False,
    ):
        """
        Infinity-style teacher-forcing inputs, built exactly as
        Infinity's MultiScaleBSQ.forward `var_inputs`:

            for si in [0, K-2]:
                var_inputs[si] = area_down(quantized_out_after_scale_si, scale[si+1])

        i.e. the running cumulative f_hat (== Infinity `quantized_out`) after scale
        si, area-downsampled to the *next* scale, in the L-dim code space (before
        post_quant_conv). Length K-1; these condition the transformer's prediction
        of scales 1..K-1 (scale 0 comes from the start/prefix token).

        If `return_gt_bits`, also returns the per-scale ground-truth bit maps
        (B, l_k, L) so a transformer step gets teacher inputs + targets from ONE
        pass -- consistent even under a Bitwise Self-Correction `bit_noise_fn`
        (see f_to_bits_or_fhat). This mirrors Infinity returning both
        `all_bit_indices` and `var_inputs` from the same forward.
        """
        B, L, D, H, W = f_BLDHW.shape
        with torch.amp.autocast("cuda", enabled=False):
            residual = f_BLDHW.float()
            f_hat = torch.zeros_like(residual)          # == Infinity quantized_out
            var_inputs: List[torch.Tensor] = []
            gt_bits: List[torch.Tensor] = []
            for si, (pd, ph, pw) in enumerate(self.v_patch_nums):
                is_last = (si == self.K - 1)
                r = residual if is_last else F.interpolate(residual, size=(pd, ph, pw), mode=self.z_down)
                q, _idx, bit_idx, _aux = self.bsq(r)
                gt_bits.append(bit_idx.reshape(B, pd * ph * pw, L))

                # BSC: requantize (possibly flipped) bits into the accumulated history
                if bit_noise_fn is not None:
                    code = self.bsq.indices_to_code(bit_noise_fn(si, bit_idx))
                    q = code.permute(0, 4, 1, 2, 3).contiguous()

                q = q * self._out_fact(si)
                if not is_last:
                    q = F.interpolate(q, size=(D, H, W), mode=self.z_up)
                q = q.contiguous()
                residual = residual - q
                f_hat = f_hat + q

                # Infinity: append running quantized_out area-downsampled to next scale
                if not is_last:
                    pd_n, ph_n, pw_n = self.v_patch_nums[si + 1]
                    var_inputs.append(
                        F.interpolate(f_hat, size=(pd_n, ph_n, pw_n), mode=self.z_down).contiguous()
                    )
        return (var_inputs, gt_bits) if return_gt_bits else var_inputs


    @torch.no_grad()
    def bits_to_fhat(self, ms_bits: List[torch.Tensor]) -> torch.Tensor:
        """Per-scale bit maps (B, l_k, L) -> full-resolution quantized latent (B, L, D, H, W)."""
        B = ms_bits[0].shape[0]
        D, H, W = self.v_patch_nums[-1]
        with torch.amp.autocast("cuda", enabled=False):
            f_hat = ms_bits[0].new_zeros(B, self.L, D, H, W, dtype=torch.float32)
            for si, bits in enumerate(ms_bits):
                pd, ph, pw = self.v_patch_nums[si]
                code = self.bsq.indices_to_code(bits.view(B, pd, ph, pw, self.L))
                q = code.permute(0, 4, 1, 2, 3).contiguous() * self._out_fact(si)
                if (pd, ph, pw) != (D, H, W):
                    q = F.interpolate(q, size=(D, H, W), mode=self.z_up)
                f_hat = f_hat + q
        return f_hat

    @torch.no_grad()
    def get_next_autoregressive_input(self, si: int, f_hat, q_BLDHW):
        """
        Inference-time step: q_BLDHW is the (already decay-scaled? no) sampled code
        at scale si, full-resolution upsampled. Adds to running f_hat and returns
        the area-downsampled conditioning for scale si+1.
        """
        D, H, W = self.v_patch_nums[-1]
        is_last = (si == self.K - 1)
        with torch.amp.autocast("cuda", enabled=False):
            f_hat = f_hat.float() + q_BLDHW.float()
            if is_last:
                return f_hat, f_hat
            pd, ph, pw = self.v_patch_nums[si + 1]
            return f_hat, F.interpolate(f_hat, size=(pd, ph, pw), mode=self.z_down)

    @torch.no_grad()
    def fhat_no_vq(self, f_BLDHW: torch.Tensor) -> torch.Tensor:
        """Same multiscale loop but skip binarization: code = q_scale * normalize(r).
        Lives in the SAME space as f_hat, so decode() of it is a valid no-VQ upper bound."""
        B, L, D, H, W = f_BLDHW.shape
        with torch.amp.autocast("cuda", enabled=False):
            residual = f_BLDHW.float()
            f_hat = torch.zeros_like(residual)
            for si, (pd, ph, pw) in enumerate(self.v_patch_nums):
                is_last = si == self.K - 1
                r = residual if is_last else F.interpolate(residual, size=(pd, ph, pw), mode=self.z_down)
                r = r.permute(0, 2, 3, 4, 1)
                code = self.bsq.q_scale * F.normalize(r, dim=-1)  # soft, no quantize()
                q = code.permute(0, 4, 1, 2, 3).contiguous() * self._out_fact(si)
                if not is_last:
                    q = F.interpolate(q, size=(D, H, W), mode=self.z_up)
                f_hat = f_hat + q
                residual = residual - q
            return f_hat

    def extra_repr(self) -> str:
        return (f"L={self.L} (vocab=2**{self.L}), v_patch_nums={self.v_patch_nums}, "
                f"K={self.K}, decay={self.use_decay_factor}")


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class BSQVAE3D(nn.Module):
    """
    Multi-scale BSQ VAE for 3D volumes. Mirrors MSVQVAE3D's init/forward so the
    two are drop-in swappable in your trainer.

    Note: `quant_embed_dim` here IS the number of bits L (the spherical code dim),
    so it is typically small (18-32), unlike the codebook-embedding dim in
    MSVQVAE3D. The pre/post-quant 1x1 convs project latent_dim <-> L.

    forward returns (mirrors MSVQVAE3D, `codes` stays None during VAE training):
        x_hat, vq_loss, codes(None), z_e, frac_unique
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 768,
        codebook_bits: int = 24,          # L; implicit vocab = 2**L
        resolution: int = 64,
        num_res_blocks_enc: int = 2,
        num_res_blocks_dec: int = 4,
        channels_enc=[64, 64, 256, 512, 512],
        channels_dec=[512, 512, 256, 64, 64],
        # multi-scale specific
        v_patch_nums=(1, 2, 3, 4, 6, 8),
        use_decay_factor: bool = True,
        # BSQ losses
        entropy_loss_weight: float = 0.1,
        commitment_loss_weight: float = 0.25,
        inv_temperature: float = 100.0,
        diversity_gamma: float = 1.0,
        gamma0: float = 1.0,
        zeta: float = 1.0,
        lfq_weight: float = 1.0,
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

        self.codebook_bits = codebook_bits

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

        # project to the L-dim spherical code and back
        self.pre_quant_conv = nn.Conv3d(latent_dim, codebook_bits, 1)
        self.post_quant_conv = nn.Conv3d(codebook_bits, latent_dim, 1)

        self.quantizer = MultiScaleBSQ3D(
            codebook_bits=codebook_bits,
            v_patch_nums=v_patch_nums,
            use_decay_factor=use_decay_factor,
            entropy_loss_weight=entropy_loss_weight,
            commitment_loss_weight=commitment_loss_weight,
            inv_temperature=inv_temperature,
            diversity_gamma=diversity_gamma,
            gamma0=gamma0,
            zeta=zeta,
            lfq_weight=lfq_weight,
        )

    def encode(self, x):
        return self.pre_quant_conv(self.encoder(x))

    def decode(self, z):
        return self.decoder(self.post_quant_conv(z))

    def forward(self, x):
        z_e = self.encode(x)
        f_hat, vq_loss, frac_unique = self.quantizer(z_e)
        x_hat = self.decode(f_hat)
        return x_hat, vq_loss, None, self.quantizer.fhat_no_vq(z_e), frac_unique

    # -------- helpers for transformer training later --------
    @torch.no_grad()
    def encode_multiscale(
        self, x,
        to_fhat: bool = False,
        bit_noise_fn: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        """Per-scale ground-truth bit maps (transformer targets), or cumulative f_hats
        if to_fhat=True. Pass `bit_noise_fn` to drive Bitwise Self-Correction (see
        MultiScaleBSQ3D.f_to_bits_or_fhat)."""
        z_e = self.encode(x)
        return self.quantizer.f_to_bits_or_fhat(z_e, to_fhat=to_fhat, bit_noise_fn=bit_noise_fn)

    @torch.no_grad()
    def encode_var_input(
        self, x,
        bit_noise_fn: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
        return_gt_bits: bool = False,
    ):
        """Infinity-style teacher-forcing `var_inputs` (list len K-1), and optionally
        the per-scale gt bit targets from the same pass. See
        MultiScaleBSQ3D.f_to_var_input."""
        z_e = self.encode(x)
        return self.quantizer.f_to_var_input(z_e, bit_noise_fn=bit_noise_fn, return_gt_bits=return_gt_bits)

    @torch.no_grad()
    def decode_multiscale(self, ms_bits: List[torch.Tensor]):
        """Decode per-scale bit maps back to a volume."""
        with torch.amp.autocast("cuda", enabled=False):
            f_hat = self.quantizer.bits_to_fhat(ms_bits)
        return self.decode(f_hat).clamp_(-1, 1)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_gpu_mem = (torch.cuda.get_device_properties(0).total_memory / 1e9
                     if torch.cuda.is_available() else 0)

    patch_size = 64
    model = BSQVAE3D(
        in_channels=1,
        latent_dim=768,
        codebook_bits=48,                 # implicit vocab 2**48
        channels_enc=[64, 64, 256, 512, 512],   # down_factor = 8 -> latent 8^3
        channels_dec=[512, 512, 256, 64, 64],
        resolution=patch_size,
        num_res_blocks_enc=2,
        num_res_blocks_dec=4,
        v_patch_nums=(1, 2, 3, 4, 6, 8),        # -> 828 tokens per volume
        use_decay_factor=True,
        skip_attn=True,
        use_checkpoint=True,
    ).to(device)

    print("Number of parameters, G", numel(model, only_trainable=True))
    model.train()

    x = torch.randn(1, 1, patch_size, patch_size, patch_size, device=device)
    x_hat, loss, codes, z_e, frac_unique = model(x)

    print(f"Input:            {tuple(x.shape)}")
    print(f"Output:           {tuple(x_hat.shape)}")
    print(f"VQ loss:          {loss.item():.4f}")
    print(f"Bit usage / scale: {[f'{u.item():.2f}' for u in frac_unique]}")

    ms_bits = model.encode_multiscale(x)
    print(f"Per-scale bit maps: {[tuple(t.shape) for t in ms_bits]} "
          f"(total tokens = {sum(t.shape[1] for t in ms_bits)}, L={model.codebook_bits})")

    x_rec = model.decode_multiscale(ms_bits)
    print(f"Round-trip decode: {tuple(x_rec.shape)}")

    if torch.cuda.is_available():
        max_memory_reserved = torch.cuda.max_memory_reserved()
        print("Max memory reserved: %0.3f Gb / %0.3f Gb"
              % (max_memory_reserved / 1e9, total_gpu_mem))
