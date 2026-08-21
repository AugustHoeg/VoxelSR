"""
Multi-Scale *Residual-Depth* Binary Spherical Quantization (RBSQ) VAE for 3D volumes.

Extends BSQVAE3D with an inner residual-DEPTH loop: at each spatial scale the
shrinking inner residual is quantized `depth` times by the shared parameter-free
BSQ, each code scaled by a hand-set geometric per-depth decay, then summed before
upsampling. This gives RQ-style residual refinement *at each scale's resolution*
(the finest scale therefore competes directly with RQVAE's full-res depth) while
keeping the multi-scale scaffolding. depth=1 reproduces BSQVAE3D exactly.

Only the forward / reconstruction paths (forward, fhat_no_vq) implement depth for
now; the bit-map / transformer-prep helpers are guarded (NotImplementedError) for
depth>1 until the transformer phase.

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
# Phi refinement convs (per-scale 3x3x3 residual convs)
# ---------------------------------------------------------------------------
# Optional learned alternative to the hand-set `out_fact` decay: a small conv
# applied AFTER upsampling each scale's code, so it can compensate the
# high-frequency loss of trilinear interpolation (VAR's Phi mechanism). Mirrors
# MSVQVAE3D's Phi so the two tokenizers stay comparable; kept local to keep
# RBSQVAE3D self-contained.
class Phi3D(nn.Conv3d):
    """(1 - r) * x + r * conv(x). Small learned refine applied after upsampling."""
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
    """N Phi's, resolved to the nearest scale by tick on [0, 1]."""
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
class MultiScaleRBSQ3D(nn.Module):
    """
    Coarse-to-fine residual quantization on the unit hypersphere, with an inner
    residual-DEPTH loop at each scale (RBSQ). At every scale the shrinking inner
    residual is quantized `depth` times by the shared, parameter-free BSQ, each
    code scaled by a hand-set per-depth decay, then summed before upsampling.
    depth=1 reproduces plain multi-scale BSQ exactly.

    Args:
        codebook_bits:       L, spherical dimensions / bits per token.
        v_patch_nums:        list of scales, small -> large. int (isotropic) or
                             (D, H, W). The LAST scale must equal the encoder
                             output resolution.
        depth:               number of inner residual-BSQ steps per scale (D),
                             uniform across scales. depth=1 == plain BSQ.
        depth_decay:         geometric per-depth magnitude base; the code at inner
                             step d is scaled by depth_decay**d (depth_fact(0)=1).
                             Shrinking step sizes let the unit-norm codes refine
                             below the code magnitude without overshooting; the best
                             value interacts with the encoder's learned latent scale,
                             so it is the primary knob to ablate (see _depth_fact).
        use_decay_factor:    if True, scale-s code magnitude = max(0.1, 1-0.1*s);
                             the BitVAE mechanism that lets unit-norm codes act as
                             shrinking residual corrections. If False, out_fact=1.
                             Only used on the out_fact fallback path (share_quant_resi=None).
        quant_resi:          residual ratio for the learned Phi refine conv.
                             0.5 => 0.5*conv(x) + 0.5*x. Only used when
                             share_quant_resi is not None.
        share_quant_resi:    None (default) => no Phi; fall back to the out_fact
                             decay (original BitVAE behaviour). Otherwise a learned
                             Phi conv is applied after each upsample instead:
                             1 = single shared Phi; N>1 = N Phi's mapped by tick;
                             0 = one Phi per scale (heavy).
        z_down / z_up:       interpolation modes for residual down / code up.
        entropy_loss_weight, commitment_loss_weight, inv_temperature,
        diversity_gamma:     passed to BSQ3D.
    """

    def __init__(
        self,
        codebook_bits: int,
        v_patch_nums: Sequence[Union[int, Tuple[int, int, int]]],
        depth: int = 1,
        depth_decay: float = 0.5,
        use_decay_factor: bool = True,
        quant_resi: float = 0.5,
        share_quant_resi: Optional[int] = None,
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
        self.depth = depth
        self.depth_decay = depth_decay
        self.use_decay_factor = use_decay_factor
        self.z_down = z_down
        self.z_up = z_up
        self.lfq_weight = lfq_weight

        # ---- Phi refinement convs (learned upsampler; replaces out_fact decay) ----
        # share_quant_resi=None -> disabled: fall back to the out_fact magnitude decay.
        self.quant_resi_ratio = quant_resi
        if share_quant_resi is None:
            self.quant_resi = None
        else:
            def _mk_phi():
                return Phi3D(codebook_bits, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()
            if share_quant_resi == 0:      # non-shared: one per scale
                self.quant_resi = PhiPartiallyShared3D(nn.ModuleList([_mk_phi() for _ in range(self.K)]))
            elif share_quant_resi == 1:    # fully shared
                self.quant_resi = PhiShared3D(_mk_phi())
            else:                          # partially shared
                self.quant_resi = PhiPartiallyShared3D(
                    nn.ModuleList([_mk_phi() for _ in range(share_quant_resi)])
                )

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

    def _depth_fact(self, d: int) -> float:
        """Hand-set per-depth magnitude decay (geometric): depth_fact(d)=decay**d.
        depth_fact(0)=1 so depth=1 reproduces single-step BSQ exactly. Successive
        inner codes are scaled down so the unit-norm spherical codes act as finer
        residual corrections instead of overshooting once the residual drops toward
        the code magnitude. The optimal schedule interacts with the encoder's
        (freely learned) latent magnitude, so `depth_decay` is the primary knob to
        ablate -- decay<1 refines below the code magnitude but too-aggressive decay
        caps the total code budget (sum_d decay**d) and plateaus early."""
        return self.depth_decay ** d

    def _quantize_scale(self, r_BLDHW: torch.Tensor, si: int):
        """Inner residual-depth loop at this scale's native resolution.

        Runs `depth` shared parameter-free BSQ steps on the shrinking inner
        residual, scales each code by the hand-set per-depth decay, peels it
        (detached, like BitVAE) and sums into this scale's code (pre-upsample).

        Returns:
            q_acc:        (B, L, pd, ph, pw) summed depth code at scale resolution.
            aux_losses:   list[Tensor] length depth, per-depth BSQ aux loss.
            depth_usage:  (depth,) tensor, per-depth normalized bit usage.
        """
        r_inner = r_BLDHW
        q_acc = torch.zeros_like(r_inner)
        aux_losses: List[torch.Tensor] = []
        depth_usage: List[torch.Tensor] = []
        for d in range(self.depth):
            code_d, _idx, bit_d, aux_d = self.bsq(r_inner)
            code_d = code_d * self._depth_fact(d)
            r_inner = r_inner - code_d.detach()      # peel inner residual (scale res)
            q_acc = q_acc + code_d
            aux_losses.append(aux_d)
            depth_usage.append(self.bsq.normalized_bit_usage(bit_d))
        return q_acc, aux_losses, torch.stack(depth_usage)

    def _require_depth1(self, where: str) -> None:
        """The bit-map / transformer-prep paths emit one code per (scale) position;
        residual depth (D codes per position) is not wired into them yet. Guard so
        depth>1 fails loudly instead of returning a depth=1-only reconstruction."""
        if self.depth > 1:
            raise NotImplementedError(
                f"{where}: residual depth>1 is not yet supported on the "
                f"bit/transformer path -- only forward()/fhat_no_vq handle depth "
                f"for now (tokenizer-recon phase). depth={self.depth}.")

    def _upsample_and_refine(self, q: torch.Tensor, si: int,
                             n_scales: Optional[int] = None) -> torch.Tensor:
        """Upsample this scale's code to the full latent grid and apply per-scale
        refinement. With Phi enabled (share_quant_resi != None) the learned conv is
        applied *after* upsampling (like VAR) so it can compensate trilinear loss;
        otherwise fall back to the scalar out_fact decay (order-invariant on that
        path). Single source of truth shared by every f_hat-building method.

        n_scales overrides self.K for the analysis path that runs a custom-length
        v_patch_nums, so is_last / the Phi tick index resolve against the right K.
        """
        D, H, W = self.v_patch_nums[-1]
        K = self.K if n_scales is None else n_scales
        is_last = (si == K - 1)
        if self.quant_resi is not None:
            if not is_last:
                q = F.interpolate(q, size=(D, H, W), mode=self.z_up)
            q = self.quant_resi[si / max(K - 1, 1)](q)
        else:
            q = q * self._out_fact(si)
            if not is_last:
                q = F.interpolate(q, size=(D, H, W), mode=self.z_up)
        return q.contiguous()

    # ---------------------------------------------------------------
    # Training forward
    # ---------------------------------------------------------------
    def forward(self, f_BLDHW: torch.Tensor):
        """
        f_BLDHW: (B, L, D, H, W) encoder output projected to L channels.

        Returns:
            f_hat:                (B, L, D, H, W) quantized latent (STE to input).
            vq_loss:              scalar (entropy + commitment, averaged over
                                  scale*depth & weighted).
            frac_unique:          list[Tensor] len K, per-scale normalized bit usage
                                  in [0,1], AVERAGED over depth (kept scalar-per-scale
                                  for continuity with the depth=1 experiments / EMA).
            frac_unique_per_depth:list[Tensor] len K, each (depth,), bit usage split
                                  by inner depth step so depth collapse is visible.
        """
        B, L, D, H, W = f_BLDHW.shape
        assert L == self.L, f"encoder produced {L} channels, expected L={self.L}"

        with torch.amp.autocast("cuda", enabled=False):
            f = f_BLDHW.float()
            residual = f
            f_hat = torch.zeros_like(f)

            all_losses: List[torch.Tensor] = []
            frac_unique: List[torch.Tensor] = []
            frac_unique_per_depth: List[torch.Tensor] = []

            for si, (pd, ph, pw) in enumerate(self.v_patch_nums):
                is_last = (si == self.K - 1)

                # 1) downsample running residual to this scale
                r = residual if is_last else F.interpolate(residual, size=(pd, ph, pw), mode=self.z_down)

                # 2) inner residual-depth loop of binary spherical quantize (scale res)
                q, aux_losses, depth_usage = self._quantize_scale(r, si)

                # 3) upsample to full grid + per-scale refine (Phi conv, else out_fact decay)
                q = self._upsample_and_refine(q, si)

                # 4) accumulate (grad path) + peel residual (detached, like BitVAE)
                f_hat = f_hat + q
                residual = residual - q.detach()

                all_losses.extend(aux_losses)
                frac_unique.append(depth_usage.mean())        # scalar, avg over depth
                frac_unique_per_depth.append(depth_usage)     # (depth,)

            # reference: stack per-(scale,depth) aux_loss; reduce with mean * lfq_weight
            vq_loss = torch.stack(all_losses).mean() * self.lfq_weight

        return f_hat, vq_loss, frac_unique, frac_unique_per_depth

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
        self._require_depth1("f_to_bits_or_fhat")
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

                q = self._upsample_and_refine(q, si, n_scales=len(patches))
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
        self._require_depth1("f_to_var_input")
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

                q = self._upsample_and_refine(q, si)
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
        self._require_depth1("bits_to_fhat")
        B = ms_bits[0].shape[0]
        D, H, W = self.v_patch_nums[-1]
        with torch.amp.autocast("cuda", enabled=False):
            f_hat = ms_bits[0].new_zeros(B, self.L, D, H, W, dtype=torch.float32)
            for si, bits in enumerate(ms_bits):
                pd, ph, pw = self.v_patch_nums[si]
                code = self.bsq.indices_to_code(bits.view(B, pd, ph, pw, self.L))
                q = code.permute(0, 4, 1, 2, 3).contiguous()
                q = self._upsample_and_refine(q, si)
                f_hat = f_hat + q
        return f_hat

    @torch.no_grad()
    def get_next_autoregressive_input(self, si: int, f_hat, q_BLDHW):
        """
        Inference-time step: q_BLDHW is this scale's sampled code at scale-si
        resolution (B, L, pd, ph, pw), i.e. BEFORE upsampling/refinement. It is
        upsampled to the full grid and passed through the same per-scale refinement
        as training (Phi conv, else out_fact decay), added to the running f_hat, and
        the area-downsampled conditioning for scale si+1 is returned. Mirrors
        MSVQVAE3D.get_next_autoregressive_input's contract (per-scale code in).
        """
        self._require_depth1("get_next_autoregressive_input")
        is_last = (si == self.K - 1)
        with torch.amp.autocast("cuda", enabled=False):
            q = self._upsample_and_refine(q_BLDHW.float(), si)
            f_hat = f_hat.float() + q
            if is_last:
                return f_hat, f_hat
            pd, ph, pw = self.v_patch_nums[si + 1]
            return f_hat, F.interpolate(f_hat, size=(pd, ph, pw), mode=self.z_down)

    @torch.no_grad()
    def fhat_no_vq(self, f_BLDHW: torch.Tensor) -> torch.Tensor:
        """Same multiscale + inner-depth loop but skip binarization: each inner code
        is q_scale * normalize(r) instead of the binarized code. Lives in the SAME
        space as f_hat (same scale/depth scaffolding, decay factors and refine), so
        decode() of it is a valid no-VQ upper bound for the depth-D architecture."""
        B, L, D, H, W = f_BLDHW.shape
        with torch.amp.autocast("cuda", enabled=False):
            residual = f_BLDHW.float()
            f_hat = torch.zeros_like(residual)
            for si, (pd, ph, pw) in enumerate(self.v_patch_nums):
                is_last = si == self.K - 1
                r = residual if is_last else F.interpolate(residual, size=(pd, ph, pw), mode=self.z_down)

                # inner depth loop, soft (mirrors _quantize_scale without quantize())
                r_inner = r
                q_acc = torch.zeros_like(r_inner)
                for d in range(self.depth):
                    code = self.bsq.q_scale * F.normalize(r_inner.permute(0, 2, 3, 4, 1), dim=-1)
                    code = code.permute(0, 4, 1, 2, 3).contiguous() * self._depth_fact(d)
                    r_inner = r_inner - code
                    q_acc = q_acc + code

                q = self._upsample_and_refine(q_acc, si)
                f_hat = f_hat + q
                residual = residual - q
            return f_hat

    def extra_repr(self) -> str:
        refine = (f"Phi(quant_resi={self.quant_resi_ratio})"
                  if self.quant_resi is not None else f"out_fact(decay={self.use_decay_factor})")
        return (f"L={self.L} (vocab=2**{self.L}), v_patch_nums={self.v_patch_nums}, "
                f"K={self.K}, depth={self.depth} (decay={self.depth_decay}), refine={refine}")


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class RBSQVAE3D(nn.Module):
    """
    Multi-scale *residual-depth* BSQ VAE for 3D volumes. Same as BSQVAE3D but the
    quantizer runs `depth` inner residual-BSQ steps per scale (depth=1 == BSQVAE3D),
    kept drop-in swappable with MSVQVAE3D / RQVAE3D in your trainer.

    Note: `codebook_bits` here IS the number of bits L (the spherical code dim), so
    it is typically small (18-48), unlike the codebook-embedding dim in MSVQVAE3D.
    The pre/post-quant 1x1 convs project latent_dim <-> L.

    forward returns (mirrors MSVQVAE3D, `codes` stays None during VAE training):
        x_hat, vq_loss, codes(None), z_no_vq, frac_unique
    Per-depth bit usage is stashed on self.frac_unique_per_depth (list len K of
    (depth,) tensors) so the 5-tuple trainer contract is preserved.
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
        depth: int = 1,
        depth_decay: float = 0.5,
        use_decay_factor: bool = True,
        quant_resi: float = 0.5,
        share_quant_resi: Optional[int] = None,
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

        self.quantizer = MultiScaleRBSQ3D(
            codebook_bits=codebook_bits,
            v_patch_nums=v_patch_nums,
            depth=depth,
            depth_decay=depth_decay,
            use_decay_factor=use_decay_factor,
            quant_resi=quant_resi,
            share_quant_resi=share_quant_resi,
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
        f_hat, vq_loss, frac_unique, frac_unique_per_depth = self.quantizer(z_e)
        x_hat = self.decode(f_hat)
        return x_hat, vq_loss, None, self.quantizer.fhat_no_vq(z_e), frac_unique, frac_unique_per_depth

    # -------- helpers for transformer training later --------
    @torch.no_grad()
    def encode_multiscale(
        self, x,
        to_fhat: bool = False,
        bit_noise_fn: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        """Per-scale ground-truth bit maps (transformer targets), or cumulative f_hats
        if to_fhat=True. Pass `bit_noise_fn` to drive Bitwise Self-Correction (see
        MultiScaleRBSQ3D.f_to_bits_or_fhat)."""
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
        MultiScaleRBSQ3D.f_to_var_input."""
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
    depth = 3
    model = RBSQVAE3D(
        in_channels=1,
        latent_dim=768,
        codebook_bits=24,                 # implicit vocab 2**24
        channels_enc=[64, 64, 256, 512, 512],   # down_factor = 8 -> latent 8^3
        channels_dec=[512, 512, 256, 64, 64],
        resolution=patch_size,
        num_res_blocks_enc=2,
        num_res_blocks_dec=4,
        v_patch_nums=(1, 2, 3, 4, 6, 8),        # -> 828 tokens per volume
        depth=depth,                            # inner residual-BSQ steps per scale
        depth_decay=0.7,                        # geometric per-depth magnitude decay
        use_decay_factor=True,                  # fallback path (share_quant_resi=None)
        quant_resi=0.5,                         # learned Phi refine (used only if
        share_quant_resi=4,                     # share_quant_resi is not None)
        skip_attn=True,
        use_checkpoint=True,
    ).to(device)

    print("Number of parameters, G", numel(model, only_trainable=True))
    print(model.quantizer.extra_repr())
    model.train()

    x = torch.randn(1, 1, patch_size, patch_size, patch_size, device=device)
    x_hat, loss, codes, z_no_vq, frac_unique = model(x)

    print(f"Input:            {tuple(x.shape)}")
    print(f"Output:           {tuple(x_hat.shape)}")
    print(f"VQ loss:          {loss.item():.4f}")
    print(f"Bit usage / scale (avg over depth): {[f'{u.item():.2f}' for u in frac_unique]}")
    print("Bit usage / (scale x depth):")
    for si, du in enumerate(model.frac_unique_per_depth):
        print(f"  scale {si}: {[f'{v:.2f}' for v in du.tolist()]}")

    # bit-map / round-trip decode is guarded for depth>1 (transformer phase pending)
    if model.quantizer.depth == 1:
        ms_bits = model.encode_multiscale(x)
        print(f"Per-scale bit maps: {[tuple(t.shape) for t in ms_bits]} "
              f"(total tokens = {sum(t.shape[1] for t in ms_bits)}, L={model.codebook_bits})")
        x_rec = model.decode_multiscale(ms_bits)
        print(f"Round-trip decode: {tuple(x_rec.shape)}")
    else:
        print(f"Skipping bit round-trip: depth={model.quantizer.depth}>1 not yet on bit path.")

    if torch.cuda.is_available():
        max_memory_reserved = torch.cuda.max_memory_reserved()
        print("Max memory reserved: %0.3f Gb / %0.3f Gb"
              % (max_memory_reserved / 1e9, total_gpu_mem))
