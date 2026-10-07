"""
FiLM-conditioned per-scale-projection Multi-Scale BSQ VAE for 3D volumes (Variant 2).

Extends BSQVAE3DPerScaleProj (Variant 1). Variant 1 gives each of the K scales
its own STATIC projection `W_k` of the full latent. Variant 2 makes that
projection CONTENT-ADAPTIVE: scale k's projection is modulated (FiLM) by the
running reconstruction `f_hat_{<k}`, so each scale can steer its bits toward
whatever earlier scales left unexplained, per-sample and per-voxel:

        z_k      = FiLM_k( W_k(h) ; f_hat_{<k} )     # gamma,beta from f_hat
                 = W_k(h) * (1 + gamma_k) + beta_k
        target_k = z_k - f_hat_{<k}
        q_k      = BSQ(downsample(target_k))         # same L-bit code as before

Still free in bits / transformer cost:
    * Vocabulary 2**L and the K-scale sequence are unchanged.
    * The conditioning `f_hat_{<k}` is a function of already-emitted bits; nothing
      extra is transmitted. The DECODER never sees the projections (it just sums
      dequantized codes), so all decoder-side / token round-trip methods are
      inherited from Variant 1 unchanged.

Warm start (important): the FiLM generator's last layer is zero-initialized, so
`gamma = beta = 0` at step 0 and the projection is EXACTLY Variant 1. Variant 2
is a strict, smooth generalization - it departs from Variant 1 only if that
helps, which keeps early training stable and makes the A/B clean. At scale 0,
`f_hat = 0`, so the conditioning is zero and it falls back to a plain projection.

Drop-in: same constructor signature as BSQVAE3D (+ optional `film_hidden`), so
select_network can wire it with the identical opt_net keys.
"""

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.basic_vae import EncoderV3 as Encoder
from models.basic_vae import DecoderV3 as Decoder
from models.BSQVAE3D import Swish, _as_dhw
from models.BSQVAE3DPerScaleProj import MultiScaleBSQ3DPerScaleProj, BSQVAE3DPerScaleProj
from utils.utils_3D_image import numel


# ---------------------------------------------------------------------------
# FiLM-modulated per-scale projection
# ---------------------------------------------------------------------------
class FiLMProj3D(nn.Module):
    """Per-scale projection `W_k` with FiLM modulation from a conditioning map.

    `forward(h, cond)` projects the full latent `h` (latent_dim -> L) and then
    applies a feature-wise affine modulation whose (gamma, beta) are produced
    from `cond` (the running reconstruction f_hat, L channels). The FiLM
    generator is zero-initialized at the output, so at init gamma=beta=0 and the
    module is exactly `W_k(h)` (identical to Variant 1's static projection).
    """

    def __init__(self, latent_dim: int, codebook_bits: int, hidden: Optional[int] = None):
        super().__init__()
        self.proj = nn.Conv3d(latent_dim, codebook_bits, 1)          # W_k
        h = hidden or codebook_bits
        self.film = nn.Sequential(                                   # cond (L) -> (gamma, beta)
            nn.Conv3d(codebook_bits, h, 1),
            Swish(),
            nn.Conv3d(h, 2 * codebook_bits, 1),
        )
        # warm start: zero the output layer so gamma=beta=0 -> identity modulation
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        z = self.proj(h)
        # detach the conditioning: modulation is "context", gradient to the
        # encoder flows through proj(h); the FiLM weights still learn from the
        # current scale's loss (via the multiplicative z*(1+gamma) path).
        gamma, beta = self.film(cond.detach()).chunk(2, dim=1)
        return z * (1.0 + gamma) + beta


# ---------------------------------------------------------------------------
# Quantizer: FiLM-conditioned per-scale projection of the full latent
# ---------------------------------------------------------------------------
class MultiScaleBSQ3DFiLMProj(MultiScaleBSQ3DPerScaleProj):
    """Variant 1's quantizer with FiLM-conditioned per-scale projections.

    Only the projection call changes vs. Variant 1 (`...(h, cond=f_hat)`); the
    residual bookkeeping, keep/drop logic, refinement and decoder-side methods
    are identical. The three encoder-side methods are re-implemented here so
    Variant 1 stays untouched.
    """

    def __init__(self, latent_dim: int, codebook_bits: int, v_patch_nums,
                 film_hidden: Optional[int] = None, **kwargs):
        super().__init__(latent_dim=latent_dim, codebook_bits=codebook_bits,
                         v_patch_nums=v_patch_nums, **kwargs)
        # replace the static per-scale projections with FiLM-conditioned ones
        self.pre_quant_convs = nn.ModuleList([
            FiLMProj3D(latent_dim, codebook_bits, hidden=film_hidden) for _ in range(self.K)
        ])

    # ---------------------------------------------------------------
    # Training forward (operates on the FULL latent h)
    # ---------------------------------------------------------------
    def forward(self, h_BCDHW: torch.Tensor):
        B, C, D, H, W = h_BCDHW.shape

        with torch.amp.autocast("cuda", enabled=False):
            h = h_BCDHW.float()
            f_hat = h.new_zeros(B, self.L, D, H, W)

            all_losses: List[torch.Tensor] = []
            frac_unique: List[torch.Tensor] = [torch.tensor(torch.nan, device=h.device)] * self.K

            for si, (pd, ph, pw) in enumerate(self.v_patch_nums):
                is_last = (si == self.K - 1)

                # FiLM-conditioned per-scale projection, then residual vs. f_hat so far
                z_si = self.pre_quant_convs[si](h, cond=f_hat)         # (B, L, D, H, W)
                target = z_si - f_hat.detach()

                r = target if is_last else F.interpolate(target, size=(pd, ph, pw), mode=self.z_down)

                keep_first = si == 0 and self.keep_first_quant
                keep_last = is_last and self.keep_last_quant
                keep_scale = torch.rand(1, generator=self._rng).item() > self.scale_drop_rate if self.use_stochastic_depth else True
                if keep_scale or keep_first or keep_last or (not self.training):
                    q, _idx, bit_idx, aux_loss = self.bsq(r)
                    all_losses.append(aux_loss)
                    frac_unique[si] = self.bsq.normalized_bit_usage(bit_idx)
                else:
                    q = torch.zeros_like(r)

                q = self._upsample_and_refine(q, si)
                f_hat = f_hat + q

            vq_loss = torch.stack(all_losses).mean() * self.lfq_weight

        return f_hat, vq_loss, frac_unique

    # ---------------------------------------------------------------
    # Analysis / transformer data-prep (ground-truth bits)
    # ---------------------------------------------------------------
    @torch.no_grad()
    def f_to_bits_or_fhat(
        self,
        h_BCDHW: torch.Tensor,
        to_fhat: bool,
        v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int, int]]]] = None,
    ) -> List[torch.Tensor]:
        B, C, D, H, W = h_BCDHW.shape
        with torch.amp.autocast("cuda", enabled=False):
            patches = [_as_dhw(pn) for pn in (v_patch_nums or self.v_patch_nums)]
            assert patches[-1] == (D, H, W)
            assert len(patches) == self.K, (
                f"per-scale projection needs len(v_patch_nums)={len(patches)} == K={self.K} "
                f"(one projection per scale)"
            )

            h = h_BCDHW.float()
            f_hat = h.new_zeros(B, self.L, D, H, W)
            out: List[torch.Tensor] = []
            for si, (pd, ph, pw) in enumerate(patches):
                is_last = (si == len(patches) - 1)
                z_si = self.pre_quant_convs[si](h, cond=f_hat)
                target = z_si - f_hat
                r = target if is_last else F.interpolate(target, size=(pd, ph, pw), mode=self.z_down)
                q, _idx, bit_idx, _aux = self.bsq(r)
                q = self._upsample_and_refine(q, si)
                f_hat = f_hat + q
                out.append(f_hat.clone() if to_fhat else bit_idx.reshape(B, pd * ph * pw, self.L))
        return out

    @torch.no_grad()
    def fhat_no_vq(self, h_BCDHW: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = h_BCDHW.shape
        with torch.amp.autocast("cuda", enabled=False):
            h = h_BCDHW.float()
            f_hat = h.new_zeros(B, self.L, D, H, W)
            for si, (pd, ph, pw) in enumerate(self.v_patch_nums):
                is_last = si == self.K - 1
                z_si = self.pre_quant_convs[si](h, cond=f_hat)
                target = z_si - f_hat
                r = target if is_last else F.interpolate(target, size=(pd, ph, pw), mode=self.z_down)
                r = r.permute(0, 2, 3, 4, 1)
                code = self.bsq.q_scale * F.normalize(r, dim=-1)       # soft, no quantize()
                q = code.permute(0, 4, 1, 2, 3).contiguous()
                q = self._upsample_and_refine(q, si)
                f_hat = f_hat + q
            return f_hat

    def extra_repr(self) -> str:
        return super().extra_repr() + " + FiLM(cond=f_hat, zero-init -> Variant-1 warm start)"


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class BSQVAE3DFiLMProj(BSQVAE3DPerScaleProj):
    """BSQVAE3D with FiLM-conditioned per-scale projections (Variant 2).

    Same as BSQVAE3DPerScaleProj except the quantizer's per-scale projections are
    FiLM-modulated by the running reconstruction. `encode` (returns the full
    latent), `decode`, `forward` and every token round-trip helper are inherited.
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 768,
        codebook_bits: int = 24,
        resolution: int = 64,
        num_res_blocks_enc: int = 2,
        num_res_blocks_dec: int = 4,
        channels_enc=[64, 64, 256, 512, 512],
        channels_dec=[512, 512, 256, 64, 64],
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8),
        use_decay_factor: bool = True,
        quant_resi: float = 0.5,
        use_prog_quant_resi: bool = False,
        entropy_loss_weight: float = 0.1,
        commitment_loss_weight: float = 0.25,
        inv_temperature: float = 100.0,
        diversity_gamma: float = 1.0,
        gamma0: float = 1.0,
        zeta: float = 1.0,
        lfq_weight: float = 1.0,
        use_stochastic_depth: bool = True,
        scale_drop_rate: float = 0.25,
        keep_last_quant: bool = True,
        keep_first_quant: bool = False,
        skip_attn: bool = True,
        attn_resolutions=[16],
        use_checkpoint: bool = False,
        # variant-2 specific
        film_hidden: Optional[int] = None,
    ):
        # Skip the parent __init__ (which builds static per-scale projections);
        # build the pieces directly with the FiLM-conditioned quantizer instead.
        nn.Module.__init__(self)

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

        # No single pre_quant_conv; per-scale FiLM projections live in the quantizer.
        self.post_quant_conv = nn.Conv3d(codebook_bits, latent_dim, 1)

        self.quantizer = MultiScaleBSQ3DFiLMProj(
            latent_dim=latent_dim,
            codebook_bits=codebook_bits,
            v_patch_nums=v_patch_nums,
            film_hidden=film_hidden,
            use_decay_factor=use_decay_factor,
            quant_resi=quant_resi,
            use_prog_quant_resi=use_prog_quant_resi,
            entropy_loss_weight=entropy_loss_weight,
            commitment_loss_weight=commitment_loss_weight,
            inv_temperature=inv_temperature,
            diversity_gamma=diversity_gamma,
            gamma0=gamma0,
            zeta=zeta,
            lfq_weight=lfq_weight,
            use_stochastic_depth=use_stochastic_depth,
            scale_drop_rate=scale_drop_rate,
            keep_last_quant=keep_last_quant,
            keep_first_quant=keep_first_quant,
        )

    # encode / decode / forward / encode_multiscale / decode_multiscale inherited.


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_gpu_mem = (torch.cuda.get_device_properties(0).total_memory / 1e9
                     if torch.cuda.is_available() else 0)

    patch_size = 64
    model = BSQVAE3DFiLMProj(
        in_channels=1,
        latent_dim=768,
        codebook_bits=30,
        channels_enc=[64, 64, 256, 512, 512],
        channels_dec=[512, 512, 256, 64, 64],
        resolution=patch_size,
        num_res_blocks_enc=2,
        num_res_blocks_dec=4,
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8),
        use_decay_factor=True,
        quant_resi=0.5,
        use_prog_quant_resi=True,
        use_stochastic_depth=True,
        scale_drop_rate=0.50,
        keep_last_quant=True,
        keep_first_quant=False,
        skip_attn=False,
        use_checkpoint=True,
    ).to(device)

    print("Number of parameters, G", numel(model, only_trainable=True))
    print(model.quantizer.extra_repr())

    # warm-start identity check: at init FiLM(h, cond) == proj(h) for any cond
    with torch.no_grad():
        h_ = torch.randn(1, 768, 8, 8, 8, device=device)
        cond_ = torch.randn(1, 30, 8, 8, 8, device=device)   # arbitrary nonzero context
        p = model.quantizer.pre_quant_convs[3]
        same = torch.allclose(p(h_, cond_), p.proj(h_), atol=1e-6)
    print(f"Warm-start identity (FiLM==proj at init): {same}")

    model.train()
    x = torch.randn(1, 1, patch_size, patch_size, patch_size, device=device)
    x_hat, loss, codes, z_e, frac_unique = model(x)

    print(f"Input:            {tuple(x.shape)}")
    print(f"Output:           {tuple(x_hat.shape)}")
    print(f"VQ loss:          {loss.item():.4f}")
    print(f"Bit usage / scale: {[f'{u.item():.2f}' for u in frac_unique]}")

    # gradient sanity: loss must reach proj AND film of every active scale
    loss2 = (x_hat - x).pow(2).mean() + loss
    loss2.backward()

    def _has_grad(m):
        return any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in m.parameters())

    proj_grads = [_has_grad(c.proj) for c in model.quantizer.pre_quant_convs]
    film_grads = [_has_grad(c.film) for c in model.quantizer.pre_quant_convs]
    print(f"Per-scale proj grads: {proj_grads}")
    print(f"Per-scale FiLM grads: {film_grads}")

    model.eval()
    ms_bits = model.encode_multiscale(x)
    print(f"Per-scale bit maps: {[tuple(t.shape) for t in ms_bits]} "
          f"(total tokens = {sum(t.shape[1] for t in ms_bits)}, L={model.codebook_bits})")

    x_rec = model.decode_multiscale(ms_bits)
    print(f"Round-trip decode: {tuple(x_rec.shape)}")

    if torch.cuda.is_available():
        max_memory_reserved = torch.cuda.max_memory_reserved()
        print("Max memory reserved: %0.3f Gb / %0.3f Gb"
              % (max_memory_reserved / 1e9, total_gpu_mem))
