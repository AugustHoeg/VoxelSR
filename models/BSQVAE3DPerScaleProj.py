"""
Per-scale-projection Multi-Scale BSQ VAE for 3D volumes (Variant 1).

Extends BSQVAE3D with a single change, aimed at the reconstruction-capacity
bottleneck *without* adding any bits or transformer cost:

    Baseline BSQVAE3D projects the full `latent_dim` (e.g. 768) encoder features
    to the L-bit code ONCE, with a single static `pre_quant_conv`. Every one of
    the K multi-scale residual steps then operates purely inside that one
    projection's L-dim output. The 768 -> L squeeze is a single, irreversible
    commitment shared by all scales.

    Variant 1 replaces that single projection with K per-scale projections
    `W_0 ... W_{K-1}` (`pre_quant_convs`). At scale k the FULL latent is
    re-read through `W_k`, and the residual target is measured against what has
    already been reconstructed:

        z_k      = W_k(h)                 # full latent -> L, re-read per scale
        target_k = z_k - f_hat_{<k}       # what this scale's view still misses
        r_k      = downsample(target_k)
        q_k      = BSQ(r_k)               # same L-bit code as baseline
        f_hat   += upsample_refine(q_k)

Why this is "free":
    * The transformer still predicts K scales x L bits (vocabulary 2**L). The
      sequence length and per-token bit-width are unchanged.
    * The per-scale conditioning (`f_hat_{<k}`) is a function of already-emitted
      bits; nothing extra is transmitted. In fact the DECODER never sees the
      projections at all - `bits_to_fhat` / `get_next_autoregressive_input` just
      sum dequantized codes, so they are inherited from BSQVAE3D unchanged.
    * Setting every W_k to the same weights recovers the baseline forward pass,
      so this is a strict generalization: K independent "views" of the latent
      instead of one shared view.

What it does / does not do:
    It enlarges the *encoding function* (the map x -> bits), letting each scale
    choose better bits for the fixed decoder. It does NOT raise the hard L-bit
    capacity ceiling; it only closes the gap to it. Cost is a little extra
    encoder compute (K 1x1 convs) and zero extra transformer cost / bits.

Drop-in: same constructor signature as BSQVAE3D, so select_network can wire it
with the identical opt_net keys.
"""

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.basic_vae import EncoderV3 as Encoder
from models.basic_vae import DecoderV3 as Decoder
from models.BSQVAE3D import BSQVAE3D, MultiScaleBSQ3D, _as_dhw
from utils.utils_3D_image import numel


# ---------------------------------------------------------------------------
# Quantizer: per-scale projection of the full latent
# ---------------------------------------------------------------------------
class MultiScaleBSQ3DPerScaleProj(MultiScaleBSQ3D):
    """MultiScaleBSQ3D with one `latent_dim -> L` projection per scale.

    The residual loop operates on the full `latent_dim` features (not a single
    pre-projected L-dim latent): each scale re-reads them through its own 1x1
    conv and quantizes the residual against the running reconstruction. Only the
    three encoder-side methods are overridden; every decoder-side / token
    round-trip method is inherited from MultiScaleBSQ3D unchanged.
    """

    def __init__(self, latent_dim: int, codebook_bits: int, v_patch_nums, **kwargs):
        super().__init__(codebook_bits=codebook_bits, v_patch_nums=v_patch_nums, **kwargs)
        # one projection per scale; re-reads the full latent at every scale.
        self.pre_quant_convs = nn.ModuleList([
            nn.Conv3d(latent_dim, codebook_bits, 1) for _ in range(self.K)
        ])

    # ---------------------------------------------------------------
    # Training forward (operates on the FULL latent h, not L-dim f)
    # ---------------------------------------------------------------
    def forward(self, h_BCDHW: torch.Tensor):
        """
        h_BCDHW: (B, latent_dim, D, H, W) full encoder features (NOT pre-projected).

        Returns (same contract as MultiScaleBSQ3D.forward):
            f_hat:        (B, L, D, H, W) quantized latent (STE-connected).
            vq_loss:      scalar (entropy + commitment, scale-averaged & weighted).
            frac_unique:  list[Tensor] len K, per-scale normalized bit usage.
        """
        B, C, D, H, W = h_BCDHW.shape

        with torch.amp.autocast("cuda", enabled=False):
            h = h_BCDHW.float()
            f_hat = h.new_zeros(B, self.L, D, H, W)

            all_losses: List[torch.Tensor] = []
            frac_unique: List[torch.Tensor] = [torch.tensor(torch.nan, device=h.device)] * self.K

            for si, (pd, ph, pw) in enumerate(self.v_patch_nums):
                is_last = (si == self.K - 1)

                # per-scale projection of the full latent, then residual vs. f_hat
                # so far. f_hat is detached here to mirror BitVAE's detached peel
                # (gradient to the encoder/projection flows through z_si only).
                z_si = self.pre_quant_convs[si](h)                      # (B, L, D, H, W)
                target = z_si - f_hat.detach()

                # 1) downsample this scale's residual target
                r = target if is_last else F.interpolate(target, size=(pd, ph, pw), mode=self.z_down)

                # 2) quantize this scale's residual (same keep/drop logic as baseline)
                keep_first = si == 0 and self.keep_first_quant
                keep_last = is_last and self.keep_last_quant
                keep_scale = torch.rand(1, generator=self._rng).item() > self.scale_drop_rate if self.use_stochastic_depth else True
                if keep_scale or keep_first or keep_last or (not self.training):
                    q, _idx, bit_idx, aux_loss = self.bsq(r)
                    all_losses.append(aux_loss)
                    frac_unique[si] = self.bsq.normalized_bit_usage(bit_idx)
                else:
                    q = torch.zeros_like(r)

                # 3) upsample to full scale and optionally refine (inherited)
                q = self._upsample_and_refine(q, si)

                # 4) accumulate (grad path)
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
        """Per-scale bit maps (or cumulative f_hats) from the FULL latent.

        Mirrors MultiScaleBSQ3D.f_to_bits_or_fhat but re-reads `h` through the
        per-scale projections. `v_patch_nums`, if given, must match the trained
        scale count (one projection per scale)."""
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
                z_si = self.pre_quant_convs[si](h)
                target = z_si - f_hat
                r = target if is_last else F.interpolate(target, size=(pd, ph, pw), mode=self.z_down)
                q, _idx, bit_idx, _aux = self.bsq(r)
                q = self._upsample_and_refine(q, si)
                f_hat = f_hat + q
                out.append(f_hat.clone() if to_fhat else bit_idx.reshape(B, pd * ph * pw, self.L))
        return out

    @torch.no_grad()
    def fhat_no_vq(self, h_BCDHW: torch.Tensor) -> torch.Tensor:
        """No-VQ upper bound: same loop but soft code = q_scale * normalize(r),
        re-reading the full latent per scale. Lives in f_hat space, so decode()
        of it is a valid no-quantization reconstruction bound."""
        B, C, D, H, W = h_BCDHW.shape
        with torch.amp.autocast("cuda", enabled=False):
            h = h_BCDHW.float()
            f_hat = h.new_zeros(B, self.L, D, H, W)
            for si, (pd, ph, pw) in enumerate(self.v_patch_nums):
                is_last = si == self.K - 1
                z_si = self.pre_quant_convs[si](h)
                target = z_si - f_hat
                r = target if is_last else F.interpolate(target, size=(pd, ph, pw), mode=self.z_down)
                r = r.permute(0, 2, 3, 4, 1)
                code = self.bsq.q_scale * F.normalize(r, dim=-1)       # soft, no quantize()
                q = code.permute(0, 4, 1, 2, 3).contiguous()
                q = self._upsample_and_refine(q, si)
                f_hat = f_hat + q
            return f_hat

    def extra_repr(self) -> str:
        return super().extra_repr() + f", per_scale_proj=K x Conv3d(->L) ({self.K} projections)"


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class BSQVAE3DPerScaleProj(BSQVAE3D):
    """BSQVAE3D with per-scale projections (Variant 1).

    Identical to BSQVAE3D except there is no single `pre_quant_conv`: `encode`
    returns the full `latent_dim` features and the K per-scale projections live
    inside the quantizer. `decode` / `post_quant_conv` and every token
    round-trip helper are inherited unchanged (decoder-side is untouched).
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
    ):
        # Skip BSQVAE3D.__init__ (it builds the single pre_quant_conv + baseline
        # quantizer we are replacing); build the pieces directly instead.
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

        # No single pre_quant_conv: the per-scale projections live in the quantizer.
        # post_quant_conv (L -> latent_dim) is unchanged from BSQVAE3D.
        self.post_quant_conv = nn.Conv3d(codebook_bits, latent_dim, 1)

        self.quantizer = MultiScaleBSQ3DPerScaleProj(
            latent_dim=latent_dim,
            codebook_bits=codebook_bits,
            v_patch_nums=v_patch_nums,
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

    def encode(self, x):
        # Return the FULL latent; per-scale projection happens in the quantizer.
        return self.encoder(x)

    # decode / forward / encode_multiscale / decode_multiscale are inherited:
    # forward() calls self.encode (full latent) -> self.quantizer (handles the
    # per-scale projection) -> self.decode (post_quant_conv -> decoder), which is
    # exactly the right wiring for this variant.


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_gpu_mem = (torch.cuda.get_device_properties(0).total_memory / 1e9
                     if torch.cuda.is_available() else 0)

    patch_size = 64
    model = BSQVAE3DPerScaleProj(
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
    model.train()

    x = torch.randn(1, 1, patch_size, patch_size, patch_size, device=device)
    x_hat, loss, codes, z_e, frac_unique = model(x)

    print(f"Input:            {tuple(x.shape)}")
    print(f"Output:           {tuple(x_hat.shape)}")
    print(f"VQ loss:          {loss.item():.4f}")
    print(f"Bit usage / scale: {[f'{u.item():.2f}' for u in frac_unique]}")

    # gradient sanity: loss must reach every per-scale projection
    loss2 = (x_hat - x).pow(2).mean() + loss
    loss2.backward()
    grads = [c.weight.grad is not None and c.weight.grad.abs().sum().item() > 0
             for c in model.quantizer.pre_quant_convs]
    print(f"Per-scale proj grads present: {grads}")

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
