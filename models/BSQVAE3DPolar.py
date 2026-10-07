"""
Polar (magnitude-augmented) Multi-Scale BSQ VAE for 3D volumes (Design 2).

Motivation
----------
Plain BSQ encodes each token as `(1/sqrt(L)) * sign(z)`: `normalize -> sign`
throws away the RADIUS `||z||` and keeps only an equal-norm hypercube corner.
Magnitude is only recovered crudely and globally via the hand-set `out_fact`
decay across scales. RQVAE3D (learned codebook in full R^768, arbitrary norm +
direction, depth-6) converges much faster and reconstructs better precisely
because it represents magnitude natively. This variant gives the radius back to
BSQ while staying bitwise-friendly for the Infinity transformer.

Polar BSQ (per token)
---------------------
    direction = (1/sqrt(L)) * sign(z)          # L bits, norm 1 (as in BSQ)
    radius    = rho_m,  m = argmin_j |  ||z|| - rho_j |   # learned 1-D codebook, M levels
    code      = radius * direction             # lands on one of M concentric shells

Per token we transmit the L sign bits PLUS one magnitude index (log2 M bits).
Magnitude is low-entropy and highly predictable, so those few bits cost the
transformer far less than the same number of *direction* bits - and they
re-introduce exactly the radial DOF BSQ was discarding.

Gradients (standard 1-D VQ-VAE composed with the sign direction):
    * direction of z  <- straight-through on sign (through F.normalize -> tangential)
    * magnitude ||z|| <- straight-through on the level selection (radial)
    * the M levels    <- codebook loss ||sg[||z||] - rho_m||^2 (+ commitment)

Warm start: levels are initialized tightly around 1.0, so at init every token
picks a radius ~= 1 and the forward pass ~= baseline BSQ (unit-norm codes). The
levels then spread to cover the encoder's magnitude distribution. Set
`n_mag_levels=1` to recover BSQ exactly.

Drop-in: same constructor signature as BSQVAE3D plus the `*_mag_*` knobs, so
select_network can wire it with the identical opt_net keys (+ the new ones).
Decoder (post_quant_conv L->latent_dim) is unchanged; the transformer's
teacher-forcing (BitwiseSelfCorrection3D) will need extending to also carry the
magnitude indices when you train the AR model on this tokenizer.
"""

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.basic_vae import EncoderV3 as Encoder
from models.basic_vae import DecoderV3 as Decoder
from models.BSQVAE3D import BSQ3D, MultiScaleBSQ3D, BSQVAE3D, _as_dhw
from utils.utils_3D_image import numel


# ---------------------------------------------------------------------------
# Polar Binary Spherical Quantizer (sign direction + learned radius codebook)
# ---------------------------------------------------------------------------
class PolarBSQ3D(BSQ3D):
    """BSQ3D with a learned 1-D magnitude (radius) codebook.

    Returns one extra tensor vs. BSQ3D: the per-voxel magnitude index.
    """

    def __init__(
        self,
        codebook_bits: int,
        n_mag_levels: int = 4,
        mag_commit_weight: float = 0.25,
        mag_loss_weight: float = 1.0,
        mag_init_spread: float = 0.1,
        **bsq_kwargs,
    ):
        super().__init__(codebook_bits=codebook_bits, **bsq_kwargs)
        self.M = n_mag_levels
        self.mag_commit_weight = mag_commit_weight
        self.mag_loss_weight = mag_loss_weight
        # radius codebook, initialized tight around 1.0 so init ~= unit-norm BSQ
        if n_mag_levels == 1:
            init = torch.ones(1)
        else:
            init = torch.linspace(1.0 - mag_init_spread, 1.0 + mag_init_spread, n_mag_levels)
        self.mag_levels = nn.Parameter(init)

    def _rho(self) -> torch.Tensor:
        return self.mag_levels.clamp_min(1e-4)  # keep radii strictly positive

    def forward(self, z_BLDHW: torch.Tensor):
        """
        z_BLDHW: (B, L, D, H, W) projected residual for this scale.

        Returns:
            quantized:    (B, L, D, H, W) polar code (radius * sign), STE-connected.
            indices:      None (kept for interface parity with BSQ3D).
            bit_indices:  (B, D, H, W, L) int  (direction bits, transformer targets).
            mag_indices:  (B, D, H, W) long    (radius level index, extra target).
            aux_loss:     scalar (direction commit + entropy penalty + magnitude VQ).
        """
        z = z_BLDHW.permute(0, 2, 3, 4, 1).contiguous()          # (B, D, H, W, L)
        z = z.float()

        m = z.norm(dim=-1)                                       # (B, D, H, W) radius ||z||
        u = F.normalize(z, dim=-1)                               # unit direction

        with torch.amp.autocast('cuda', enabled=False):
            # ---- direction: sign on the sphere (identical to BSQ) ----
            dir_code = self.quantize(u)                          # sign STE, (B,D,H,W,L) in {-1,+1}
            dir_code = self.q_scale * dir_code                   # norm 1
            bit_indices = (dir_code > 0).int()

            # ---- radius: nearest level in the learned 1-D codebook ----
            rho = self._rho()                                    # (M,)
            dists = (m.unsqueeze(-1) - rho).abs()                # (B,D,H,W,M)
            mag_indices = dists.argmin(dim=-1)                   # (B,D,H,W)
            rho_sel = rho[mag_indices]                           # (B,D,H,W)
            # VQ-VAE straight-through: forward uses rho_sel, gradient flows to m
            # (encoder magnitude); the codebook itself learns via mag_loss below.
            mag_ste = m + (rho_sel - m).detach()                 # (B,D,H,W)

            # ---- polar code = radius * unit-direction ----
            code = dir_code * mag_ste.unsqueeze(-1)              # (B,D,H,W,L), norm = rho_sel

            # ---- losses ----
            per_sample_entropy, codebook_entropy, _ = self.soft_entropy_loss(u)
            entropy_penalty = self.gamma0 * per_sample_entropy - self.diversity_gamma * codebook_entropy
            commit_dir = F.mse_loss(u, dir_code.detach())        # direction commitment (unit space)

            # 1-D VQ on magnitude: codebook pulls levels to data; commit pulls encoder to levels
            mag_codebook = F.mse_loss(rho_sel, m.detach())
            mag_commit = F.mse_loss(m, rho_sel.detach())
            mag_loss = mag_codebook + self.mag_commit_weight * mag_commit

            aux_loss = (commit_dir * self.commitment_loss_weight
                        + (self.zeta * entropy_penalty / self.inv_temperature) * self.entropy_loss_weight
                        + self.mag_loss_weight * mag_loss)

            quantized = code.permute(0, 4, 1, 2, 3).contiguous()  # (B, L, D, H, W)
            return quantized, None, bit_indices, mag_indices, aux_loss

    def indices_to_code(self, bit_indices: torch.Tensor, mag_indices: torch.Tensor) -> torch.Tensor:
        """(bits (...,L) {0,1}, mag (...) long) -> polar code (...,L)."""
        dir_code = self.bits_to_code(bit_indices) * self.q_scale      # (...,L), norm 1
        rho_sel = self._rho()[mag_indices]                            # (...,)
        return dir_code * rho_sel.unsqueeze(-1)

    @torch.no_grad()
    def normalized_mag_usage(self, mag_indices: torch.Tensor) -> torch.Tensor:
        """Diagnostic in [0, 1]: entropy of the level histogram / log(M). ~1 => levels balanced."""
        if self.M == 1:
            return torch.zeros((), device=mag_indices.device)
        p = torch.bincount(mag_indices.reshape(-1), minlength=self.M).float()
        p = p / p.sum().clamp_min(1.0)
        ent = -(p * (p + 1e-8).log()).sum()
        return ent / torch.log(torch.tensor(float(self.M), device=mag_indices.device))


# ---------------------------------------------------------------------------
# Multi-scale residual Polar-BSQ bottleneck
# ---------------------------------------------------------------------------
class MultiScaleBSQ3DPolar(MultiScaleBSQ3D):
    """MultiScaleBSQ3D whose per-scale quantizer is PolarBSQ3D.

    forward keeps the (f_hat, vq_loss, frac_unique) contract (magnitude is baked
    into the codes). The token round-trip methods carry the magnitude index so
    decode_multiscale reconstructs faithfully.
    """

    def __init__(self, codebook_bits: int, v_patch_nums,
                 n_mag_levels: int = 4, mag_commit_weight: float = 0.25,
                 mag_loss_weight: float = 1.0, mag_init_spread: float = 0.1, **kwargs):
        super().__init__(codebook_bits=codebook_bits, v_patch_nums=v_patch_nums, **kwargs)
        # swap the plain BSQ3D built by super() for the polar version, copying the
        # loss hyper-params it already stored (avoids re-listing the kwargs).
        b = self.bsq
        self.bsq = PolarBSQ3D(
            codebook_bits=codebook_bits,
            entropy_loss_weight=b.entropy_loss_weight,
            commitment_loss_weight=b.commitment_loss_weight,
            inv_temperature=b.inv_temperature,
            diversity_gamma=b.diversity_gamma,
            gamma0=b.gamma0,
            zeta=b.zeta,
            n_mag_levels=n_mag_levels,
            mag_commit_weight=mag_commit_weight,
            mag_loss_weight=mag_loss_weight,
            mag_init_spread=mag_init_spread,
        )

    # ---------------------------------------------------------------
    # Training forward (same scaffold as BSQVAE3D, polar bsq returns 5-tuple)
    # ---------------------------------------------------------------
    def forward(self, f_BLDHW: torch.Tensor):
        B, L, D, H, W = f_BLDHW.shape
        assert L == self.L, f"encoder produced {L} channels, expected L={self.L}"

        with torch.amp.autocast("cuda", enabled=False):
            f = f_BLDHW.float()
            residual = f
            f_hat = torch.zeros_like(f)

            all_losses: List[torch.Tensor] = []
            frac_unique: List[torch.Tensor] = [torch.tensor(torch.nan, device=f_BLDHW.device)] * self.K

            for si, (pd, ph, pw) in enumerate(self.v_patch_nums):
                is_last = (si == self.K - 1)
                r = residual if is_last else F.interpolate(residual, size=(pd, ph, pw), mode=self.z_down)

                keep_first = si == 0 and self.keep_first_quant
                keep_last = is_last and self.keep_last_quant
                keep_scale = torch.rand(1, generator=self._rng).item() > self.scale_drop_rate if self.use_stochastic_depth else True
                if keep_scale or keep_first or keep_last or (not self.training):
                    q, _idx, bit_idx, _mag_idx, aux_loss = self.bsq(r)
                    all_losses.append(aux_loss)
                    frac_unique[si] = self.bsq.normalized_bit_usage(bit_idx)
                else:
                    q = torch.zeros_like(r)

                q = self._upsample_and_refine(q, si)
                f_hat = f_hat + q
                residual = residual - q.detach()

            vq_loss = torch.stack(all_losses).mean() * self.lfq_weight

        return f_hat, vq_loss, frac_unique

    # ---------------------------------------------------------------
    # Analysis / transformer data-prep (bits + magnitude indices)
    # ---------------------------------------------------------------
    @torch.no_grad()
    def f_to_bits_or_fhat(
        self,
        f_BLDHW: torch.Tensor,
        to_fhat: bool,
        v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int, int]]]] = None,
    ) -> List:
        """to_fhat=True  -> list[Tensor(B, L, D, H, W)]  cumulative reconstructions.
        to_fhat=False -> list[(bits (B, l_k, L) int, mag (B, l_k) long)] per scale."""
        B, L, D, H, W = f_BLDHW.shape
        with torch.amp.autocast("cuda", enabled=False):
            patches = [_as_dhw(pn) for pn in (v_patch_nums or self.v_patch_nums)]
            assert patches[-1] == (D, H, W)

            residual = f_BLDHW.float()
            f_hat = torch.zeros_like(residual)
            out: List = []
            for si, (pd, ph, pw) in enumerate(patches):
                is_last = (si == len(patches) - 1)
                r = residual if is_last else F.interpolate(residual, size=(pd, ph, pw), mode=self.z_down)
                q, _idx, bit_idx, mag_idx, _aux = self.bsq(r)
                q = self._upsample_and_refine(q, si)
                f_hat = f_hat + q
                residual = residual - q
                if to_fhat:
                    out.append(f_hat.clone())
                else:
                    out.append((bit_idx.reshape(B, pd * ph * pw, L),
                                mag_idx.reshape(B, pd * ph * pw)))
        return out

    @torch.no_grad()
    def bits_to_fhat(self, ms_tokens: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """list[(bits (B, l_k, L), mag (B, l_k))] -> full-res quantized latent (B, L, D, H, W)."""
        B = ms_tokens[0][0].shape[0]
        D, H, W = self.v_patch_nums[-1]
        with torch.amp.autocast("cuda", enabled=False):
            f_hat = ms_tokens[0][0].new_zeros(B, self.L, D, H, W, dtype=torch.float32)
            for si, (bits, mag) in enumerate(ms_tokens):
                pd, ph, pw = self.v_patch_nums[si]
                code = self.bsq.indices_to_code(bits.view(B, pd, ph, pw, self.L),
                                                mag.view(B, pd, ph, pw))
                q = code.permute(0, 4, 1, 2, 3).contiguous()
                q = self._upsample_and_refine(q, si)
                f_hat = f_hat + q
        return f_hat

    @torch.no_grad()
    def fhat_no_vq(self, f_BLDHW: torch.Tensor) -> torch.Tensor:
        """No-VQ upper bound: keep the continuous residual (direction AND magnitude),
        i.e. skip both the sign and the radius quantization. Lives in f_hat space."""
        B, L, D, H, W = f_BLDHW.shape
        with torch.amp.autocast("cuda", enabled=False):
            residual = f_BLDHW.float()
            f_hat = torch.zeros_like(residual)
            for si, (pd, ph, pw) in enumerate(self.v_patch_nums):
                is_last = si == self.K - 1
                r = residual if is_last else F.interpolate(residual, size=(pd, ph, pw), mode=self.z_down)
                q = self._upsample_and_refine(r, si)   # continuous code = r (no normalize/sign/quant)
                f_hat = f_hat + q
                residual = residual - q
            return f_hat

    def extra_repr(self) -> str:
        mag_bits = max(0, (self.bsq.M - 1).bit_length())  # ceil(log2 M)
        return super().extra_repr() + f", polar=radius-codebook(M={self.bsq.M}, +{mag_bits} mag bits/token)"


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class BSQVAE3DPolar(BSQVAE3D):
    """BSQVAE3D with polar (magnitude-augmented) BSQ. Only __init__ changes; encode
    / decode / forward / encode_multiscale / decode_multiscale are inherited."""

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
        # polar-specific
        n_mag_levels: int = 4,
        mag_commit_weight: float = 0.25,
        mag_loss_weight: float = 1.0,
        mag_init_spread: float = 0.1,
    ):
        nn.Module.__init__(self)  # skip BSQVAE3D.__init__ (builds the plain quantizer)

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

        # pre/post-quant are unchanged from BSQVAE3D (L <-> latent_dim)
        self.pre_quant_conv = nn.Conv3d(latent_dim, codebook_bits, 1)
        self.post_quant_conv = nn.Conv3d(codebook_bits, latent_dim, 1)

        self.quantizer = MultiScaleBSQ3DPolar(
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
            n_mag_levels=n_mag_levels,
            mag_commit_weight=mag_commit_weight,
            mag_loss_weight=mag_loss_weight,
            mag_init_spread=mag_init_spread,
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
    model = BSQVAE3DPolar(
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
        n_mag_levels=4,
    ).to(device)

    print("Number of parameters, G", numel(model, only_trainable=True))
    print(model.quantizer.extra_repr())
    print(f"Radius levels at init: {model.quantizer.bsq.mag_levels.detach().cpu().numpy()}")

    model.train()
    x = torch.randn(1, 1, patch_size, patch_size, patch_size, device=device)
    x_hat, loss, codes, z_e, frac_unique = model(x)

    print(f"Input:            {tuple(x.shape)}")
    print(f"Output:           {tuple(x_hat.shape)}")
    print(f"VQ loss:          {loss.item():.4f}")
    print(f"Bit usage / scale: {[f'{u.item():.2f}' for u in frac_unique]}")

    # gradient sanity: recon+vq loss must reach the magnitude codebook
    ((x_hat - x).pow(2).mean() + loss).backward()
    g = model.quantizer.bsq.mag_levels.grad
    print(f"mag_levels grad present: {g is not None and g.abs().sum().item() > 0}")

    model.eval()
    # round-trip faithfulness: tokens (bits+mag) must rebuild the same f_hat as forward
    with torch.no_grad():
        tokens = model.encode_multiscale(x)
        fhat_rt = model.quantizer.bits_to_fhat(tokens)
        fhat_fwd, _, _ = model.quantizer(model.encode(x))
        print(f"Token round-trip preserves f_hat (incl. magnitude): "
              f"{torch.allclose(fhat_rt, fhat_fwd, atol=1e-4)}")

    print(f"Per-scale tokens: {[(tuple(b.shape), tuple(m.shape)) for b, m in tokens]} "
          f"(total tokens = {sum(b.shape[1] for b, _ in tokens)}, L={model.codebook_bits})")

    x_rec = model.decode_multiscale(tokens)
    print(f"Round-trip decode: {tuple(x_rec.shape)}")

    if torch.cuda.is_available():
        max_memory_reserved = torch.cuda.max_memory_reserved()
        print("Max memory reserved: %0.3f Gb / %0.3f Gb"
              % (max_memory_reserved / 1e9, total_gpu_mem))
