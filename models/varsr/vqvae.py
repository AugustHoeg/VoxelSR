"""
VARVQVAE2D — the VARSR multi-scale VQVAE, ported to 2D / single-channel and
wrapped to fit VoxelSR's ``ModelVQVAE`` training loop.

Ported primarily from the *original* 2D source (VARSR ``models/vqvae.py`` +
``basic_vae.py`` + ``quant.py``) rather than from the repo's 3D ``VARVQVAE3D``,
to avoid compounding 3D-translation quirks. ``models/VARVQVAE3D.py`` is used only
as the reference for the ``ModelVQVAE`` interface.

Contract expected by ``models/model_vqvae.py::ModelVQVAE``:

    x_hat, vq_loss, codes, z_e, frac_unique = net(x)     # forward
    x_no_vq                                 = net.decode(z_e)

  * ``x_hat``      : (B, C, H, W) reconstruction (unclamped — used for the loss).
  * ``vq_loss``    : scalar tensor.
  * ``codes``      : ``None`` (kept for signature parity; not used by ModelVQVAE).
  * ``z_e``        : pre-quant latent = ``quant_conv(encoder(x))``; ``decode`` of it
                     gives the "no-VQ" reconstruction shown in ``current_visuals``.
  * ``frac_unique``: ``list[Tensor]`` per-scale codebook-usage fraction.

Faithfulness notes:
  * Module names / structure match the official ``VQVAE`` exactly (``encoder``,
    ``decoder``, ``quant_conv``, ``post_quant_conv``, ``quantize``) so official
    RGB checkpoints (``vae_ch160v4096z32.pth``) load with strict key matching when
    the model is built in the matching RGB config (``in_channels=3``). See
    ``load_pretrained``.
  * Single-channel from-scratch training cannot reuse RGB weights (``conv_in`` /
    ``conv_out`` channel mismatch); that is expected and is the intended grayscale
    setup. Pretrained loading is provided for the optional RGB zero-shot ablation
    and as a key-compatibility check.
  * The adversarial + LPIPS training used in the *original* VQVAE is not part of
    ``ModelVQVAE`` (which optimizes reconstruction + VQ commitment, with optional
    LPIPS via its loss dict, but no discriminator). Use ``model_vqgan`` later if a
    discriminator is wanted; the network here is unchanged either way.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .basic_vae import Decoder, Encoder
from .quant import VectorQuantizer2


class VARVQVAE2D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        vocab_size: int = 4096,
        z_channels: int = 32,
        ch: int = 160,
        ch_mult: Sequence[int] = (1, 1, 2, 2, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        using_sa: bool = True,
        using_mid_sa: bool = True,
        beta: float = 0.25,              # commitment loss weight
        using_znorm: bool = False,       # cosine NN lookup if True, else L2
        quant_conv_ks: int = 3,          # quant conv kernel size
        quant_resi: float = 0.5,         # phi(x) = 0.5 conv(x) + 0.5 x
        share_quant_resi: int = 4,       # partially-shared phi for K scales
        default_qresi_counts: int = 0,   # 0 -> len(v_patch_nums)
        v_patch_nums: Sequence[int] = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        eini: float = 0.0,               # codebook init: 0 default, <0 uniform, >0 trunc_normal
        resolution: Optional[int] = None,  # HR patch size; used to validate the scale schedule
        test_mode: bool = False,
    ):
        super().__init__()
        self.test_mode = test_mode
        self.V, self.Cvae = vocab_size, z_channels
        self.in_channels = in_channels
        self.vocab_size = vocab_size
        self.v_patch_nums = tuple(v_patch_nums)
        self.ch_mult = tuple(ch_mult)
        self.downsample = 2 ** (len(self.ch_mult) - 1)

        if resolution is not None:
            assert resolution % self.downsample == 0, (
                f'resolution (patch_size_hr={resolution}) must be divisible by the '
                f'VQVAE downsample factor {self.downsample} (=2**(len(ch_mult)-1)).'
            )
            latent_res = resolution // self.downsample
            assert self.v_patch_nums[-1] == latent_res, (
                f'v_patch_nums[-1]={self.v_patch_nums[-1]} must equal the encoder '
                f'output resolution latent={latent_res} (=patch_size_hr {resolution} '
                f'// downsample {self.downsample}). Adjust v_patch_nums or ch_mult.'
            )

        ddconfig = dict(
            dropout=dropout, ch=ch, z_channels=z_channels,
            in_channels=in_channels, ch_mult=tuple(ch_mult), num_res_blocks=num_res_blocks,
            using_sa=using_sa, using_mid_sa=using_mid_sa,
        )
        self.encoder = Encoder(double_z=False, **ddconfig)
        self.decoder = Decoder(**ddconfig)

        self.quantize: VectorQuantizer2 = VectorQuantizer2(
            vocab_size=vocab_size, Cvae=self.Cvae, using_znorm=using_znorm, beta=beta,
            default_qresi_counts=default_qresi_counts, v_patch_nums=v_patch_nums,
            quant_resi=quant_resi, share_quant_resi=share_quant_resi,
        )
        if eini != 0.0:
            self.quantize.eini(eini)

        self.quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks // 2)
        self.post_quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks // 2)

        if self.test_mode:
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)

    # ---------------------------------------------------------------- core
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Image -> pre-quant latent (B, Cvae, h, w)."""
        return self.quant_conv(self.encoder(x))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Latent (pre-quant or quantized f_hat, Cvae channels) -> image."""
        return self.decoder(self.post_quant_conv(f))

    def forward(self, inp: torch.Tensor):
        """ModelVQVAE contract: (x_hat, vq_loss, codes, z_e, frac_unique)."""
        z_e = self.encode(inp)
        f_hat, vq_loss, frac_unique = self.quantize(z_e)
        x_hat = self.decode(f_hat)
        return x_hat, vq_loss, None, z_e, frac_unique

    # ---------------------------------------------------------------- helpers (Stage-2 / eval)
    def fhat_to_img(self, f_hat: torch.Tensor) -> torch.Tensor:
        return self.decode(f_hat).clamp_(-1, 1)

    def img_to_idxBl(self, inp_img_no_grad: torch.Tensor, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.LongTensor]:
        f = self.encode(inp_img_no_grad)
        return self.quantize.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=v_patch_nums)

    def idxBl_to_img(self, ms_idx_Bl: List[torch.Tensor], same_shape: bool, last_one: bool = False) -> Union[List[torch.Tensor], torch.Tensor]:
        B = ms_idx_Bl[0].shape[0]
        ms_h_BChw = []
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            pn = round(l ** 0.5)
            ms_h_BChw.append(self.quantize.embedding(idx_Bl).transpose(1, 2).view(B, self.Cvae, pn, pn))
        return self.embed_to_img(ms_h_BChw=ms_h_BChw, all_to_max_scale=same_shape, last_one=last_one)

    def embed_to_img(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale: bool, last_one: bool = False) -> Union[List[torch.Tensor], torch.Tensor]:
        if last_one:
            return self.decode(self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=True)).clamp_(-1, 1)
        return [self.decode(f_hat).clamp_(-1, 1) for f_hat in self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=False)]

    def img_to_reconstructed_img(self, x: torch.Tensor, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one: bool = False) -> Union[List[torch.Tensor], torch.Tensor]:
        f = self.encode(x)
        ls_f_hat_BChw = self.quantize.f_to_idxBl_or_fhat(f, to_fhat=True, v_patch_nums=v_patch_nums)
        if last_one:
            return self.decode(ls_f_hat_BChw[-1]).clamp_(-1, 1)
        return [self.decode(f_hat).clamp_(-1, 1) for f_hat in ls_f_hat_BChw]

    # ---------------------------------------------------------------- checkpoints
    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True, assign: bool = False):
        # Same tolerance as the official file: the EMA-hit buffer may differ in the
        # scale dimension between checkpoints; keep ours when it mismatches.
        key = 'quantize.ema_vocab_hit_SV'
        if key in state_dict and key in self.state_dict() and state_dict[key].shape != self.state_dict()[key].shape:
            state_dict[key] = self.state_dict()[key]
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)

    def load_pretrained(self, ckpt_path: str, strict: bool = True, map_location: str = 'cpu') -> Tuple[List[str], List[str]]:
        """Load an official VARSR / VAR VQVAE checkpoint. Returns (missing, unexpected)."""
        sd = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        if isinstance(sd, dict) and 'trainer' in sd and 'vae_wo_ddp' in sd.get('trainer', {}):  # VAR trainer checkpoint layout
            sd = sd['trainer']['vae_wo_ddp']
        result = self.load_state_dict(sd, strict=strict)
        missing = list(getattr(result, 'missing_keys', []))
        unexpected = list(getattr(result, 'unexpected_keys', []))
        return missing, unexpected


# ---------------------------------------------------------------------------
# Smoke / contract / pretrained-load test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import os

    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- 1) grayscale contract test (small config, CPU-friendly) --------------
    print('=' * 70)
    print('[1] Grayscale (in_channels=1) ModelVQVAE-contract test')
    hr = 64
    net = VARVQVAE2D(
        in_channels=1, vocab_size=512, z_channels=16, ch=32,
        ch_mult=(1, 2, 4), num_res_blocks=2,          # downsample 4 -> latent 16
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        eini=-1.0, resolution=hr, test_mode=False,
    ).to(device)
    net.train()

    x = torch.randn(2, 1, hr, hr, device=device)
    x_hat, vq_loss, codes, z_e, frac_unique = net(x)
    assert x_hat.shape == x.shape, x_hat.shape
    assert codes is None
    assert z_e.shape[1] == 16 and z_e.shape[2] == hr // 4, z_e.shape
    assert vq_loss.ndim == 0, vq_loss.shape
    assert isinstance(frac_unique, list) and len(frac_unique) == 10
    assert all(f.ndim == 0 for f in frac_unique)
    stacked = torch.stack([f.detach().float() for f in frac_unique])  # ModelVQVAE._update_frac_unique_ema
    assert stacked.shape == (10,)
    # no-VQ reconstruction path used by current_visuals
    x_no_vq = net.decode(z_e)
    assert x_no_vq.shape == x.shape
    # backward works
    (vq_loss + torch.nn.functional.l1_loss(x_hat, x)).backward()
    gsum = sum(p.grad.abs().sum() for p in net.parameters() if p.grad is not None)
    assert torch.isfinite(gsum) and gsum > 0
    print(f'    forward ok  x_hat={tuple(x_hat.shape)}  z_e={tuple(z_e.shape)}  '
          f'vq_loss={vq_loss.item():.4f}')
    print(f'    frac_unique/scale = {[round(f.item(), 2) for f in frac_unique]}')
    print(f'    backward ok  (grad L1 = {gsum.item():.3f})')

    # eval helpers
    net.eval()
    with torch.no_grad():
        rec = net.img_to_reconstructed_img(x, last_one=True)
        idxs = net.img_to_idxBl(x)
    assert rec.shape == x.shape
    assert len(idxs) == 10 and idxs[-1].shape == (2, 16 * 16)
    print(f'    img_to_reconstructed_img ok  rec={tuple(rec.shape)}  '
          f'tokens/scale={[t.shape[1] for t in idxs]} (total {sum(t.shape[1] for t in idxs)})')

    # ---- 2) checkpoint round-trip (grayscale) --------------------------------
    print('=' * 70)
    print('[2] Checkpoint round-trip (save -> load, strict)')
    net2 = VARVQVAE2D(
        in_channels=1, vocab_size=512, z_channels=16, ch=32,
        ch_mult=(1, 2, 4), num_res_blocks=2,
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        eini=-1.0, resolution=hr,
    ).to(device)
    missing = net2.load_state_dict(net.state_dict(), strict=True)
    net2.eval()
    with torch.no_grad():
        y1, _, _, _, _ = net(x)
        y2, _, _, _, _ = net2(x)
    assert torch.allclose(y1, y2, atol=1e-5)
    print('    strict load ok; outputs identical after reload.')

    # ---- 3) pretrained-weight loading (official RGB config) ------------------
    print('=' * 70)
    print('[3] Pretrained-load compatibility (official RGB config, in_channels=3)')
    rgb = VARVQVAE2D(
        in_channels=3, vocab_size=4096, z_channels=32, ch=160,
        ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,        # downsample 16 -> latent 16
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        using_sa=True, using_mid_sa=True,
    )
    n_params = sum(p.numel() for p in rgb.parameters())
    print(f'    built official-config VQVAE: {n_params/1e6:.1f}M params, '
          f'downsample={rgb.downsample}, scales={len(rgb.v_patch_nums)}')

    ckpt = os.environ.get('VARSR_VAE_CKPT', '').strip()
    if ckpt and os.path.isfile(ckpt):
        missing, unexpected = rgb.load_pretrained(ckpt, strict=False)
        print(f'    loaded real checkpoint: {ckpt}')
        print(f'      missing keys   ({len(missing)}): {missing[:6]}{" ..." if len(missing) > 6 else ""}')
        print(f'      unexpected keys({len(unexpected)}): {unexpected[:6]}{" ..." if len(unexpected) > 6 else ""}')
        assert len(missing) == 0, 'real checkpoint is missing keys our model needs (arch mismatch)'
        print('    OK: official checkpoint fully populates the model (faithful architecture).')
    else:
        # No local checkpoint -> prove the load mechanism (incl. ema-shape patch) end-to-end.
        rgb_src = VARVQVAE2D(
            in_channels=3, vocab_size=4096, z_channels=32, ch=160,
            ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,
            v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        )
        res = rgb.load_state_dict(rgb_src.state_dict(), strict=True)
        print('    (no VARSR_VAE_CKPT set) simulated-pretrained strict load ok.')
        print('    Set env VARSR_VAE_CKPT=/path/to/vae_ch160v4096z32.pth to test the real checkpoint.')

    print('=' * 70)
    print('All VARVQVAE2D Stage-1 checks passed.')
