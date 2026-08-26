"""Vendored ResShift components (https://github.com/zsyOAOA/ResShift).

Only the pieces needed for a *pixel-space* (no-autoencoder) 2D SR baseline are
kept:

* ``unet.UNetModelSwin`` -- the Swin-UNet denoiser (parametrized for arbitrary
  input/lq channel counts so single-channel data works).
* ``gaussian_diffusion`` / ``respace`` / ``script_util`` -- the residual-shifting
  Markov-chain diffusion. With ``first_stage_model=None`` the "encode/decode
  first stage" calls collapse to a bicubic upsample of the LR image, i.e. the
  diffusion runs directly in pixel space.

Dropped from upstream: the ``ldm`` autoencoder, ``basicsr``, ``datapipe``
degradations, ``trainer.py`` / ``main.py`` (replaced by VoxelSR's ModelBase),
``resample.py`` importance sampling, and the plain-DDPM sampler paths.

See ``models/model_resshift.py`` for the ModelBase wrapper that drives these.
"""

from .unet import UNetModelSwin
from .script_util import create_gaussian_diffusion


def build_resshift_diffusion(diffusion_opt):
    """Build the residual-shifting diffusion in pixel space from a config dict.

    Pixel-space specialization of ``create_gaussian_diffusion``:
      * ``latent_flag=False``   -> input scaling tuned for image-range tensors
      * ``scale_factor=None``   -> no latent-code rescaling
    The autoencoder is never attached (callers pass ``first_stage_model=None``),
    so ``encode_first_stage`` simply bicubically upsamples the LR by ``sf``.
    """
    o = dict(diffusion_opt)
    schedule_kwargs = o.pop('schedule_kwargs', None)
    if schedule_kwargs is None and 'power' in o:
        schedule_kwargs = {'power': o.pop('power')}
    return create_gaussian_diffusion(
        normalize_input=o.get('normalize_input', True),
        schedule_name=o.get('schedule_name', 'exponential'),
        sf=o['sf'],
        min_noise_level=o.get('min_noise_level', 0.04),
        steps=o.get('steps', 15),
        kappa=o.get('kappa', 2.0),
        etas_end=o.get('etas_end', 0.99),
        schedule_kwargs=schedule_kwargs,
        weighted_mse=o.get('weighted_mse', False),
        predict_type=o.get('predict_type', 'xstart'),
        timestep_respacing=o.get('timestep_respacing', None),
        scale_factor=None,
        latent_flag=False,
    )


__all__ = ['UNetModelSwin', 'create_gaussian_diffusion', 'build_resshift_diffusion']
