"""OSEDiff baseline (2D) -- vendored under models/osediff/, mirroring
models/resshift/ and models/varsr/.

Faithful port of the one-step diffusion SR networks from
https://github.com/cswry/OSEDiff. OSEDiff LoRA-finetunes a pretrained
Stable Diffusion VAE-encoder + UNet (decoder frozen) and distills the SD prior
into a *single* denoising step via Variational Score Distillation (VSD).

Kept (this module): the three core networks -- ``OSEDiff_gen`` (generator),
``OSEDiff_reg`` (VSD fake-/real-score regularizer) and ``OSEDiff_test``
(inference) -- plus the ``initialize_vae`` / ``initialize_unet`` LoRA injectors.

Dropped from upstream (non-essential / replaced by VoxelSR):
  * ``models/autoencoder_kl.py`` & ``models/unet_2d_condition.py`` -- stock
    diffusers copies; we import ``AutoencoderKL`` / ``UNet2DConditionModel``
    straight from diffusers (see ``osediff.py`` docstring).
  * ``ram`` / DAPE prompt extraction and ``open-clip`` / ``fairscale`` -- the
    volumetric domain has no natural-image captions, so the ModelBase wrapper
    conditions on a fixed null prompt instead.
  * ``basicsr`` real-world degradations, ``train_osediff*.py`` / ``test_*.py``
    (replaced by VoxelSR's ModelBase), and the ``accelerate`` trainer glue.

Grayscale->RGB channel handling and null-prompt conditioning live in the
ModelBase wrapper ``models/model_osediff.py`` (kept out of the vendored code so
this module stays faithful to the 3-channel, prompt-driven upstream).
"""

from .osediff import OSEDiff_gen, OSEDiff_reg, OSEDiff_test, initialize_vae, initialize_unet

__all__ = ['OSEDiff_gen', 'OSEDiff_reg', 'OSEDiff_test', 'initialize_vae', 'initialize_unet']
