"""VARSR baseline (2D) — vendored under models/varsr/, mirroring models/resshift/.

Stage 1 (this module): the multi-scale VQVAE tokenizer, ported to 2D /
single-channel from the official VARSR source and wrapped to fit VoxelSR's
``ModelVQVAE`` training loop. See ``vqvae.VARVQVAE2D``.
"""

from .vqvae import VARVQVAE2D

__all__ = ['VARVQVAE2D']
