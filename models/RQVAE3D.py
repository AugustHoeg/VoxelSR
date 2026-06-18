import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F

from models.VQGAN3D import Encoder, Decoder


class VQEmbedding(nn.Embedding):
    """
    VQ codebook with EMA updates and dead-code restart.

    Index n_embed is reserved as a padding/mask token (always-zero embedding),
    which is useful for MaskGIT-style transformers built on top of RQVAE.

    Args:
        n_embed:               number of active codebook entries
        embed_dim:             embedding dimension
        decay:                 EMA momentum for codebook update
        restart_unused_codes:  reassign dead entries to random batch vectors
        eps:                   Laplace smoothing epsilon for EMA normalisation
    """

    def __init__(self, n_embed, embed_dim, decay=0.99, restart_unused_codes=True, eps=1e-5):
        super().__init__(n_embed + 1, embed_dim, padding_idx=n_embed)
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.restart_unused_codes = restart_unused_codes

        for p in self.parameters():
            p.requires_grad_(False)

        self.register_buffer('cluster_size_ema', torch.zeros(n_embed))
        self.register_buffer('embed_ema', self.weight[:-1].detach().clone())

    @torch.no_grad()
    def compute_distances(self, inputs):
        # Cast to weight dtype to be safe under mixed precision
        codebook_t = self.weight[:-1].t()  # (embed_dim, n_embed)
        inputs_flat = inputs.reshape(-1, inputs.shape[-1]).to(codebook_t.dtype)
        inputs_sq = inputs_flat.pow(2).sum(1, keepdim=True)
        cb_sq = codebook_t.pow(2).sum(0, keepdim=True)
        dists = torch.addmm(inputs_sq + cb_sq, inputs_flat, codebook_t, alpha=-2.0)
        return dists.reshape(*inputs.shape[:-1], -1)  # (..., n_embed)

    @torch.no_grad()
    def find_nearest(self, inputs):
        return self.compute_distances(inputs).argmin(dim=-1)

    @torch.no_grad()
    def _tile_with_noise(self, x, target_n):
        n, d = x.shape
        n_rep = (target_n + n - 1) // n
        std = x.new_ones(d) * 0.01 / (d ** 0.5)
        x = x.repeat(n_rep, 1)[:target_n]
        return x + torch.rand_like(x) * std

    @torch.no_grad()
    def _update_buffers(self, vectors, idxs):
        v = vectors.reshape(-1, self.weight.shape[-1]).float()
        idx = idxs.reshape(-1)
        n_v = v.shape[0]

        one_hot = v.new_zeros(self.n_embed, n_v)
        one_hot.scatter_(0, idx.unsqueeze(0), 1.0)

        cluster_size = one_hot.sum(1)
        vectors_sum = one_hot @ v

        if dist.is_initialized():
            dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)
            dist.all_reduce(vectors_sum, op=dist.ReduceOp.SUM)

        self.cluster_size_ema.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.embed_ema.mul_(self.decay).add_(vectors_sum, alpha=1 - self.decay)

        if self.restart_unused_codes:
            if n_v < self.n_embed:
                v = self._tile_with_noise(v, self.n_embed)
            rand_v = v[torch.randperm(v.shape[0], device=v.device)][:self.n_embed]
            if dist.is_initialized():
                dist.broadcast(rand_v, 0)
            usage = (self.cluster_size_ema >= 1).float().unsqueeze(1)
            self.embed_ema.mul_(usage).add_(rand_v * (1 - usage))
            self.cluster_size_ema.mul_(usage.squeeze(1))
            self.cluster_size_ema.add_(1 - usage.squeeze(1))

    @torch.no_grad()
    def _update_embedding(self):
        n = self.cluster_size_ema.sum()
        smoothed = n * (self.cluster_size_ema + self.eps) / (n + self.n_embed * self.eps)
        self.weight[:-1] = self.embed_ema / smoothed.unsqueeze(1)

    def embed(self, idxs):
        """Look up embeddings by index without any update."""
        return super().forward(idxs)

    def forward(self, inputs):
        """
        inputs: (..., embed_dim)
        Returns:
            embeds: (..., embed_dim)  nearest codebook vectors
            idxs:   (...,)            LongTensor of codebook indices
        """
        idxs = self.find_nearest(inputs)
        if self.training:
            self._update_buffers(inputs, idxs)
        embeds = self.embed(idxs)
        if self.training:
            self._update_embedding()
        return embeds, idxs


class RQBottleneck3D(nn.Module):
    """
    Residual Quantization bottleneck for 3D volumetric latents.

    Applies n_rq_depth VQ codebooks sequentially. Each codebook quantizes the
    residual left by the previous stage. Each spatial position of the encoder
    output is treated as an independent latent_dim-dimensional vector.

    Commitment loss is computed at every intermediate cumulative reconstruction,
    which encourages all depth levels to contribute meaningfully.

    Args:
        latent_dim:            channel dimension of encoder output
        n_embed:               codebook size — int for uniform across depths,
                               list[int] for per-depth sizes
        n_rq_depth:            number of residual quantization steps (= code depth D)
        decay:                 EMA momentum — float or list[float] per depth
        shared_codebook:       if True, all depths share one codebook instance
        restart_unused_codes:  randomly reassign dead codebook entries during training
    """

    def __init__(
        self,
        latent_dim,
        n_embed,
        n_rq_depth,
        decay=0.99,
        shared_codebook=False,
        restart_unused_codes=True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_rq_depth = n_rq_depth
        self.shared_codebook = shared_codebook

        n_embed_list = [n_embed] * n_rq_depth if isinstance(n_embed, int) else list(n_embed)
        decay_list   = [decay]   * n_rq_depth if isinstance(decay, float)  else list(decay)
        assert len(n_embed_list) == len(decay_list) == n_rq_depth

        if shared_codebook:
            cb = VQEmbedding(
                n_embed_list[0], latent_dim,
                decay=decay_list[0],
                restart_unused_codes=restart_unused_codes,
            )
            self.codebooks = nn.ModuleList([cb] * n_rq_depth)
        else:
            self.codebooks = nn.ModuleList([
                VQEmbedding(
                    n_embed_list[i], latent_dim,
                    decay=decay_list[i],
                    restart_unused_codes=restart_unused_codes,
                )
                for i in range(n_rq_depth)
            ])

    def quantize(self, x):
        """
        x: (B, Dz, Dy, Dx, C)  [channel-last]

        Returns:
            quant_list: list of n_rq_depth cumulative quantized tensors (B, Dz, Dy, Dx, C)
                        quant_list[i] = sum of codebook outputs from depth 0..i
            codes:      (B, Dz, Dy, Dx, n_rq_depth) LongTensor
        """
        residual = x.detach().clone()
        aggregated = torch.zeros_like(x)
        quant_list, code_list = [], []

        for cb in self.codebooks:
            quant, code = cb(residual)
            residual = residual - quant
            aggregated = aggregated + quant
            quant_list.append(aggregated.clone())
            code_list.append(code.unsqueeze(-1))

        codes = torch.cat(code_list, dim=-1)
        return quant_list, codes

    def compute_commitment_loss(self, x, quant_list):
        """
        Average MSE between encoder output and each cumulative quantization.
        Averaging over depths encourages every level to carry useful information.
        """
        losses = [F.mse_loss(x, q.detach()) for q in quant_list]
        return torch.stack(losses).mean()

    def forward(self, x):
        """
        x: (B, C, Dz, Dy, Dx)

        Returns:
            z_q:             (B, C, Dz, Dy, Dx) straight-through quantized latent
            commitment_loss: scalar training loss for encoder alignment
            codes:           (B, Dz, Dy, Dx, n_rq_depth) LongTensor
        """
        x_cl = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, Dz, Dy, Dx, C)
        quant_list, codes = self.quantize(x_cl)
        commitment_loss = self.compute_commitment_loss(x_cl, quant_list)

        z_q_cl = quant_list[-1]
        z_q = z_q_cl.permute(0, 4, 1, 2, 3)           # (B, C, Dz, Dy, Dx)
        z_q = x + (z_q - x).detach()                   # straight-through estimator
        return z_q, commitment_loss, codes

    @torch.no_grad()
    def embed_codes(self, codes):
        """
        Reconstruct quantized latent by summing all depth embeddings.

        codes: (B, Dz, Dy, Dx, n_rq_depth)
        Returns: (B, C, Dz, Dy, Dx)
        """
        depth_slices = codes.unbind(dim=-1)
        embeds = sum(cb.embed(c) for cb, c in zip(self.codebooks, depth_slices))
        return embeds.permute(0, 4, 1, 2, 3)

    @torch.no_grad()
    def embed_codes_partial(self, codes, up_to_depth):
        """
        Reconstruct using only the first up_to_depth codebook levels.
        Enables progressive decoding and quality vs. compute trade-off.

        codes: (B, Dz, Dy, Dx, n_rq_depth)
        Returns: (B, C, Dz, Dy, Dx)
        """
        depth_slices = codes[..., :up_to_depth].unbind(dim=-1)
        embeds = sum(
            cb.embed(c) for cb, c in zip(self.codebooks[:up_to_depth], depth_slices)
        )
        return embeds.permute(0, 4, 1, 2, 3)

    @torch.no_grad()
    def embed_codes_by_depth(self, codes):
        """
        Return per-depth embeddings without summing, for use by the SR transformer.

        codes: (B, Dz, Dy, Dx, n_rq_depth)
        Returns: list of n_rq_depth tensors, each (B, C, Dz, Dy, Dx)
        """
        depth_slices = codes.unbind(dim=-1)
        return [
            cb.embed(c).permute(0, 4, 1, 2, 3)
            for cb, c in zip(self.codebooks, depth_slices)
        ]


class RQVAE3D(nn.Module):
    """
    3D Residual-Quantized VAE for volumetric super-resolution.

    Reuses the VQGAN3D encoder/decoder backbone. The VQ bottleneck is replaced
    by an RQ bottleneck that produces n_rq_depth stacked codes per spatial
    position, enabling progressive reconstruction and multi-depth SR.

    Args:
        in_channels:           image channels (1 for grayscale volumetric)
        latent_dim:            encoder output / codebook embedding dimension
        n_embed:               codebook size (int for uniform, list for per-depth)
        n_rq_depth:            number of residual quantization depths
        resolution:            isotropic spatial resolution of the input patch
        num_res_blocks:        residual blocks per encoder/decoder stage
        decay:                 EMA decay for codebook updates
        shared_codebook:       share one codebook instance across all RQ depths
        restart_unused_codes:  randomly reassign dead codebook entries
        use_checkpoint:        reserved for gradient checkpointing (not yet used)
    """

    def __init__(
        self,
        in_channels=1,
        latent_dim=256,
        n_embed=1024,
        n_rq_depth=4,
        resolution=64,
        num_res_blocks=2,
        decay=0.99,
        shared_codebook=False,
        restart_unused_codes=True,
        use_checkpoint=False,
    ):
        super().__init__()
        self.n_rq_depth = n_rq_depth
        self.use_checkpoint = use_checkpoint

        # Attention at the coarsest encoder/decoder resolution (16x downsampled)
        attn_res = max(resolution // 16, 1)

        self.encoder = Encoder(
            image_channels=in_channels,
            latent_dim=latent_dim,
            num_res_blocks=num_res_blocks,
            resolution=resolution,
            attn_resolutions=(attn_res,),
        )

        self.quantizer = RQBottleneck3D(
            latent_dim=latent_dim,
            n_embed=n_embed,
            n_rq_depth=n_rq_depth,
            decay=decay,
            shared_codebook=shared_codebook,
            restart_unused_codes=restart_unused_codes,
        )

        self.decoder = Decoder(
            image_channels=in_channels,
            latent_dim=latent_dim,
            num_res_blocks=num_res_blocks,
            resolution=attn_res,
            attn_resolutions=(attn_res,),
        )

    def encode(self, x):
        """
        x: (B, C, D, H, W)

        Returns:
            z_e:             continuous encoder output  (B, latent_dim, Dz, Dy, Dx)
            z_q:             straight-through quantized (B, latent_dim, Dz, Dy, Dx)
            commitment_loss: scalar
            codes:           (B, Dz, Dy, Dx, n_rq_depth) LongTensor
        """
        z_e = self.encoder(x)
        z_q, commitment_loss, codes = self.quantizer(z_e)
        return z_e, z_q, commitment_loss, codes

    def decode(self, z_q):
        return self.decoder(z_q)

    @torch.no_grad()
    def get_codes(self, x):
        """Encode volume to discrete codes only (no stored intermediate tensors)."""
        z_e = self.encoder(x)
        _, _, codes = self.quantizer(z_e)
        return codes

    @torch.no_grad()
    def decode_codes(self, codes):
        """Reconstruct volume from full-depth discrete codes."""
        z_q = self.quantizer.embed_codes(codes)
        return self.decoder(z_q)

    @torch.no_grad()
    def decode_codes_partial(self, codes, up_to_depth):
        """
        Progressive reconstruction using only the first up_to_depth codebook levels.
        Useful for visualising the contribution of each RQ depth during analysis.
        """
        z_q = self.quantizer.embed_codes_partial(codes, up_to_depth)
        return self.decoder(z_q)

    def forward(self, x):
        """
        Returns:
            x_hat:           reconstructed volume  (B, C, D, H, W)
            commitment_loss: RQ commitment loss (scalar, for training)
            codes:           (B, Dz, Dy, Dx, n_rq_depth) LongTensor
        """
        z_e, z_q, commitment_loss, codes = self.encode(x)
        x_hat = self.decode(z_q)
        return x_hat, commitment_loss, codes


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    patch_size = 64

    model = RQVAE3D(
        in_channels=1,
        latent_dim=256,
        n_embed=1024,
        n_rq_depth=4,
        resolution=patch_size,
    ).to(device)

    x = torch.randn(1, 1, patch_size, patch_size, patch_size, device=device)
    x_hat, loss, codes = model(x)

    print(f"Input:            {tuple(x.shape)}")
    print(f"Output:           {tuple(x_hat.shape)}")
    print(f"Codes:            {tuple(codes.shape)}  (spatial × n_rq_depth)")
    print(f"Commitment loss:  {loss.item():.4f}")

    # Progressive decoding — quality should improve with each depth
    for d in range(1, model.n_rq_depth + 1):
        recon = model.decode_codes_partial(codes, d)
        err = F.mse_loss(recon, x).item()
        print(f"  Partial decode depth={d}: MSE={err:.4f}")
