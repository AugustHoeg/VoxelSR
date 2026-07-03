import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from contextlib import contextmanager

from torch.nn import functional as F

from models.VQGAN3D import Encoder, Decoder


class VQEmbedding(nn.Embedding):
    """VQ embedding module with ema update."""

    def __init__(self, n_embed, embed_dim, ema=True, decay=0.99, restart_unused_codes=True, eps=1e-5):
        super().__init__(n_embed + 1, embed_dim, padding_idx=n_embed)

        self.ema = ema
        self.decay = decay
        self.eps = eps
        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed

        if self.ema:
            _ = [p.requires_grad_(False) for p in self.parameters()]

            # padding index is not updated by EMA
            self.register_buffer('cluster_size_ema', torch.zeros(n_embed))
            self.register_buffer('embed_ema', self.weight[:-1, :].detach().clone())

    @torch.no_grad()
    def compute_distances(self, inputs):
        codebook_t = self.weight[:-1, :].t()

        (embed_dim, _) = codebook_t.shape
        inputs_shape = inputs.shape
        assert inputs_shape[-1] == embed_dim

        inputs_flat = inputs.reshape(-1, embed_dim)
        inputs_flat = inputs_flat.to(codebook_t.dtype)  # // August

        inputs_norm_sq = inputs_flat.pow(2.).sum(dim=1, keepdim=True)
        codebook_t_norm_sq = codebook_t.pow(2.).sum(dim=0, keepdim=True)
        distances = torch.addmm(
            inputs_norm_sq + codebook_t_norm_sq,
            inputs_flat,
            codebook_t,
            alpha=-2.0,
        )
        distances = distances.reshape(*inputs_shape[:-1], -1)  # [B, d, h, w, n_embed or n_embed+1]
        return distances


    @torch.no_grad()
    def find_nearest_embedding(self, inputs):
        distances = self.compute_distances(inputs)  # [B, d, h, w, n_embed or n_embed+1]
        embed_idxs = distances.argmin(dim=-1)  # use padding index or not

        return embed_idxs, distances


    @torch.no_grad()
    def _tile_with_noise(self, x, target_n):
        B, embed_dim = x.shape
        n_repeats = (target_n + B -1) // B
        std = x.new_ones(embed_dim) * 0.01 / np.sqrt(embed_dim)
        x = x.repeat(n_repeats, 1)
        x = x + torch.rand_like(x) * std
        return x


    @torch.no_grad()
    def _update_buffers(self, vectors, idxs):
        n_embed, embed_dim = self.weight.shape[0] - 1, self.weight.shape[-1]

        vectors = vectors.reshape(-1, embed_dim)
        idxs = idxs.reshape(-1)

        n_vectors = vectors.shape[0]
        n_total_embed = n_embed

        one_hot_idxs = vectors.new_zeros(n_total_embed, n_vectors)
        one_hot_idxs.scatter_(dim=0, index=idxs.unsqueeze(0), src=vectors.new_ones(1, n_vectors))

        cluster_size = one_hot_idxs.sum(dim=1)
        vectors_sum_per_cluster = one_hot_idxs @ vectors

        if dist.is_initialized():
            dist.all_reduce(vectors_sum_per_cluster, op=dist.ReduceOp.SUM)
            dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)

        self.cluster_size_ema.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.embed_ema.mul_(self.decay).add_(vectors_sum_per_cluster, alpha=1 - self.decay)

        if self.restart_unused_codes:
            if n_vectors < n_embed:
                vectors = self._tile_with_noise(vectors, n_embed)
            n_vectors = vectors.shape[0]
            _vectors_random = vectors[torch.randperm(n_vectors, device=vectors.device)][:n_embed]

            if dist.is_initialized():
                dist.broadcast(_vectors_random, 0)

            usage = (self.cluster_size_ema.view(-1, 1) >= 1).float()
            self.embed_ema.mul_(usage).add_(_vectors_random * (1 - usage))
            self.cluster_size_ema.mul_(usage.view(-1))
            self.cluster_size_ema.add_(torch.ones_like(self.cluster_size_ema) * (1 - usage).view(-1))


    @torch.no_grad()
    def _update_embedding(self):
        n_embed = self.weight.shape[0] - 1
        n = self.cluster_size_ema.sum()
        normalized_cluster_size = n * (self.cluster_size_ema + self.eps) / (n + n_embed * self.eps)
        self.weight[:-1, :] = self.embed_ema / normalized_cluster_size.reshape(-1, 1)


    def forward(self, inputs):
        embed_idxs, dists = self.find_nearest_embedding(inputs)
        if self.training:
            if self.ema:
                self._update_buffers(inputs, embed_idxs)

        embeds = self.embed(embed_idxs)

        if self.ema and self.training:
            self._update_embedding()

        return embeds, embed_idxs, dists


    def embed(self, idxs):
        embeds = super().forward(idxs)
        return embeds


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
        decay_list = [decay] * n_rq_depth if isinstance(decay, float)  else list(decay)
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

        for codebook in self.codebooks:
            quant, code, dists = codebook(residual)
            residual.sub_(quant)  #
            aggregated.add_(quant)  #
            quant_list.append(aggregated.clone())
            code_list.append(code.unsqueeze(-1))

        codes = torch.cat(code_list, dim=-1)
        return quant_list, codes


    def compute_commitment_loss(self, x, quant_list):

        loss_list = []

        for idx, quant in enumerate(quant_list):
            partial_loss = F.mse_loss(x, quant.detach())
            loss_list.append(partial_loss)

        commitment_loss = torch.mean(torch.stack(loss_list))
        return commitment_loss


    def forward(self, x):
        """
        x: (B, C, Dz, Dy, Dx)

        Returns:
            z_q:             (B, C, Dz, Dy, Dx) straight-through quantized latent
            commitment_loss: scalar training loss for encoder alignment
            codes:           (B, Dz, Dy, Dx, n_rq_depth) LongTensor
        """
        x_code = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, Dz, Dy, Dx, C)
        quant_list, codes = self.quantize(x_code)
        commitment_loss = self.compute_commitment_loss(x_code, quant_list)

        z_q_last = quant_list[-1]  # Aggregated quantized codes
        z_q = z_q_last.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, Dz, Dy, Dx)
        z_q = x + (z_q - x).detach()  # straight-through estimator
        return z_q, commitment_loss, codes


    @torch.no_grad()
    def embed_code(self, code):
        assert code.shape[1:] == self.code_shape

        code_slices = torch.chunk(code, chunks=code.shape[-1], dim=-1)

        if self.shared_codebook:
            embeds = [self.codebooks[0].embed(code_slice) for i, code_slice in enumerate(code_slices)]
        else:
            embeds = [self.codebooks[i].embed(code_slice) for i, code_slice in enumerate(code_slices)]

        embeds = torch.cat(embeds, dim=-2).sum(-2)
        embeds = embeds.permute(0, 4, 1, 2, 3).contiguous()

        return embeds


    @torch.no_grad()
    def embed_partial_code(self, code, code_idx, decode_type="select"):
        r"""
        Decode the input codes, using [0, 1, ..., code_idx] codebooks.

        Arguments:
            code (Tensor): codes of input image
            code_idx (int): the index of the last selected codebook for decoding

        Returns:
            embeds (Tensor): quantized feature map
        """

        B, d, h, w, n_rq_depth = code.shape

        code_slices = torch.chunk(code, chunks=code.shape[-1], dim=-1)
        if self.shared_codebook:
            embeds = [self.codebooks[0].embed(code_slice) for i, code_slice in enumerate(code_slices)]
        else:
            embeds = [self.codebooks[i].embed(code_slice) for i, code_slice in enumerate(code_slices)]

        if decode_type == "select":
            embeds = embeds[code_idx].view(B, d, h, w, -1)
        elif decode_type == "add":
            embeds = torch.cat(embeds[: code_idx + 1], dim=-2).sum(-2)
        else:
            raise NotImplementedError(f"{decode_type} is not implemented in partial decoding")

        embeds = embeds.permute(0, 4, 1, 2, 3).contiguous()

        return embeds


    @torch.no_grad()
    def embed_code_with_depth(self, code):
        """
        Return per-depth embeddings without summing

        Caution: RQ-VAE does not use scale of codebook, thus assume all scales are ones.
        """

        code_slices = torch.chunk(code, chunks=code.shape[-1], dim=-1)

        if self.shared_codebook:
            embeds = [self.codebooks[0].embed(code_slice).squeeze(-2) for i, code_slice in enumerate(code_slices)]
        else:
            embeds = [self.codebooks[i].embed(code_slice).squeeze(-2) for i, code_slice in enumerate(code_slices)]

        return embeds


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
        quant_embed_dim=256,  #
        n_embed=1024,
        n_rq_depth=4,
        resolution=64,
        num_res_blocks=2,
        channels=[256, 128, 64, 64],
        decay=0.99,
        shared_codebook=False,
        restart_unused_codes=True,
        skip_attn=False,
        use_checkpoint=False,
    ):
        super().__init__()
        self.n_rq_depth = n_rq_depth
        self.use_checkpoint = use_checkpoint

        self.encoder = Encoder(
            image_channels=in_channels,
            latent_dim=latent_dim,
            num_res_blocks=num_res_blocks,
            resolution=resolution,
            attn_resolutions=[resolution // 4],
            channels=channels,
            skip_attn=skip_attn,
            use_checkpoint=use_checkpoint,
        )

        self.quantizer = RQBottleneck3D(
            latent_dim=quant_embed_dim,
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
            resolution=resolution // 4,
            attn_resolutions=[resolution // 4],
            channels=channels[::-1],
            skip_attn=skip_attn,
            use_checkpoint=use_checkpoint,
        )

        # Project encoder output (latent_dim) into quantizer space (quant_embed_dim),
        # then back — matching the z_channels → embed_dim convention of the reference.
        self.pre_quant_conv = nn.Conv3d(latent_dim, quant_embed_dim, 1)
        self.post_quant_conv = nn.Conv3d(quant_embed_dim, latent_dim, 1)
        

    def encode(self, x):
        """
        x: (B, C, D, H, W)
        """
        z_e = self.encoder(x)
        z_e = self.pre_quant_conv(z_e)
        return z_e


    def decode(self, z_q):
        z_q = self.post_quant_conv(z_q)
        z_q = self.decoder(z_q)
        return z_q


    @torch.no_grad()
    def get_code(self, x):
        """Encode volume to discrete codes only (no stored intermediate tensors)."""
        z_e = self.encode(x)
        _, _, code = self.quantizer(z_e)
        return code


    @torch.no_grad()
    def decode_code(self, code):
        """Reconstruct volume from full-depth discrete codes."""
        z_q = self.quantizer.embed_code(code)
        return self.decode(z_q)


    @torch.no_grad()
    def decode_code_partial(self, code, code_idx):
        """
        Progressive reconstruction using only the first code_idx codebook levels.
        Useful for visualising the contribution of each RQ depth during analysis.
        """
        z_q = self.quantizer.embed_partial_code(code, code_idx)
        return self.decode(z_q)

    def forward(self, x):
        """
        Returns:
            x_hat:           reconstructed volume  (B, C, D, H, W)
            commitment_loss: RQ commitment loss (scalar, for training)
            codes:           (B, Dz, Dy, Dx, n_rq_depth) LongTensor
        """

        z_e = self.encode(x)
        z_q, commitment_loss, codes = self.quantizer(z_e)
        x_hat = self.decode(z_q)
        return x_hat, commitment_loss, codes, z_e


class DualRQVAE3D(RQVAE3D):
    """
    RQVAE3D with a second encoder E* for domain adaptation.

    E, Q, D are trained on downsampled LR only. E* maps out-of-distribution LR
    into the same codebook space via reconstruction through frozen D
    (target = downsampled LR) and optional pre-quant distillation.
    """

    def __init__(self, in_channels=1, latent_dim=256, latent_dim_star=256,
                 quant_embed_dim=256, n_embed=1024, n_rq_depth=4, resolution=64, num_res_blocks=2,
                 num_res_blocks_star=2, channels=[64, 64, 128 ,256],
                 channels_star=[64, 64, 128, 256], decay=0.99, shared_codebook=False,
                 restart_unused_codes=True, skip_attn=False, use_checkpoint=False):
        super().__init__(
            in_channels=in_channels, latent_dim=latent_dim,
            quant_embed_dim=quant_embed_dim, n_embed=n_embed,
            n_rq_depth=n_rq_depth, resolution=resolution,
            num_res_blocks=num_res_blocks,
            channels=channels,
            decay=decay,
            shared_codebook=shared_codebook,
            restart_unused_codes=restart_unused_codes,
            skip_attn=skip_attn,
            use_checkpoint=use_checkpoint,
        )

        self.encoder_star = Encoder(
            image_channels=in_channels,
            latent_dim=latent_dim_star,
            num_res_blocks=num_res_blocks_star,
            resolution=resolution,
            attn_resolutions=[resolution // 4],
            channels=channels_star,
            skip_attn=skip_attn,
            use_checkpoint=use_checkpoint,
        )
        self.pre_quant_conv_star = nn.Conv3d(latent_dim_star, quant_embed_dim, 1)

    @contextmanager
    def frozen_encoder(self):
        self.encoder.requires_grad_(False)
        self.pre_quant_conv.requires_grad_(False)
        try:
            yield
        finally:
            self.encoder.requires_grad_(True)
            self.pre_quant_conv.requires_grad_(True)

    @contextmanager
    def frozen_decoder(self):
        """Freeze the shared decoder during star-path training.

        Gradient still flows *through* the frozen decoder back to E*, but the
        decoder weights themselves do not accumulate gradients — and DDP
        excludes them from all-reduce for that backward pass.
        """
        self.decoder.requires_grad_(False)
        self.post_quant_conv.requires_grad_(False)
        try:
            yield
        finally:
            self.decoder.requires_grad_(True)
            self.post_quant_conv.requires_grad_(True)

    def split_optimizer_params(self):
        """Partition parameters into main (E/Q/D) and star (E*) groups."""
        star_prefixes = ('encoder_star.', 'pre_quant_conv_star.')
        main_params, star_params = [], []
        for name, p in self.named_parameters():
            if any(name.startswith(pfx) for pfx in star_prefixes):
                if p.requires_grad:
                    star_params.append(p)
            else:
                if p.requires_grad:
                    main_params.append(p)
        return main_params, star_params

    def encode_star(self, x):
        z_e = self.encoder_star(x)
        z_e = self.pre_quant_conv_star(z_e)
        return z_e

    def forward(self, x, star_mode=False):
        """
        :param x:           input volume  (B, C, D, H, W)
        :param star_mode:   use E* encoder; disables codebook EMA and skips decoder grad accumulation
        :return:            x_hat:           reconstructed volume  (B, C, D, H, W)
                            commitment_loss: RQ commitment loss (scalar, for training)
                            codes:           (B, Dz, Dy, Dx, n_rq_depth) LongTensor
                            z_e:             pre-quantization latent (B, C, Dz, Dy, Dx)
        """
        if star_mode:
            z_e = self.encode_star(x)
            was_training = self.quantizer.training
            self.quantizer.eval()
            z_q, commitment_loss, codes = self.quantizer(z_e)
            if was_training:
                self.quantizer.train()
            return self.decode(z_q), commitment_loss, codes, z_e
        # main path
        z_e = self.encode(x)
        z_q, commitment_loss, codes = self.quantizer(z_e)
        return self.decode(z_q), commitment_loss, codes, z_e


class LatentMLPD3D(nn.Module):
    def __init__(self, in_channels, ndf=256, n_layers=6, use_spectral_norm=True):
        super().__init__()
        from torch.nn.utils.parametrizations import spectral_norm
        sn = spectral_norm if use_spectral_norm else (lambda m: m)

        layers, prev = [], in_channels
        for _ in range(n_layers):
            layers += [sn(nn.Conv3d(prev, ndf, 3, padding=1, bias=True)),
                       nn.LeakyReLU(0.2, inplace=True)]
            prev = ndf
        self.trunk = nn.Sequential(*layers)
        self.head  = sn(nn.Conv3d(ndf, 1, 1))  # per-position scalar map

    def forward(self, x):
        # (B, C, d, h, w) -> (B, d, h, w)
        return self.head(self.trunk(x)).squeeze(1)


if __name__ == '__main__':
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / 10**9 if torch.cuda.is_available() else 0

    patch_size = 32

    model = RQVAE3D(
        in_channels=1,
        latent_dim=256,
        channels=[64, 128, 256],
        quant_embed_dim=256,
        n_embed=1024,
        n_rq_depth=4,
        resolution=patch_size,
        num_res_blocks=4,
        skip_attn=True,
        use_checkpoint=True,
    ).to(device)

    model.train()

    x = torch.randn(1, 1, patch_size, patch_size, patch_size, device=device)
    x_hat, loss, codes, z_e = model(x)

    print(f"Input: {x.shape}")
    print(f"Output: {x_hat.shape}")
    print(f"Codes: {codes.shape} (spatial * n_rq_depth)")
    print(f"Commitment loss:  {loss.item():.4f}")

    max_memory_reserved = torch.cuda.max_memory_reserved()
    print("Maximum memory reserved: %0.3f Gb / %0.3f Gb" % (max_memory_reserved / 10**9, total_gpu_mem))

    print("Done")
