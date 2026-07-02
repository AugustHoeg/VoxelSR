import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from models.VQVAE3D import CodeBook, DownBlock, GroupNorm, ResidualBlock, Swish, UpBlock


class NonLocalBlock(nn.Module):
    def __init__(self, channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = channels

        self.gn = GroupNorm(channels)
        self.q = nn.Conv3d(channels, channels, 1, 1, 0)
        self.k = nn.Conv3d(channels, channels, 1, 1, 0)
        self.v = nn.Conv3d(channels, channels, 1, 1, 0)
        self.proj_out = nn.Conv3d(channels, channels, 1, 1, 0)

    def forward(self, x):
        h_ = self.gn(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, d, h, w = q.shape

        q = q.reshape(b, c, d*h*w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, d*h*w)
        v = v.reshape(b, c, d*h*w)

        attn = torch.bmm(q, k)
        attn = attn * (int(c)**(-0.5))
        attn = F.softmax(attn, dim=2)
        attn = attn.permute(0, 2, 1)

        A = torch.bmm(v, attn)
        A = A.reshape(b, c, d, h, w)
        A = self.proj_out(A)

        return x + A


def channel_schedule(base_channels, channel_mult):
    """Per-stage channel widths, e.g. base_channels=64, channel_mult=(1,1,2,4) -> [64,64,128,256]."""
    return [base_channels * m for m in channel_mult]


class Encoder(nn.Module):
    def __init__(self, image_channels=1, latent_dim=256, num_res_blocks=2, resolution=256,
                 attn_resolutions=(16,), channels=[64, 64, 128, 256], skip_attn=False, use_checkpoint=False):
        super(Encoder, self).__init__()
        self.use_checkpoint = use_checkpoint
        layers = [nn.Conv3d(image_channels, channels[0], 3, 1, 1)]
        for i in range(len(channels)-1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions and not skip_attn:
                    layers.append(NonLocalBlock(in_channels))
            if i != len(channels)-2:
                layers.append(DownBlock(channels[i+1], channels[i+1]))
                resolution //= 2
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        if not skip_attn:
            layers.append(NonLocalBlock(channels[-1]))
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(GroupNorm(channels[-1]))
        layers.append(Swish())
        layers.append(nn.Conv3d(channels[-1], latent_dim, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_checkpoint:
            out = checkpoint.checkpoint_sequential(self.model, len(self.model), x, use_reentrant=False)
        else:
            out = self.model(x)
        return out
        

class Decoder(nn.Module):
    def __init__(self, image_channels=1, latent_dim=256, num_res_blocks=2, resolution=16,
                 attn_resolutions=(16,), channels=[256, 128, 64, 64], skip_attn=False, use_checkpoint=False):
        super(Decoder, self).__init__()
        self.use_checkpoint = use_checkpoint
        layers = [nn.Conv3d(latent_dim, channels[0], 3, 1, 1)]
        layers.append(ResidualBlock(channels[0], channels[0]))
        if not skip_attn:
            layers.append(NonLocalBlock(channels[0]))
        layers.append(ResidualBlock(channels[0], channels[0]))

        for i in range(len(channels)-1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions and not skip_attn:
                    layers.append(NonLocalBlock(in_channels))
            if i != 0:
                layers.append(UpBlock(channels[i+1], channels[i+1]))
                resolution *= 2

        layers.append(GroupNorm(channels[-1]))
        layers.append(Swish())
        layers.append(nn.Conv3d(channels[-1], image_channels, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_checkpoint:
            out = checkpoint.checkpoint_sequential(self.model, len(self.model), x, use_reentrant=False)
        else:
            out = self.model(x)
        return out


class VQModel3D(nn.Module):
    def __init__(self, in_channels,
                 latent_dim=256,
                 channels=[64, 64, 128, 256],
                 num_embeddings=1024,
                 resolution=64,
                 skip_attn=False,
                 use_checkpoint=False):
        super(VQModel3D, self).__init__()

        self.encoder = Encoder(
            image_channels=in_channels,
            latent_dim=latent_dim,
            channels=channels,
            resolution=resolution,
            attn_resolutions=[resolution // 4],
            skip_attn=skip_attn,
            use_checkpoint=use_checkpoint
        )

        self.decoder = Decoder(
            image_channels=in_channels,
            latent_dim=latent_dim,
            channels=channels[::-1],
            resolution=resolution // 4,  # Assuming 4x downsampling in encoder
            attn_resolutions=[resolution // 4],
            skip_attn=skip_attn,
            use_checkpoint=use_checkpoint,
        )

        self.codebook = CodeBook(num_embeddings=num_embeddings, embedding_dim=latent_dim)
        self.use_checkpoint = use_checkpoint

    def encode(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, q_indices, num_codes = self.codebook(z_e)
        return z_e, z_q, vq_loss, q_indices

    def decode(self, z_q):
        return self.decoder(z_q)

    def compute_loss(self, x):
        z_e, z_q, vq_loss, _ = self.encode(x)

        # Reconstruction loss
        x_hat = self.decoder(z_q)
        recon_loss = F.mse_loss(x_hat, x)

        return recon_loss + vq_loss

    def forward(self, x):

        z_e = self.encoder(x)
        z_q, vq_loss, q_indices, num_codes = self.codebook(z_e)
        x_hat = self.decoder(z_q)

        return x_hat, vq_loss, q_indices, z_e


class PatchGAN3D(nn.Module):
    """3D PatchGAN discriminator matching the NLayerDiscriminator used in VQGAN.
    Groups are set to 1 (InstanceNorm-style) to stay stable at small 3D batch sizes."""
    def __init__(self, in_channels=1, ndf=64, n_layers=3):
        super().__init__()
        kw, padw = 4, 1
        sequence = [
            nn.Conv3d(in_channels, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(ndf * 2 ** n, 512)
            sequence += [
                nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.GroupNorm(num_groups=min(32, nf), num_channels=nf),
                nn.LeakyReLU(0.2, True),
            ]
        nf_prev = nf
        nf = min(ndf * 2 ** n_layers, 512)
        sequence += [
            nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.GroupNorm(num_groups=min(32, nf), num_channels=nf),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(nf, 1, kernel_size=kw, stride=1, padding=padw),
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / 10**9 if torch.cuda.is_available() else 0

    patch_size = 128
    x = torch.randn(1, 1, patch_size, patch_size, patch_size).to(device)  # Example input

    channels = [64, 64, 128, 256]
    model = VQModel3D(in_channels=1,
                      latent_dim=256,
                      channels=channels,
                      resolution=patch_size,
                      skip_attn=True,
                      use_checkpoint=True).to(device)

    z_e, z_q, vq_loss, q_indices = model.encode(x)
    x_hat = model.decode(z_q)

    print("Encoded shape:", z_e.shape)
    print("Quantized shape:", z_q.shape)
    print("Reconstructed shape:", x_hat.shape)

    max_memory_reserved = torch.cuda.max_memory_reserved()
    print("Maximum memory reserved: %0.3f Gb / %0.3f Gb" % (max_memory_reserved / 10**9, total_gpu_mem))
