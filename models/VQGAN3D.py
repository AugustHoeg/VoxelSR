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

        return x + A


class Encoder(nn.Module):
    def __init__(self, image_channels=1, latent_dim=256, num_res_blocks=2, resolution=256, attn_resolutions=(16,), use_checkpoint=False):
        super(Encoder, self).__init__()
        self.use_checkpoint = use_checkpoint
        channels = [128, 128, 128, 256, 256, 512]
        layers = [nn.Conv3d(image_channels, channels[0], 3, 1, 1)]
        for i in range(len(channels)-1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            if i != len(channels)-2:
                layers.append(DownBlock(channels[i+1], channels[i+1]))
                resolution //= 2
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(NonLocalBlock(channels[-1]))
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(GroupNorm(channels[-1]))
        layers.append(Swish())
        layers.append(nn.Conv3d(channels[-1], latent_dim, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_checkpoint:
            out = checkpoint.checkpoint_sequential(self.model, len(self.model), x)
        else:
            out = self.model(x)
        return out
        

class Decoder(nn.Module):
    def __init__(self, image_channels=1, latent_dim=256, num_res_blocks=2, resolution=16, attn_resolutions=(16,), use_checkpoint=False):
        super(Decoder, self).__init__()
        self.use_checkpoint = use_checkpoint
        channels = [512, 256, 256, 128, 128, 128]
        layers = [nn.Conv3d(latent_dim, channels[0], 3, 1, 1)]
        layers.append(ResidualBlock(channels[0], channels[0]))
        layers.append(NonLocalBlock(channels[0]))
        layers.append(ResidualBlock(channels[0], channels[0]))

        for i in range(len(channels)-1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
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
            out = checkpoint.checkpoint_sequential(self.model, len(self.model), x)
        else:
            out = self.model(x)
        return out


class VQModel3D(nn.Module):
    def __init__(self, in_channels,
                 latent_dim=256,
                 num_embeddings=1024,
                 resolution=32,
                 use_checkpoint=False):
        super(VQModel3D, self).__init__()

        self.encoder = Encoder(
            image_channels=in_channels,
            latent_dim=latent_dim,
            resolution=resolution,
            attn_resolutions=[resolution // 16],
            use_checkpoint=use_checkpoint
        )

        self.decoder = Decoder(
            image_channels=in_channels,
            latent_dim=latent_dim,
            resolution=resolution//16,  # Assuming 16x downsampling in encoder
            attn_resolutions=[resolution // 16],
            use_checkpoint=use_checkpoint
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

        return x_hat, vq_loss


class PatchGAN3D(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(128, 1, 3, 1, 1)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / 10**9 if torch.cuda.is_available() else 0

    patch_size = 128
    x = torch.randn(1, 1, patch_size, patch_size, patch_size).to(device)  # Example input

    model = VQModel3D(in_channels=1, latent_dim=256, resolution=patch_size, use_checkpoint=True).to(device)

    z_e, z_q, vq_loss, q_indices = model.encode(x)
    x_hat = model.decode(z_q)

    print("Encoded shape:", z_e.shape)
    print("Quantized shape:", z_q.shape)
    print("Reconstructed shape:", x_hat.shape)

    max_memory_reserved = torch.cuda.max_memory_reserved()
    print("Maximum memory reserved: %0.3f Gb / %0.3f Gb" % (max_memory_reserved / 10**9, total_gpu_mem))
