import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

        if in_channels != out_channels:
            self.channel_up = nn.Conv3d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.channel_up(x) + self.block(x)
        else:
            return x + self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, 6, 2, 2)

    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels=256):
        super(Encoder, self).__init__()

        self.down1 = DownBlock(in_channels, hidden_channels)
        self.down2 = DownBlock(hidden_channels, hidden_channels)

        self.res1 = ResidualBlock(hidden_channels, hidden_channels)
        self.res2 = ResidualBlock(hidden_channels, hidden_channels)

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)

        x = self.res1(x)
        x = self.res2(x)

        return x

class Decoder(nn.Module):
    def __init__(self, hidden_channels=256, out_channels=1):
        super(Decoder, self).__init__()

        self.res1 = ResidualBlock(hidden_channels, hidden_channels)
        self.res2 = ResidualBlock(hidden_channels, hidden_channels)

        self.up1 = UpBlock(hidden_channels, hidden_channels)
        self.up2 = UpBlock(hidden_channels, out_channels)

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)

        x = self.up1(x)
        x = self.up2(x)

        return x

class CodeBook(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super(CodeBook, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.beta = beta

    def forward(self, z_e):
        B, C, D, H, W = z_e.shape
        z_e_flat = z_e.view(B, C, -1).permute(0, 2, 1)  # (B, DHW, C)

        # Compute distances to codebook embeddings
        dists = torch.cdist(z_e_flat, self.embedding.weight)  # (B, DHW, num_embeddings)
        q_indices = torch.argmin(dists, dim=-1)  # (B, DHW)

        z_q = self.embedding(q_indices)  # (B, DHW, embedding_dim)
        z_q = z_q.permute(0, 2, 1).view(B, self.embedding_dim, D, H, W)  # (B, embedding_dim, H, W)

        # Compute vq + commitment loss
        vq_loss = F.mse_loss(z_e.detach(), z_q) + self.beta * F.mse_loss(z_e, z_q.detach())

        # Copy the gradients of z_e to z_q, and use z_q values in forward pass
        z_q = z_e + (z_q - z_e).detach()  # Straight-through estimator

        return z_q, vq_loss

class VQVAE3D(nn.Module):
    def __init__(self, in_channels, hidden_channels=256, num_embeddings=512, use_checkpoint=False):
        super(VQVAE3D, self).__init__()

        self.encoder = Encoder(in_channels, hidden_channels)
        self.decoder = Decoder(hidden_channels, out_channels=in_channels)
        self.codebook = CodeBook(num_embeddings=num_embeddings, embedding_dim=hidden_channels)
        self.use_checkpoint = use_checkpoint

    def encode(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss = self.codebook(z_e)

        return z_e, z_q, vq_loss

    def decode(self, z_q):
        return self.decoder(z_q)

    def compute_loss(self, x):
        z_e, z_q, vq_loss = self.encode(x)

        # Reconstruction loss
        x_hat = self.decoder(z_q)
        recon_loss = F.mse_loss(x_hat, x)

        return recon_loss + vq_loss

    def forward(self, x):

        z_e = self.encoder(x)
        z_q, vq_loss = self.codebook(z_e)
        x_hat = self.decoder(z_q)

        return x_hat, vq_loss


if __name__ == '__main__':
    model = VQVAE3D(in_channels=1, hidden_channels=256)
    x = torch.randn(2, 1, 32, 32, 32)  # Example input
    z_e, z_q, vq_loss = model.encode(x)
    x_hat = model.decode(z_q)

    print("Encoded shape:", z_e.shape)
    print("Quantized shape:", z_q.shape)
    print("Reconstructed shape:", x_hat.shape)

