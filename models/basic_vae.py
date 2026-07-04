import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from models.VQVAE3D import GroupNorm, ResidualBlock, Swish, DownBlock
from models.VQGAN3D import NonLocalBlock
from models.models_3D import PixelUnshuffle3D
from utils.utils_3D_image import numel


class EncoderUnShuffle(nn.Module):
    def __init__(self, in_channels=1, patch_size=4, pre_conv_dim=1, embed_dims=[64, 128, 256, 512], use_checkpoint=False):
        super(Encoder, self).__init__()
        self.use_checkpoint = use_checkpoint

        self.pre_conv = nn.Conv3d(in_channels=in_channels, out_channels=pre_conv_dim, kernel_size=3, padding=1)
        self.tokenizer = PixelUnshuffle3D(patch_size)

        self.res_blocks = nn.ModuleList()
        self.res_blocks.append(ResidualBlock(in_channels=pre_conv_dim * patch_size**3, out_channels=embed_dims[0]))
        for i in range(len(embed_dims) - 1):
            self.res_blocks.append(ResidualBlock(in_channels=embed_dims[i], out_channels=embed_dims[i + 1]))


    def forward(self, x):

        x = self.pre_conv(x)
        x_tokens = self.tokenizer(x)

        for i, block in enumerate(self.res_blocks):
            if self.use_checkpoint:
                x_tokens = checkpoint.checkpoint(block, x_tokens)
            else:
                x_tokens = block(x_tokens)

        return x_tokens

class Encoder(nn.Module):
    def __init__(self, image_channels=1, latent_dim=256, num_res_blocks=2, resolution=256,
                 attn_resolutions=(16,), channels=[16, 64, 256, 1024], skip_attn=False, use_checkpoint=False):
        super(Encoder, self).__init__()
        self.use_checkpoint = use_checkpoint
        layers = [nn.Conv3d(image_channels, channels[0], 3, 1, 1)]
        for i in range(len(channels)-1):
            if i != len(channels)-2:
                layers.append(PixelUnshuffle3D(2))
                layers.append(nn.Conv3d(channels[i] * 8, channels[i + 1], 3, 1, 1))
                resolution //= 2
            for j in range(num_res_blocks):
                groups = channels[i+1] if channels[i+1] < 32 else 32
                layers.append(ResidualBlock(channels[i + 1], channels[i + 1], num_groups=groups))
                if resolution in attn_resolutions and not skip_attn:
                    layers.append(NonLocalBlock(channels[i + 1]))
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




if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / 10**9 if torch.cuda.is_available() else 0

    patch_size = 128
    x = torch.randn(1, 1, patch_size, patch_size, patch_size).to(device)  # Example input

    net = Encoder(use_checkpoint=True).to(device)

    print("Number of parameters, G", numel(net, only_trainable=True))

    out = net(x)

    max_memory_reserved = torch.cuda.max_memory_reserved()
    print("Maximum memory reserved: %0.3f Gb / %0.3f Gb" % (max_memory_reserved / 10**9, total_gpu_mem))
