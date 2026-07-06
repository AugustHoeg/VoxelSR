import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from models.VQVAE3D import GroupNorm, ResidualBlock, Swish, DownBlock
from models.VQGAN3D import NonLocalBlock
from models.models_3D import PixelUnshuffle3D, SRBlock3D
from utils.utils_3D_image import numel


class EncoderUnShuffle(nn.Module):
    def __init__(self, in_channels=1, patch_size=4, pre_conv_dim=1, embed_dims=[64, 128, 256, 512], use_checkpoint=False):
        super(EncoderUnShuffle, self).__init__()
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
    def __init__(
        self,
        image_channels=1,
        latent_dim=768,
        num_res_blocks=2,
        resolution=128,
        attn_resolutions=(16,),
        channels=[32, 64, 256, 512, 512],
        skip_attn=False,
        use_checkpoint=False,
    ):
        super(Encoder, self).__init__()
        self.use_checkpoint = use_checkpoint
        layers = [nn.Conv3d(image_channels, channels[0], 3, 1, 1)]
        for i in range(len(channels) - 1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            groups = 32 if in_channels % 32 == 0 else in_channels
            layers.append(ResidualBlock(in_channels, out_channels, num_groups=groups))
            
            ### Downsampling
            if i != len(channels) - 2:
                layers.append(PixelUnshuffle3D(2))
                layers.append(nn.Conv3d(out_channels * 8, out_channels, 3, 1, 1))
                groups = 32 if out_channels % 32 == 0 else out_channels
                layers.append(ResidualBlock(out_channels, out_channels, num_groups=groups))
                resolution //= 2
            else:
                layers.append(nn.Conv3d(in_channels, out_channels, 3, 1, 1))
            ###
            
            for j in range(num_res_blocks - 1):
                groups = 32 if out_channels % 32 == 0 else out_channels
                layers.append(ResidualBlock(out_channels, out_channels, num_groups=groups))
                if resolution in attn_resolutions and not skip_attn:
                    layers.append(NonLocalBlock(out_channels))

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
    def __init__(
        self,
        image_channels=1,
        latent_dim=768,
        num_res_blocks=2,
        resolution=16,
        attn_resolutions=(16,),
        channels=[768, 768, 384, 96, 24],
        skip_attn=False,
        use_checkpoint=False,
    ):
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
            for j in range(num_res_blocks - 1):
                groups = 32 if in_channels % 32 == 0 else in_channels
                layers.append(ResidualBlock(in_channels, in_channels, num_groups=groups))
                if resolution in attn_resolutions and not skip_attn:
                    layers.append(NonLocalBlock(in_channels))

            ### Upsampling
            if i != 0:
                layers.append(
                    SRBlock3D(
                        in_c=in_channels,
                        n=in_channels,
                        upsample_method="pixelshuffle3D",
                        upscale_factor=2,
                        skip_act=True,
                    )
                )
                groups = 32 if out_channels % 32 == 0 else out_channels
                layers.append(ResidualBlock(in_channels, out_channels, num_groups=groups))
                resolution *= 2
            else:
                layers.append(nn.Conv3d(in_channels, out_channels, 3, 1, 1))
            ###

            groups = 32 if out_channels % 32 == 0 else out_channels
            layers.append(ResidualBlock(out_channels, out_channels, num_groups=groups))

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


class EncoderV2(nn.Module):
    def __init__(
        self,
        image_channels=1,
        latent_dim=512,
        num_res_blocks=2,
        resolution=128,
        attn_resolutions=(16,),
        channels=[64, 64, 256, 512, 512],
        skip_attn=False,
        use_checkpoint=False,
    ):
        super(EncoderV2, self).__init__()
        self.use_checkpoint = use_checkpoint
        layers = [nn.Conv3d(image_channels, channels[0], 3, 1, 1)]
        for i in range(len(channels) - 1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            groups = 32 if in_channels % 32 == 0 else in_channels
            layers.append(ResidualBlock(in_channels, out_channels, num_groups=groups))

            for j in range(num_res_blocks - 1):
                groups = 32 if out_channels % 32 == 0 else out_channels
                layers.append(ResidualBlock(out_channels, out_channels, num_groups=groups))
                if resolution in attn_resolutions and not skip_attn:
                    layers.append(NonLocalBlock(out_channels))

            ### Downsampling
            if i != len(channels) - 2:
                layers.append(PixelUnshuffle3D(2))
                layers.append(nn.Conv3d(out_channels * 8, out_channels, 3, 1, 1))
                groups = 32 if out_channels % 32 == 0 else out_channels
                # layers.append(ResidualBlock(out_channels, out_channels, num_groups=groups))
                resolution //= 2
            else:
                layers.append(nn.Conv3d(in_channels, out_channels, 3, 1, 1))
            ###

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


class DecoderV2(nn.Module):
    def __init__(
        self,
        image_channels=1,
        latent_dim=512,
        num_res_blocks=2,
        resolution=16,
        attn_resolutions=(16,),
        channels=[512, 512, 256, 64, 64],
        skip_attn=False,
        use_checkpoint=False,
    ):
        super(DecoderV2, self).__init__()
        self.use_checkpoint = use_checkpoint
        layers = [nn.Conv3d(latent_dim, channels[0], 3, 1, 1)]
        layers.append(ResidualBlock(channels[0], channels[0]))
        if not skip_attn:
            layers.append(NonLocalBlock(channels[0]))
        layers.append(ResidualBlock(channels[0], channels[0]))

        for i in range(len(channels) - 1):
            in_channels = channels[i]
            out_channels = channels[i + 1]

            ### Upsampling
            if i != 0:
                layers.append(
                    SRBlock3D(
                        in_c=in_channels,
                        n=in_channels,
                        upsample_method="pixelshuffle3D",
                        upscale_factor=2,
                        skip_act=True,
                    )
                )
                groups = 32 if out_channels % 32 == 0 else out_channels
                # layers.append(ResidualBlock(in_channels, in_channels, num_groups=groups))
                resolution *= 2
            else:
                layers.append(nn.Conv3d(in_channels, in_channels, 3, 1, 1))
            ###

            for j in range(num_res_blocks - 1):
                if resolution in attn_resolutions and not skip_attn:
                    layers.append(NonLocalBlock(in_channels))
                groups = 32 if in_channels % 32 == 0 else in_channels
                layers.append(ResidualBlock(in_channels, in_channels, num_groups=groups))

            groups = 32 if out_channels % 32 == 0 else out_channels
            layers.append(ResidualBlock(in_channels, out_channels, num_groups=groups))

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


class DecoderClaude(nn.Module):
    """
    Symmetric counterpart to Encoder. Reverses every encoder stage using
    pixelshuffle-based upsampling with ICNR init (via SRBlock3D) instead of
    ConvTranspose3d, to avoid checkerboard artifacts.

    Args mirror Encoder. The caller passes:
        channels    : the ENCODER channel list reversed, e.g. [768, 768, 384, 96, 24]
        resolution  : the BOTTLENECK resolution (i.e. encoder input res / 2**down_factor)
    """
    def __init__(
        self,
        image_channels=1,
        latent_dim=768,
        num_res_blocks=2,
        resolution=16,
        attn_resolutions=(16,),
        channels=[768, 768, 384, 96, 24],
        skip_attn=False,
        use_checkpoint=False,
    ):
        super(Decoder, self).__init__()
        self.use_checkpoint = use_checkpoint

        # Mirror of encoder's final Conv3d(channels[-1] -> latent_dim)
        layers = [nn.Conv3d(latent_dim, channels[0], 3, 1, 1)]

        # Mirror of encoder's post-loop mid-blocks (Res + Attn + Res)
        groups_head = 32 if channels[0] % 32 == 0 else channels[0]
        layers.append(ResidualBlock(channels[0], channels[0], num_groups=groups_head))
        if not skip_attn:
            layers.append(NonLocalBlock(channels[0]))
        layers.append(ResidualBlock(channels[0], channels[0], num_groups=groups_head))

        for i in range(len(channels) - 1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            groups_in = 32 if in_channels % 32 == 0 else in_channels
            groups_out = 32 if out_channels % 32 == 0 else out_channels

            # Attention at pre-upsample resolution (mirrors encoder's post-downsample attn)
            if resolution in attn_resolutions and not skip_attn:
                layers.append(NonLocalBlock(in_channels))

            # Mirror of encoder's `for j in range(num_res_blocks - 1)` residuals
            for j in range(num_res_blocks - 1):
                layers.append(ResidualBlock(in_channels, in_channels, num_groups=groups_in))

            # Mirror of encoder's Res(out, out) that sits right after unshuffle+compress
            layers.append(ResidualBlock(in_channels, in_channels, num_groups=groups_in))

            if i != 0:
                # Upsampling stage: mirror of encoder's PixelUnshuffle + Conv(in*8 -> out).
                # SRBlock3D in 'pixelshuffle3D' mode does:
                #   Conv3d(in -> out, 3x3)  ->  Conv3d(out -> out*8, 3x3, ICNR)  ->  PixelShuffle3D(2)
                layers.append(SRBlock3D(
                    in_c=in_channels, n=out_channels,
                    upsample_method='pixelshuffle3D',
                    upscale_factor=2,
                    skip_act=True,
                ))
                resolution *= 2
            else:
                # First iteration: mirror of encoder's channel-widening-only last stage.
                layers.append(nn.Conv3d(in_channels, out_channels, 3, 1, 1))

            # Mirror of encoder's Res(in, in) at pre-downsample width
            layers.append(ResidualBlock(out_channels, out_channels, num_groups=groups_out))

        # Mirror of encoder's initial Conv3d(image_channels -> channels[0])
        groups_final = 32 if channels[-1] % 32 == 0 else channels[-1]
        layers.append(GroupNorm(channels[-1], num_groups=groups_final))
        layers.append(Swish())
        layers.append(nn.Conv3d(channels[-1], image_channels, 3, 1, 1))
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

    patch_size = 64
    latent_dim = 512
    channels = [64, 64, 256, 512, 512]
    num_res_blocks = 2
    attn_resolutions = (16,)

    x = torch.randn(1, 1, patch_size, patch_size, patch_size).to(device)

    encoder = EncoderV2(
        image_channels=1,
        latent_dim=latent_dim,
        num_res_blocks=num_res_blocks,
        resolution=patch_size,
        attn_resolutions=attn_resolutions,
        channels=channels,
        skip_attn=False,
        use_checkpoint=True,
    ).to(device)

    down_factor = 2 ** (len(channels) - 2)
    decoder = DecoderV2(
        image_channels=1,
        latent_dim=latent_dim,
        num_res_blocks=num_res_blocks,
        resolution=patch_size // down_factor,
        attn_resolutions=attn_resolutions,
        channels=channels[::-1],
        skip_attn=False,
        use_checkpoint=True,
    ).to(device)

    # encoder = encoder.to(memory_format=torch.channels_last_3d)
    # decoder = decoder.to(memory_format=torch.channels_last_3d)

    print("Encoder params:", numel(encoder, only_trainable=True))
    print("Decoder params:", numel(decoder, only_trainable=True))

    z = encoder(x)
    x_hat = decoder(z)

    print(f"Input:   {x.shape}")
    print(f"Latent:  {z.shape}")
    print(f"Output:  {x_hat.shape}")
    assert x_hat.shape == x.shape, "roundtrip shape mismatch"

    max_memory_reserved = torch.cuda.max_memory_reserved()
    print("Maximum memory reserved: %0.3f Gb / %0.3f Gb" % (max_memory_reserved / 10**9, total_gpu_mem))
