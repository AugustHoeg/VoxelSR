import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.utils.checkpoint as checkpoint
from utils.utils_3D_image import ICNR, numel

from models.FlashAttentionTest import STLayerV2, compute_mask
from models.models_3D import SRBlock3D

from collections import namedtuple

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv3d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv3d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv3d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv3d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv3d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, x):
        return self.lrelu(self.conv(x))


class DegradeNet(nn.Module):
    def __init__(self, down_factor=4, in_channels=1, out_channels=1, num_feats=64, use_checkpoint=True, requires_grad=True):
        super(DegradeNet, self).__init__()

        self.conv_first = nn.Conv3d(in_channels, num_feats, 3, 1, 1, bias=True)
        self.use_checkpoint = use_checkpoint

        self.conv_blocks = nn.ModuleList()

        if down_factor >= 2:
            self.conv_blocks.append(
                nn.Sequential(
                    RRDB(num_feats, gc=32),
                    RRDB(num_feats, gc=32),
                    ConvBlock(num_feats, num_feats, stride=2)
                )
            )
        else:
            self.conv_blocks.append(
                nn.Sequential(
                    RRDB(num_feats, gc=32),
                    RRDB(num_feats, gc=32),
                    ConvBlock(num_feats, num_feats, stride=1)
                )
            )

        self.conv_blocks.append(ConvBlock(num_feats, num_feats, stride=1))

        if down_factor >= 4:
            self.conv_blocks.append(
                nn.Sequential(
                    RRDB(num_feats, gc=32),
                    RRDB(num_feats, gc=32),
                    ConvBlock(num_feats, num_feats, stride=2)
                )
            )
        else:
            self.conv_blocks.append(
                nn.Sequential(
                    RRDB(num_feats, gc=32),
                    RRDB(num_feats, gc=32),
                    ConvBlock(num_feats, num_feats, stride=1)
                )
            )

        self.conv_last = nn.Conv3d(num_feats, out_channels, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.output_names = ['blk%d' % i for i in range(len(self.conv_blocks))] + ['output']

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x, ret_features=False):

        outs = []

        if self.use_checkpoint:
            z = checkpoint.checkpoint(self.conv_first, x)
        else:
            z = self.conv_first(x)

        for i, block in enumerate(self.conv_blocks):
            if self.use_checkpoint:
                z = checkpoint.checkpoint(block, z)
            else:
                z = block(z)

            if ret_features:
                outs.append(z)

        z = self.conv_last(self.lrelu(z))

        if ret_features:
            outs.append(z)  # append the final output
            outputs = namedtuple("Outputs", self.output_names)
            return outputs(*outs)
        else:
            return z


class PatchEmbed3D(nn.Module):
    """
    Very small 3D patch embedder: conv projection (proj) or flatten fallback.
    """

    def __init__(self, in_channels=1, dim=96, patch_size=4, method="proj", out_format="image"):
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim
        self.patch_size = patch_size
        self.method = method
        self.out_format = out_format

        if patch_size == 1:
            kernel_size = 3
            stride = 1
            pad = 1
        else:
            kernel_size = patch_size
            stride = patch_size
            pad = 1 if patch_size >= 3 else 0

        if self.method == "proj":
            self.proj = nn.Conv3d(in_channels, dim, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)
        else:
            # fallback: flatten (used rarely)
            self.proj = None

    def forward(self, x):
        # x: (B, C, H, W, D)
        if self.method == "proj":
            x = self.proj(x)  # (B, dim, Hp, Wp, Dp)
            if self.out_format == "tokens":
                x = x.flatten(2).transpose(1, 2)  # (B, N, C)
            elif self.out_format == "image":
                x = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, Hp, Wp, Dp, C)
            elif self.out_format == "same":
                pass
        else:
            x = x.flatten(2).transpose(1, 2)
        return x


class Group(nn.Module):

    def __init__(self, input_size, patch_size, dim, skip_dim, depth, window_size=8, num_heads=4, mlp_ratio=4, dp_rates=None):
        super().__init__()

        self.layers = nn.ModuleList()
        self.compress_layers = nn.ModuleList()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        for layer_idx in range(depth):
            channel_dim = dim + skip_dim * layer_idx
            compress_dim = skip_dim if layer_idx < depth - 1 else dim

            self.window_size = window_size
            self.shift_size = window_size // 2 if (layer_idx % 2) == 1 else 0
            self.num_heads = num_heads

            self.layers.append(
                    STLayerV2(level=0,
                            embed_dims=[channel_dim],
                            context_sizes=[input_size],
                            patch_sizes=[patch_size],
                            sizes_p=[(input_size//patch_size, input_size//patch_size, input_size//patch_size)],
                            num_heads=self.num_heads,
                            window_sizes=[self.window_size],
                            shift_size=self.shift_size,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=True,
                            drop=0.,
                            attn_drop=0.,
                            drop_path=dp_rates[layer_idx],
                            act_layer=nn.GELU,
                            norm_layer=nn.LayerNorm,
                            pretrained_window_size=0,
                            enable_ape_x=False,
                            patch_pe_method="window_relative",
                            return_x_image=False,
                            token_upsample_method="Monaipixelshuffle",
                    )
            )

            self.compress_layers.append(
                nn.Conv3d(channel_dim, compress_dim, kernel_size=1, stride=1, padding=0)
            )

            if layer_idx % 2 == 1:
                mlp_ratio = max(1, mlp_ratio // 2)

        # Dp, Hp, Wp = [input_size // patch_size] * 3  # hardcoded for now
        # self.mask_matrix = compute_mask((Dp, Hp, Wp), self.window_size, self.shift_size).cuda()  # hardcoded for now

    def forward(self, x):

        x_input = x
        next_x = x

        ###### Dense-connected block structure ######
        for i in range(len(self.layers) - 1):

            x = x.permute(0, 2, 3, 4, 1)  # B, C, D, H, W -> B, D, H, W, C

            # Main layer
            x = self.layers[i](x)
            x = x.permute(0, 4, 1, 2, 3).contiguous()  # B, C, D, H, W)
            x = self.act(self.compress_layers[i](x))

            # Concatenate
            next_x = torch.cat([next_x, x], 1)
            x = next_x

        # Final layer
        next_x = next_x.permute(0, 2, 3, 4, 1)
        x = self.layers[-1](next_x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.act(self.compress_layers[-1](x))

        x = 0.2 * x + x_input

        return x


class TransitionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, stride=2):
        super().__init__()
        #self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv = nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        return self.conv(x)


class PatchMerging3D(nn.Module):
    """
    Patch merging layer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm) -> None:
        """
        Args:
            dim: number of feature channels.
            norm_layer: normalization layer.
        """

        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)  # B, C, D, H, W -> B, D, H, W, C

        b, d, h, w, c = x.shape
        pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, 0::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = self.norm(x)
        x = self.reduction(x)

        x = x.permute(0, 4, 1, 2, 3).contiguous()  # B, D, H, W, C -> B, C, D, H, W
        return x


class FlashDegradeNet(nn.Module):
    def __init__(self,
                 input_size,
                 down_factor,
                 num_blks,
                 blk_layers,
                 in_chans,
                 shallow_feat,
                 embed_dim,
                 num_heads,
                 mlp_ratio,
                 attn_window_size,
                 patch_size,
                 skip_dims,
                 drop_path_rate=0.0,
                 use_checkpoint=False,
                 upsample_method="nearest",
                 requires_grad=True):
        super(FlashDegradeNet, self).__init__()

        self.use_checkpoint = use_checkpoint

        # Shallow feature extraction blocks (simple convs)
        self.sfe_blk = nn.Conv3d(in_chans, shallow_feat, kernel_size=3, stride=1, padding=1, bias=True)

        # Patch embedding blocks (proj only)
        self.patch_embedding = PatchEmbed3D(
            in_channels=shallow_feat,
            dim=embed_dim,
            patch_size=patch_size,
            method="proj",
            out_format="same"
        )
        #
        # # dropout (kept for API)
        # self.pos_drop = nn.Dropout(p=0.0)

        self.LX_blocks = nn.ModuleList()
        cur = 0
        dp_rates = [x for x in np.linspace(0, drop_path_rate, num_blks * blk_layers)]
        for i in range(num_blks):
            blocks = nn.Sequential()
            dp = dp_rates[cur: cur + blk_layers]
            blocks.append(
                Group(input_size=input_size,
                        patch_size=patch_size,
                        dim=embed_dim,
                        skip_dim=skip_dims[i],
                        depth=blk_layers,
                        window_size=attn_window_size,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        dp_rates=dp
                )
            )
            if i == 0 and down_factor >= 2:
                input_size = input_size // 2  # downsample by 2
                blocks.append(PatchMerging3D(embed_dim, norm_layer=nn.LayerNorm))
                #blocks.append(TransitionLayer(embed_dim, 2 * embed_dim, stride=2))
                embed_dim = 2 * embed_dim  # double feature channels
            elif i == 1 and down_factor >= 4:
                input_size = input_size // 2  # downsample by 2
                blocks.append(PatchMerging3D(embed_dim, norm_layer=nn.LayerNorm))
                #blocks.append(TransitionLayer(embed_dim, 2 * embed_dim, stride=2))
                embed_dim = 2 * embed_dim   # double feature channels
            elif i < num_blks - 1:
                blocks.append(TransitionLayer(embed_dim, embed_dim, stride=1))

            cur += blk_layers

            self.LX_blocks.append(blocks)

        self.Final_blk = nn.Sequential(
            nn.Identity()
        )
        if patch_size >= 2:
            self.Final_blk.append(
                SRBlock3D(embed_dim,
                          embed_dim,
                          k_size=6, pad=2,
                          upsample_method=upsample_method,
                          upscale_factor=patch_size,
                          use_checkpoint=False)
            )
        if patch_size >= 4:
            self.Final_blk.append(
                SRBlock3D(embed_dim,
                        embed_dim,
                        k_size=6, pad=2,
                        upsample_method=upsample_method,
                        upscale_factor=patch_size,
                        use_checkpoint=False)
            )

        self.conv_sfe = nn.Conv3d(in_channels=embed_dim,
                                    out_channels=shallow_feat,
                                    kernel_size=3, stride=1, padding=1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv_last = nn.Sequential(
            nn.Conv3d(in_channels=shallow_feat, out_channels=in_chans, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.output_names = ['embedding'] + ['blk%d' % i for i in range(len(self.LX_blocks))] + ['output']

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x, ret_features=False):
        """
        x: (B, C, H, W, D)
        returns: upsampled output (B, C, up_H, up_W, up_D)
        """

        outs = []

        # SFE
        if self.use_checkpoint:
            emb = checkpoint.checkpoint(self.sfe_blk, x)  # (B, shallow_feats[level], D, H, W)
        else:
            emb = self.sfe_blk(x)  # (B, shallow_feats[level], D, H, W)

        # Patch embedding
        if self.use_checkpoint:
            emb = checkpoint.checkpoint(self.patch_embedding, emb)
        else:
            emb = self.patch_embedding(emb)

        outs.append(emb)  # append features after patch embedding

        # Pass through blocks
        for i, blk in enumerate(self.LX_blocks):
            if self.use_checkpoint:
                emb = checkpoint.checkpoint(blk, emb)
            else:
                emb = blk(emb)

            if ret_features:
                outs.append(emb)  # append features after each block

        if self.use_checkpoint:
            final_feats = checkpoint.checkpoint(self.Final_blk, emb)
        else:
            final_feats = self.Final_blk(emb)

        # output
        out = self.conv_last(self.lrelu(self.conv_sfe(final_feats)))

        if ret_features:
            outs.append(out)  # append the final output
            outputs = namedtuple("Outputs", self.output_names)
            return outputs(*outs)
        else:
            return out



class FlashDegradeAE(nn.Module):
    def __init__(self,
                 input_size,
                 down_factor,
                 num_blks,
                 blk_layers,
                 in_chans,
                 shallow_feat,
                 embed_dim,
                 num_heads,
                 mlp_ratio,
                 attn_window_size,
                 patch_size,
                 skip_dims,
                 drop_path_rate=0.0,
                 use_checkpoint=False,
                 upsample_method="nearest",
                 requires_grad=True):
        super(FlashDegradeAE, self).__init__()

        self.use_checkpoint = use_checkpoint

        # Shallow feature extraction blocks (simple convs)
        self.sfe_blk = nn.Conv3d(in_chans, shallow_feat, kernel_size=3, stride=1, padding=1, bias=True)

        # Patch embedding blocks (proj only)
        self.patch_embedding = PatchEmbed3D(
            in_channels=shallow_feat,
            dim=embed_dim,
            patch_size=patch_size,
            method="proj",
            out_format="same"
        )
        #
        # # dropout (kept for API)
        # self.pos_drop = nn.Dropout(p=0.0)

        self.LX_blocks_down = nn.ModuleList()
        self.LX_blocks_up = nn.ModuleList()

        cur = 0
        dp_rates = [x for x in np.linspace(0, drop_path_rate, 2 * num_blks * blk_layers)]

        for i in range(num_blks):
            blocks_down = nn.Sequential()
            blocks_up = nn.Sequential()
            dp = dp_rates[cur: cur + blk_layers]
            blocks_down.append(
                Group(input_size=input_size,
                        patch_size=patch_size,
                        dim=embed_dim,
                        skip_dim=skip_dims[i],
                        depth=blk_layers,
                        window_size=attn_window_size,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        dp_rates=dp
                )
            )

            blocks_up.append(
                Group(input_size=input_size,
                      patch_size=patch_size,
                      dim=embed_dim,
                      skip_dim=skip_dims[i],
                      depth=blk_layers,
                      window_size=attn_window_size,
                      num_heads=num_heads,
                      mlp_ratio=mlp_ratio,
                      dp_rates=dp
                      )
            )
            from monai.networks.nets.swin_unetr import window_reverse
            if i == 0 and down_factor >= 2:
                input_size = input_size // 2  # downsample by 2
                blocks_down.append(PatchMerging3D(embed_dim, norm_layer=nn.LayerNorm))
                blocks_up.append(
                    SRBlock3D(embed_dim,
                              embed_dim,
                              k_size=6, pad=2,
                              upsample_method='Pixelshuffle3D',
                              upscale_factor=patch_size,
                              use_checkpoint=False)
                )
                embed_dim = 2 * embed_dim  # double feature channels
            elif i == 1 and down_factor >= 4:
                input_size = input_size // 2  # downsample by 2
                blocks_down.append(PatchMerging3D(embed_dim, norm_layer=nn.LayerNorm))
                blocks_up.append(
                    SRBlock3D(embed_dim,
                              embed_dim,
                              k_size=6, pad=2,
                              upsample_method='Pixelshuffle3D',
                              upscale_factor=patch_size,
                              use_checkpoint=False)
                )
                embed_dim = 2 * embed_dim   # double feature channels
            elif i < num_blks - 1:
                pass
                # blocks_down.append(TransitionLayer(embed_dim, embed_dim, stride=1))

            cur += blk_layers

            self.LX_blocks_down.append(blocks_down)
            self.LX_blocks_up.append(blocks_up)


        self.LR_blk = nn.Sequential(
            nn.Identity()
        )
        if patch_size >= 2:
            self.LR_blk.append(
                SRBlock3D(embed_dim,
                          embed_dim,
                          k_size=6, pad=2,
                          upsample_method=upsample_method,
                          upscale_factor=patch_size,
                          use_checkpoint=False)
            )
        if patch_size >= 4:
            self.LR_blk.append(
                SRBlock3D(embed_dim,
                        embed_dim,
                        k_size=6, pad=2,
                        upsample_method=upsample_method,
                        upscale_factor=patch_size,
                        use_checkpoint=False)
            )

        self.conv_sfe_lr = nn.Conv3d(in_channels=embed_dim,
                                    out_channels=shallow_feat,
                                    kernel_size=3, stride=1, padding=1, bias=True)

        self.conv_sfe_hr = nn.Conv3d(in_channels=embed_dim,
                                    out_channels=shallow_feat,
                                    kernel_size=3, stride=1, padding=1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv_last_lr = nn.Sequential(
            nn.Conv3d(in_channels=shallow_feat, out_channels=in_chans, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.conv_last_hr = nn.Sequential(
            nn.Conv3d(in_channels=shallow_feat, out_channels=in_chans, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.output_names = ['embedding'] + ['blk%d' % i for i in range(len(self.LX_blocks))] + ['output']

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x, ret_features=False):
        """
        x: (B, C, H, W, D)
        returns: upsampled output (B, C, up_H, up_W, up_D)
        """

        outs = []

        # SFE
        if self.use_checkpoint:
            emb = checkpoint.checkpoint(self.sfe_blk, x)  # (B, shallow_feats[level], D, H, W)
        else:
            emb = self.sfe_blk(x)  # (B, shallow_feats[level], D, H, W)

        # Patch embedding
        if self.use_checkpoint:
            emb = checkpoint.checkpoint(self.patch_embedding, emb)
        else:
            emb = self.patch_embedding(emb)

        outs.append(emb)  # append features after patch embedding

        # Encoder blocks
        for i, blk in enumerate(self.LX_blocks_down):
            if self.use_checkpoint:
                emb = checkpoint.checkpoint(blk, emb)
            else:
                emb = blk(emb)

            # Only append features for encoder blocks, not decoder blocks
            if ret_features:
                outs.append(emb)  # append features after each block

        # Predict LR output from the bottleneck features
        if self.use_checkpoint:
            lr_feats = checkpoint.checkpoint(self.LR_blk, emb)
        else:
            lr_feats = self.LR_blk(emb)

        lr_out = self.conv_last_lr(self.lrelu(self.conv_sfe_lr(lr_feats)))

        if ret_features:  # append LR output as well if ret_features is True
            outs.append(lr_out)  # append features after each block

        # Decoder blocks
        for i, blk in enumerate(self.LX_blocks_up):
            if self.use_checkpoint:
                emb = checkpoint.checkpoint(blk, emb)
            else:
                emb = blk(emb)

        # output
        hr_out = self.conv_last(self.lrelu(self.conv_sfe(emb)))

        if ret_features:
            outs.append(out)  # append the final output
            outputs = namedtuple("Outputs", self.output_names)
            return outputs(*outs)
        else:
            return out


def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / 10 ** 9 if torch.cuda.is_available() else 0

    batch_size = 1
    patch_size_hr = 128
    down_factor = 4

    # Create a random input: batch size 1, 3 channels, 224x224 image
    x_hr = torch.randn(batch_size, 1, patch_size_hr, patch_size_hr, patch_size_hr).to(device)

    x_lr = torch.randn((batch_size, 1, patch_size_hr // down_factor,
                        patch_size_hr // down_factor,
                        patch_size_hr // down_factor)).cuda()

    loss_func = nn.MSELoss()

    if False:
        print("Test DegradationModel")
        net = DegradeNet(down_factor=down_factor,
                               in_channels=1,
                               out_channels=1,
                               num_feats=32,
                               use_checkpoint=True).to(device)

        print("Number of parameters, G", numel(net, only_trainable=True))

        net.train()  # inference mode

        # Forward pass
        start = time.time()
        out = net(x_hr)
        stop = time.time()
        print("Time elapsed:", stop - start)

        loss = loss_func(out, x_lr)
        loss.backward()

        print("Output shape:", out.shape)

        max_memory_reserved = torch.cuda.max_memory_reserved()
        print("Maximum memory reserved: %0.3f Gb / %0.3f Gb" % (max_memory_reserved / 10 ** 9, total_gpu_mem))

    # Test FlashDegradeNet
    net = FlashDegradeNet(
        input_size=patch_size_hr,
        down_factor=down_factor,
        num_blks=6,
        blk_layers=3,
        in_chans=1,
        shallow_feat=32,
        embed_dim=96,
        num_heads=4,
        mlp_ratio=4,
        attn_window_size=8,
        patch_size=2,
        skip_dims=[32, 32, 32, 32, 32, 32],
        drop_path_rate=0.0,
        use_checkpoint=True,
        upsample_method="pixelshuffle3D",
        requires_grad=True,
    ).to(device)

    print("Number of parameters, G", numel(net, only_trainable=True))

    # Forward pass
    start = time.time()
    out = net(x_hr, ret_features=False)
    stop = time.time()
    print("Time elapsed:", stop - start)

    loss = loss_func(out, x_lr)
    loss.backward()

    print("Output shape:", out.shape)

    max_memory_reserved = torch.cuda.max_memory_reserved()
    print("Maximum memory reserved: %0.3f Gb / %0.3f Gb" % (max_memory_reserved / 10 ** 9, total_gpu_mem))


    print("Done")




if __name__ == "__main__":
    test()
