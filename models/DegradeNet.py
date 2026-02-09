import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.utils.checkpoint as checkpoint
from utils.utils_3D_image import ICNR, numel

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


class DegradeNetV2(nn.Module):
    def __init__(self, down_factor=4, in_channels=1, out_channels=1, num_feats=64, use_checkpoint=True, requires_grad=True):
        super(DegradeNetV2, self).__init__()

        raise NotImplementedError("DegradeNetV2 is not implemented yet. Please use DegradeNet instead.")

        # minimal retained attributes
        self.num_blks = num_blks
        self.blk_layers = blk_layers
        self.up_factor = up_factor
        self.upsample_method = upsample_method
        self.input_size = input_size
        self.num_levels = num_levels
        self.context_sizes = context_sizes
        self.patch_sizes = patch_sizes
        self.shallow_feats = shallow_feats
        self.pre_up_feats = pre_up_feats
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attn_window_sizes = attn_window_sizes

        # Shallow feature extraction blocks (simple convs)
        self.sfe_blks = nn.ModuleList()
        for level in range(num_levels):
            input_feats = in_chans if level == 0 else shallow_feats[level - 1]
            shallow_feat = shallow_feats[level]
            if level == num_levels - 1:
                sfe_blk = nn.Sequential(
                    nn.Conv3d(input_feats, shallow_feat, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv3d(shallow_feat, shallow_feat, kernel_size=3, stride=1, padding=1, bias=True),
                )
            else:
                sfe_blk = nn.Sequential(
                    nn.Conv3d(input_feats, shallow_feat, kernel_size=3, stride=1, padding=1, bias=True)
                )
            self.sfe_blks.append(sfe_blk)

        # Patch embedding blocks (proj only)
        self.patch_embedding_blks = nn.ModuleList()
        for level in range(num_levels):
            self.patch_embedding_blks.append(
                PatchEmbed3D(in_channels=shallow_feats[level],
                             dim=self.embed_dims[level],
                             patch_size=patch_sizes[level],
                             method="proj",
                             out_format="same")
            )
        #
        # # dropout (kept for API)
        # self.pos_drop = nn.Dropout(p=0.0)

        self.LX_blocks = nn.ModuleList()
        cur = 0
        for level in range(num_levels):
            blocks = nn.ModuleList()
            dp_rates = [x for x in np.linspace(0, drop_path_rate, self.num_blks[level] * self.blk_layers[level])]
            dp = dp_rates[cur: cur + self.blk_layers[level]]
            for _ in range(self.num_blks[level]):
                blocks.append(
                    STGroup(input_size=context_sizes[level],
                            patch_size=patch_sizes[level],
                            dim=self.embed_dims[level],
                            skip_dim=skip_dims[level],
                            depth=self.blk_layers[level],
                            window_size=self.attn_window_sizes[level],
                            num_heads=self.num_heads,
                            mlp_ratio=self.mlp_ratio,
                            dp_rates=dp,
                            prev_dim=self.embed_dims[level - 1] if level > 0 else None,
                            prev_patch_size=patch_sizes[level - 1] if level > 0 else None)
                )
            self.LX_blocks.append(blocks)

        self.Final_blk = nn.Sequential(
            nn.Identity()
        )
        if patch_sizes[-1] > 1:
            self.Final_blk.append(
                SRBlock3D(self.embed_dims[-1],
                          self.embed_dims[-1],
                          k_size=6, pad=2,
                          upsample_method="nearest",
                          upscale_factor=patch_sizes[-1],
                          use_checkpoint=False)
            )

        # convolution to map final features (image space)
        self.conv_image = nn.Conv3d(in_channels=self.embed_dims[-1],
                                    out_channels=shallow_feats[-1],
                                    kernel_size=3, stride=1, padding=1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Upsampling SR blocks (supports 2/3/4)
        recon_feats = shallow_feats[-1]
        if self.up_factor == 2:
            feats_2x = pre_up_feats[0]
            self.SR0 = SRBlock3D(shallow_feats[-1], feats_2x, k_size=6, pad=2,
                                 upsample_method=upsample_method, upscale_factor=2, use_checkpoint=False)
            recon_feats = feats_2x
        elif self.up_factor == 3:
            feats_3x = pre_up_feats[0]
            self.SR0 = SRBlock3D(shallow_feats[-1], feats_3x, k_size=6, pad=2,
                                 upsample_method=upsample_method, upscale_factor=3, use_checkpoint=False)
            recon_feats = feats_3x
        elif self.up_factor == 4:
            feats_2x = pre_up_feats[0]
            feats_4x = pre_up_feats[1]
            self.SR0 = SRBlock3D(shallow_feats[-1], feats_2x, k_size=6, pad=2,
                                 upsample_method=upsample_method, upscale_factor=2, use_checkpoint=False)
            self.SR1 = SRBlock3D(feats_2x, feats_4x, k_size=6, pad=2,
                                 upsample_method=upsample_method, upscale_factor=2, use_checkpoint=False)
            recon_feats = feats_4x

        self.HRconv = nn.Conv3d(recon_feats, recon_feats, 3, 1, 1, bias=True)
        self.conv_last = nn.Sequential(
            nn.Conv3d(in_channels=recon_feats, out_channels=in_chans, kernel_size=3, stride=1, padding=1, bias=True)
        )

    def crop_next(self, x, level):
        _, _, H, W, D = x.shape
        size = self.context_sizes[level]

        start_h = (H - size) // 2
        start_w = (W - size) // 2
        start_d = (D - size) // 2

        end_h = start_h + size
        end_w = start_w + size
        end_d = start_d + size

        return x[:, :, start_h:end_h, start_w:end_w, start_d:end_d]

    def forward(self, x):
        """
        x: (B, C, H, W, D)
        returns: upsampled output (B, C, up_H, up_W, up_D)
        """

        LX = x  # image-space features (B, C_img, D, H, W)
        prev_x = None

        for level in range(0, self.num_levels):
            # SFE
            LX = self.sfe_blks[level](LX)  # (B, shallow_feats[level], D, H, W)

            # Patch embedding
            LX_emb = self.patch_embedding_blks[level](LX)

            # Pass through LX blocks (placeholder identity blocks)
            for i, blk in enumerate(self.LX_blocks[level]):
                if self.use_checkpoint:
                    LX_emb = checkpoint.checkpoint(blk, LX_emb, prev_x)
                else:
                    LX_emb = blk(LX_emb, prev_x)

                if i == self.num_blks[level] - 1 and level != self.num_levels - 1:
                    prev_x = LX_emb

            # Optionally crop next input image for the next level
            if level < self.num_levels - 1:
                LX = self.crop_next(LX, level + 1)

        if self.use_checkpoint:
            final_feats = checkpoint.checkpoint(self.Final_blk, LX_emb)
        else:
            final_feats = self.Final_blk(LX_emb)

        # final_feats: use highest-level image features
        final_feats = self.lrelu(self.conv_image(final_feats))  # image-space conv + act

        # Long skip connection
        out = LX + final_feats

        # Upsampling
        if self.up_factor == 2:
            out = self.SR0(out)
        elif self.up_factor == 3:
            out = self.SR0(out)
        elif self.up_factor == 4:
            out = self.SR0(out)
            out = self.SR1(out)

        # Recon
        out = self.conv_last(self.lrelu(self.HRconv(out)))
        return out


def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / 10 ** 9 if torch.cuda.is_available() else 0

    patch_size_hr = 128
    down_factor = 4
    print("Test DegradationModel")
    net = DegradeNet(down_factor=down_factor,
                           in_channels=1,
                           out_channels=1,
                           num_feats=64,
                           use_checkpoint=True).to(device)

    print("Number of parameters, G", numel(net, only_trainable=True))

    net.train()  # inference mode

    # Create a random input: batch size 1, 3 channels, 224x224 image
    x_hr = torch.randn(1, 1, patch_size_hr, patch_size_hr, patch_size_hr).to(device)

    x_lr = torch.randn((1, 1, patch_size_hr // down_factor,
                        patch_size_hr // down_factor,
                        patch_size_hr // down_factor)).cuda()

    loss_func = nn.MSELoss()

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

    print("Done")


if __name__ == "__main__":
    test()
