import monai.networks.utils
import numpy as np
import torch
import torch.nn as nn
import config
from utils.utils_3D_image import deconvICNR, ICNR, ICNR3D, numel
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SqueezeExcitation, self).__init__()
        reduced_channels = in_channels // reduction
        self.fc1 = nn.Conv3d(in_channels, reduced_channels, kernel_size=1)
        self.fc2 = nn.Conv3d(reduced_channels, in_channels, kernel_size=1)

    def forward(self, x):
        y = F.adaptive_avg_pool3d(x, 1)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y

class FusedMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, kernel_size, stride, se_ratio=0.0):
        super(FusedMBConv, self).__init__()
        self.stride = stride
        self.has_se = se_ratio is not None and se_ratio > 0
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Expansion Convolution (Regular Conv)
        expanded_channels = in_channels * expansion_factor
        self.expand_conv = nn.Conv3d(in_channels, expanded_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.bn0 = nn.BatchNorm3d(expanded_channels)
        self.act0 = nn.ReLU6(inplace=True)

        # Squeeze and Excitation
        if self.has_se:
            self.se = SqueezeExcitation(expanded_channels, reduction=int(1/se_ratio))

        # Pointwise Convolution
        self.project_conv = nn.Conv3d(expanded_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)

        # Residual connection
        self.use_residual = (in_channels == out_channels) and (stride == 1)

    def forward(self, x):
        out = self.act0(self.bn0(self.expand_conv(x)))
        if self.has_se:
            out = self.se(out)
        out = self.bn1(self.project_conv(out))

        if self.use_residual:
            return x + out
        else:
            return out


class Generator(nn.Module):
    def __init__(self, in_c, scale_factor=2, n_sr_vec = [256,256], n_res_vec=[64,64,64,64,64], n_vec=[64,64,3], k_vec=[9,3,9]):
        super().__init__()

        self.conv0 = nn.Conv2d(in_c, n_vec[0], kernel_size=k_vec[0], stride=1, padding=4)
        self.act0 = nn.PReLU(num_parameters=n_vec[0])

        self.res0 = ResidualBlock(n_vec[0], n_res_vec[0], 3)
        self.res1 = ResidualBlock(n_res_vec[0], n_res_vec[1], 3)
        self.res2 = ResidualBlock(n_res_vec[1], n_res_vec[2], 3)
        self.res3 = ResidualBlock(n_res_vec[2], n_res_vec[3], 3)
        self.res4 = ResidualBlock(n_res_vec[3], n_res_vec[4], 3)

        self.conv1 = nn.Conv2d(n_res_vec[4], n_vec[1], kernel_size=k_vec[1], stride=1, padding=1)
        self.norm0 = nn.BatchNorm2d(n_vec[1])

        self.SR0 = SRBlock(n_vec[1], scale_factor=2)
        self.SR1 = SRBlock(n_vec[1], scale_factor=2)

        self.conv2 = nn.Conv2d(n_vec[1], n_vec[2], kernel_size=k_vec[2], stride=1, padding=4)

    def forward(self, input):

        # Initial convolution and PReLU activation
        x = self.act0(self.conv0(input))

        # Residual block network
        z = self.res0(x)
        z = self.res1(z)
        z = self.res2(z)
        z = self.res3(z)
        z = self.res4(z)

        # Convolution with skip connection after residual network
        z = self.norm0(self.conv1(z))
        z = x + z

        # Super-resolution block
        z = self.SR0(z)
        z = self.SR1(z)

        # Final convolution
        out = self.conv2(z)

        return out


class MultiLevelDenseNet(nn.Module):
    def __init__(self, up_factor=1, in_c=1, k_factor=12, k_size=3, num_dense_blocks=4, upsample_method="deconv_nn_resize", use_checkpoint=True):
        super().__init__()

        self.num_dense_blocks = num_dense_blocks
        self.use_checkpoint = use_checkpoint
        self.up_factor = up_factor

        self.conv0 = nn.Conv3d(in_c, 2 * k_factor, k_size, stride=1, padding=1)

        self.dense_blocks = nn.ModuleList()
        for i in range(self.num_dense_blocks):
            self.dense_blocks.append(
                DenseBlock(2 * k_factor, k_factor, k_size)
            )

        self.compress_layers = nn.ModuleList()
        for i in range(num_dense_blocks-1):
            inputs = (2 + 6 * (i + 1))
            self.compress_layers.append(
                nn.Conv3d(inputs * k_factor, 2 * k_factor, kernel_size=1, stride=1, padding=0)
            )

        #self.dense_block0 = DenseBlock(2 * k_factor, k_factor, k_size)
        #self.dense_block1 = DenseBlock(2 * k_factor, k_factor, k_size)
        #self.dense_block2 = DenseBlock(2 * k_factor, k_factor, k_size)
        #self.dense_block3 = DenseBlock(2 * k_factor, k_factor, k_size)

        #self.compress0 = nn.Conv3d(8 * k_factor, 2 * k_factor, kernel_size=1, stride=1, padding=0)
        #self.compress1 = nn.Conv3d(14 * k_factor, 2 * k_factor, kernel_size=1, stride=1, padding=0)
        #self.compress2 = nn.Conv3d(20 * k_factor, 2 * k_factor, kernel_size=1, stride=1, padding=0)
        inputs = (2 + 6 * (self.num_dense_blocks))

        if upsample_method is None:
            # Direct Feature Combination from paper: https://arxiv.org/pdf/2003.01217v1.pdf
            self.recon = nn.Conv3d(inputs * k_factor, 1, kernel_size=1, stride=1, padding=0)
        else:
            # Reconstruction via bottleneck from paper: https://openaccess.thecvf.com/content_ICCV_2017/papers/Tong_Image_Super-Resolution_Using_ICCV_2017_paper.pdf
            self.bottleneck = nn.Conv3d(inputs * k_factor, 8 * k_factor, kernel_size=1, stride=1, padding=0)
            self.act_bottle = nn.PReLU(num_parameters=8 * k_factor)

            # Batch norm operation here?
            if up_factor >= 2:
                self.SR0 = SRBlock3D(8 * k_factor, 8 * k_factor, 6, 2, upsample_method, upscale_factor=2)  # In paper on densenets, they use 8*k channels for in and output
                self.recon = nn.Conv3d(8 * k_factor, 1, kernel_size=k_size, stride=1, padding=1)
            if up_factor >= 4:
                self.SR1 = SRBlock3D(8 * k_factor, 8 * k_factor, 6, 2, upsample_method, upscale_factor=2)
                self.recon = nn.Conv3d(8 * k_factor, 1, kernel_size=k_size, stride=1, padding=1)
            # Final reconstruction with kernel size 3
            if up_factor == 1:
                self.recon = nn.Conv3d(8 * k_factor, 1, kernel_size=k_size, stride=1, padding=1)


    def forward(self, input):

        """
        The checkpointing implemented in this function could be improved by instead adding checkpoints
        for each dense unit and maybe inside each SR block
        :param input:
        :return:
        """

        # Initial convolution with 2k output filters
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.conv0, input)
        else:
            x = self.conv0(input)

        # Old version:
        # z = self.dense_block0(x)
        # skip = torch.cat([x, z], 1)
        # comp = self.compress0(skip)
        #
        # z = self.dense_block1(comp)
        # skip = torch.cat([skip, z], 1)
        # comp = self.compress1(skip)
        #
        # z = self.dense_block2(comp)
        # skip = torch.cat([skip, z], 1)
        # comp = self.compress2(skip)
        #
        # z = self.dense_block3(comp)
        # final_skip = torch.cat([skip, z], 1)

        # New version:
        next_input = x
        skip = x
        for i in range(self.num_dense_blocks-1):
            if self.use_checkpoint:
                z = checkpoint.checkpoint(self.dense_blocks[i], next_input)
            else:
                z = self.dense_blocks[i](next_input)
            skip = torch.cat([skip, z], 1)
            next_input = self.compress_layers[i](skip)

        if self.use_checkpoint:
            z = checkpoint.checkpoint(self.dense_blocks[-1], next_input)
        else:
            z = self.dense_blocks[-1](next_input)

        # Rest here is the same:
        final_skip = torch.cat([skip, z], 1)

        #z = self.act_bottle(self.bottleneck(final_skip))
        if self.use_checkpoint:
            z = checkpoint.checkpoint(self.bottleneck, final_skip)
        else:
            z = self.bottleneck(final_skip)

        if self.up_factor >= 2:
            if self.use_checkpoint:
                z = checkpoint.checkpoint(self.SR0, z)
            else:
                z = self.SR0(z)
        if self.up_factor >= 4:
            if self.use_checkpoint:
                z = checkpoint.checkpoint(self.SR1, z)
            else:
                z = self.SR1(z)

        if self.use_checkpoint:
            out = checkpoint.checkpoint(self.recon, z)
        else:
            out = self.recon(z)

        return out


class DenseBlock(nn.Module):
    def __init__(self, in_c, k_factor=12, k_size=3):
        super().__init__()

        # Four or seven dense blocks
        self.dense_unit0 = DenseUnit(in_c, k_factor, k_size)
        self.dense_unit1 = DenseUnit(in_c + k_factor, k_factor, k_size)
        self.dense_unit2 = DenseUnit(in_c + 2*k_factor, k_factor, k_size)
        self.dense_unit3 = DenseUnit(in_c + 3*k_factor, k_factor, k_size)

    def forward(self, input):

        x0 = self.dense_unit0(input)
        skip0 = torch.cat([input, x0], 1)

        x1 = self.dense_unit1(skip0)
        skip1 = torch.cat([skip0, x1], 1)

        x2 = self.dense_unit2(skip1)
        skip2 = torch.cat([skip1, x2], 1)

        x3 = self.dense_unit3(skip2)
        out = torch.cat([skip2, x3], 1)

        return out


class DenseUnit(nn.Module):
    def __init__(self, in_c, k_factor=12, k_size=3):
        super().__init__()

        self.norm0 = nn.BatchNorm3d(num_features=in_c)
        self.act0 = nn.ELU(alpha=1.0)
        self.conv0 = nn.Conv3d(in_c, k_factor, kernel_size=k_size, stride=1, padding=1)

    def forward(self, x):
        return self.conv0(self.act0(self.norm0(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_c, n=64, k_size=3):
        super().__init__()

        self.conv0 = nn.Conv2d(in_c, n, kernel_size=k_size, stride=1, padding=1)
        self.norm0 = nn.BatchNorm2d(n)
        self.act0 = nn.PReLU(num_parameters=n)

        self.conv1 = nn.Conv2d(n, n, kernel_size=k_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(n)

    def forward(self, x):
        r = self.norm0(self.conv0(x))
        r = self.act0(r)
        r = self.norm1(self.conv1(r))
        out = x + r

        return out


class SRBlock(nn.Module):
    def __init__(self, in_c, k_size=3, scale_factor=2):
        super().__init__()

        self.conv0 = nn.Conv2d(in_c, in_c * scale_factor**2, kernel_size=k_size, stride=1, padding=1)
        self.shuffle0 = nn.PixelShuffle(scale_factor)
        # We could also try ELU or LReLU activation here
        self.act0 = nn.PReLU(num_parameters=in_c)

    def forward(self, x):
        x = self.conv0(x)
        out = self.act0(self.shuffle0(x))

        return out


class SRBlock3D(nn.Module):
    def __init__(self, in_c, n, k_size=6, pad=2, upsample_method="deconv_nn_resize", upscale_factor=2, use_checkpoint=False, skip_act=False):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.upsample_method = upsample_method
        #self.channel_reduction = channel_reduction
        self.in_c = in_c
        self.n = n
        self.skip_act = skip_act

        if self.upsample_method == 'deconv_nn_resize':
            if upscale_factor == 3:
                raise Exception("deconv_nn_resize not feasible for 3x upscaling")

            self.deconv0 = nn.ConvTranspose3d(in_c, n, kernel_size=k_size, stride=2, padding=pad)
            if self.skip_act:
                self.act0 = nn.Identity()
            else:
                self.act0 = nn.PReLU(num_parameters=n)

            # ICNR is an initialization method for sub-pixel convolution which removes checkerboarding
            # From the paper: Checkerboard artifact free sub-pixel convolution.
            # https://gist.github.com/A03ki/2305398458cb8e2155e8e81333f0a965
            # weight = ICNR(self.deconv0.weight, initializer=nn.init.normal_, upscale_factor=upscale_factor, mean=0.0, std=0.02)

            # Trying custom ICNR method using approach explained in this paper: https://arxiv.org/pdf/1812.11440.pdf
            deconv_weight = deconvICNR(self.deconv0.weight, initializer=nn.init.normal_, upscale_factor=upscale_factor, mean=0.0, std=0.02)
            self.deconv0.weight.data.copy_(deconv_weight)

        elif self.upsample_method == 'pixelshuffle3D':
            if in_c != n:
                pre_conv_channels = n
                self.pre_conv = nn.Conv3d(in_c, pre_conv_channels, kernel_size=3, stride=1, padding=1)
            else:
                pre_conv_channels = in_c

            # Pixelshuffle3D w. ICNR
            # Follows method suggested in https://gist.github.com/A03ki/2305398458cb8e2155e8e81333f0a965
            self.upscale_factor = upscale_factor
            self.pre_up_conv = nn.Conv3d(pre_conv_channels, pre_conv_channels*(upscale_factor**3), kernel_size=3, stride=1, padding=1)
            weight = ICNR3D(self.pre_up_conv.weight, initializer=nn.init.normal_, upscale_factor=upscale_factor, mean=0.0, std=0.02)
            self.pre_up_conv.weight.data.copy_(weight)

            if self.skip_act:
                self.act0 = nn.Identity()
            else:
                self.act0 = nn.PReLU(num_parameters=n)

        elif self.upsample_method == "Monaipixelshuffle":
            #Monai pixelshuffle upsample block
            if in_c != n:
                pre_conv_channels = n
                self.pre_conv = nn.Conv3d(in_c, pre_conv_channels, kernel_size=3, stride=1, padding=1)
            else:
                pre_conv_channels = in_c
                self.pre_conv = nn.Identity()

            # Supports ICNR and padding/pooling natively
            self.sr_block = monai.networks.blocks.UpSample(3,
                                           in_channels=pre_conv_channels,
                                           out_channels=None,
                                           scale_factor=upscale_factor,
                                           kernel_size=None,
                                           size=None,
                                           mode="pixelshuffle",
                                           pre_conv='default',
                                           interp_mode="linear",
                                           align_corners=True,
                                           bias=True,
                                           apply_pad_pool=True)

        else:
            # Interpolation followed by convolution + activation
            # Supported modes are 'nearest' and 'trilinear'
            align_corners = True if self.upsample_method == "trilinear" else None

            self.interp = nn.Upsample(scale_factor=upscale_factor, mode=self.upsample_method, align_corners=align_corners)  # Supports 3D natively
            self.post_conv = nn.Conv3d(in_c, n, kernel_size=3, stride=1, padding=1, bias=True)
            if self.skip_act:
                self.act0 = nn.Identity()
            else:
                self.act0 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):

        if self.upsample_method == 'deconv_nn_resize':
            # Taken from this paper, rightmost method in figure 3: https://arxiv.org/pdf/1812.11440.pdf
            #if self.use_checkpoint:
            #    x = checkpoint.checkpoint(self.deconv0, x)
            #else:
            #    x = self.deconv0(x)
            # out = checkpoint.checkpoint(self.act0, x)

            #x = self.deconv0(x)
            #out = self.act0(x)

            out = self.act0(self.deconv0(x))

            #out = self.deconv0(x)

        elif self.upsample_method == 'pixelshuffle3D':
            # Pixelshuffle3D w. ICNR implementation from kuoweilai
            # https://github.com/kuoweilai/pixelshuffle3d/blob/master/pixelshuffle3d.py
            # Follows method suggested in https://gist.github.com/A03ki/2305398458cb8e2155e8e81333f0a965

            # Convolution to reduce channels before increasing depth
            if self.in_c != self.n:
                x = self.pre_conv(x)

            # Initial convolution to increase depth
            x = self.pre_up_conv(x)

            # Pixelshuffle 3D
            batch_size, channels, in_depth, in_height, in_width = x.size()
            nOut = channels // self.upscale_factor ** 3

            out_depth = in_depth * self.upscale_factor
            out_height = in_height * self.upscale_factor
            out_width = in_width * self.upscale_factor

            x = x.contiguous().view(batch_size, nOut, self.upscale_factor, self.upscale_factor, self.upscale_factor, in_depth, in_height, in_width)
            x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
            x = x.view(batch_size, nOut, out_depth, out_height, out_width)

            # Activation after pixelshuffle
            out = self.act0(x)

        elif self.upsample_method == 'Monaipixelshuffle':
            out = self.sr_block(self.pre_conv(x))
            #out = self.sr_block(x)

        else:
            # Interpolation followed by convolution + activation
            out = self.act0(self.post_conv(self.interp(x)))

        return out

# class SRBlock3D(nn.Module):
#     def __init__(self, in_channels, out_channels, k_size=6, pad=2, upsample_method="deconv_nn_resize", upscale_factor=2):
#         super().__init__()
#         self.upsample_method = upsample_method
#
#         if self.upsample_method == 'deconv_nn_resize':
#             self.deconv0 = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=k_size, stride=2, padding=pad)
#             self.act0 = nn.PReLU(num_parameters=out_channels)
#             weight = ICNR(self.deconv0.weight, initializer=nn.init.normal_, upscale_factor=2, mean=0.0, std=0.02)
#             self.deconv0.weight.data.copy_(weight)
#
#         elif self.upsample_method == 'pixelshuffle3D':
#             self.upscale_factor = upscale_factor
#             self.pre_conv = nn.Conv3d(in_channels, in_channels*(upscale_factor**3), kernel_size=3, stride=1, padding=1)
#             weight = ICNR(self.pre_conv.weight, initializer=nn.init.normal_, upscale_factor=upscale_factor, mean=0.0, std=0.02)
#             self.pre_conv.weight.data.copy_(weight)
#             self.act0 = nn.PReLU(num_parameters=out_channels)
#
#         elif self.upsample_method == "Monaipixelshuffle":
#             self.sr_block = monai.networks.blocks.UpSample(3,
#                                                            in_channels=in_channels,
#                                                            out_channels=None,
#                                                            scale_factor=upscale_factor,
#                                                            kernel_size=None,
#                                                            size=None,
#                                                            mode="pixelshuffle",
#                                                            pre_conv='default',
#                                                            interp_mode="linear",
#                                                            align_corners=True,
#                                                            bias=True,
#                                                            apply_pad_pool=True)
#
#         else:
#             self.interp = nn.Upsample(scale_factor=upscale_factor, mode=upsample_method)
#             self.post_conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
#             self.act = nn.LeakyReLU(0.2, inplace=True)
#
#     def forward(self, x):
#         if self.upsample_method == 'deconv_nn_resize':
#             x = self.deconv0(x)
#             out = self.act0(x)
#
#         elif self.upsample_method == 'pixelshuffle3D':
#             x = self.pre_conv(x)
#             batch_size, channels, in_depth, in_height, in_width = x.size()
#             n_out = channels // self.upscale_factor ** 3
#             out_depth = in_depth * self.upscale_factor
#             out_height = in_height * self.upscale_factor
#             out_width = in_width * self.upscale_factor
#             x = x.contiguous().view(batch_size, n_out, self.upscale_factor, self.upscale_factor, self.upscale_factor, in_depth, in_height, in_width)
#             x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
#             x = x.view(batch_size, n_out, out_depth, out_height, out_width)
#             out = self.act0(x)
#
#         elif self.upsample_method == 'Monaipixelshuffle':
#             out = self.sr_block(x)
#
#         else:
#             out = self.act(self.post_conv(self.interp(x)))
#
#         return out


class LRconvBlock3D(nn.Module):
    def __init__(self, input_size, in_c, n, k_size, stride):
        super().__init__()

        self.conv0 = nn.Conv3d(in_c, n, kernel_size=k_size, stride=stride, padding=1)
        #self.norm0 = nn.LayerNorm([DCSRN_config.BATCH_SIZE, n, 16, 16, 16])
        self.norm0 = nn.LayerNorm(input_size)
        self.act0 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv0(x)
        out = self.act0(self.norm0(x))

        return out


class Discriminator(nn.Module):
    def __init__(self, input_size=64, in_c=1, n_conv_vec=[64,64,128,128,256,256,512,512], n_dense=[1024, 1], k_size=3, use_checkpoint=True):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.conv0 = nn.Conv3d(in_c, n_conv_vec[0], kernel_size=k_size, padding=1, stride=1)
        self.act0 = nn.LeakyReLU(0.2, inplace=True)

        dim = [int(np.ceil(input_size/2**i)) for i in range(1,5)]

        self.LRconv0 = LRconvBlock3D(dim[0], n_conv_vec[0], n_conv_vec[1], k_size, stride=2)
        self.LRconv1 = LRconvBlock3D(dim[0], n_conv_vec[1], n_conv_vec[2], k_size, stride=1)
        self.LRconv2 = LRconvBlock3D(dim[1], n_conv_vec[2], n_conv_vec[3], k_size, stride=2)
        self.LRconv3 = LRconvBlock3D(dim[1], n_conv_vec[3], n_conv_vec[4], k_size, stride=1)
        self.LRconv4 = LRconvBlock3D(dim[2], n_conv_vec[4], n_conv_vec[5], k_size, stride=2)
        self.LRconv5 = LRconvBlock3D(dim[2], n_conv_vec[5], n_conv_vec[6], k_size, stride=1)
        self.LRconv6 = LRconvBlock3D(dim[3], n_conv_vec[6], n_conv_vec[7], k_size, stride=2)

        # self.LRconv0 = LRconvBlock3D(dim[0], n_conv_vec[0], n_conv_vec[1], k_size, stride=2)
        # self.LRconv1 = LRconvBlock3D(dim[0], n_conv_vec[1], n_conv_vec[2], k_size, stride=1)
        # self.LRconv2 = LRconvBlock3D(dim[1], n_conv_vec[2], n_conv_vec[3], k_size, stride=2)
        # self.LRconv3 = LRconvBlock3D(dim[1], n_conv_vec[3], n_conv_vec[4], k_size, stride=1)
        # self.LRconv4 = LRconvBlock3D(dim[2], n_conv_vec[4], n_conv_vec[5], k_size, stride=2)
        # self.LRconv5 = LRconvBlock3D(dim[2], n_conv_vec[5], n_conv_vec[6], k_size, stride=1)
        # self.LRconv6 = LRconvBlock3D(dim[3], n_conv_vec[6], n_conv_vec[7], k_size, stride=2)

        self.flatten = nn.Flatten()
        ll_size = int(n_conv_vec[7] * dim[3]**3)
        self.dense0 = nn.Linear(ll_size, n_dense[0])
        self.act1 = nn.LeakyReLU(0.2, inplace=True)  # Should perhaps be nn.LeakyReLU(0.2, inplace=True)
        self.dense1 = nn.Linear(n_dense[0], n_dense[1])

        self.act_sigmoid = nn.Sigmoid()


    def forward(self, input):

        # Initial convolution and LeakyRelu activation
        x = self.act0(self.conv0(input))

        # LeakyRelu convolution block network
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.LRconv0, x)
            x = checkpoint.checkpoint(self.LRconv1, x)
            x = checkpoint.checkpoint(self.LRconv2, x)
            x = checkpoint.checkpoint(self.LRconv3, x)
            x = checkpoint.checkpoint(self.LRconv4, x)
            x = checkpoint.checkpoint(self.LRconv5, x)
            x = checkpoint.checkpoint(self.LRconv6, x)
        else:
            x = self.LRconv0(x)
            x = self.LRconv1(x)
            x = self.LRconv2(x)
            x = self.LRconv3(x)
            x = self.LRconv4(x)
            x = self.LRconv5(x)
            x = self.LRconv6(x)

        # Dense block network + LeakyRelu
        x = self.dense0(self.flatten(x))
        x = self.act1(x)
        out = self.dense1(x)

        # Final sigmoid activation (Remember to remove if BCEWithLogitsLoss() is used in training loop)
        #out = self.act_sigmoid(out)

        return out

class DiscriminatorV2(nn.Module):
    def __init__(self, patch_size=64, up_factor=1, in_c=1, n_conv_vec=[64,64,128,128,256,256,512,512], n_dense=[1024, 1], k_size=3, use_checkpoint=True):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.conv0 = nn.Conv3d(in_c, n_conv_vec[0], kernel_size=k_size, padding=1, stride=1)
        self.act0 = nn.LeakyReLU(0.2, inplace=True)

        dim = [int(np.ceil(patch_size*up_factor / 2 ** i)) for i in range(1, 5)]
        #dim = [int(torch.ceil(input_size/2**i)) for i in range(1,5)]

        self.blocks = nn.ModuleList()
        dim_idx = 0
        for idx in range(len(n_conv_vec) - 1):
            stride = 2 - (idx % 2)
            self.blocks.append(
                LRconvBlock3D(
                    dim[dim_idx],
                    n_conv_vec[idx],
                    n_conv_vec[idx+1],
                    k_size=3,
                    stride=stride,  # set stride to 2 for every other block
                    #padding=1,
                    #bias=True,
                )
            )
            dim_idx = dim_idx + (stride % 2)  # Set input features to output of previous block

        #self.LRconv0 = LRconvBlock3D(dim[0], n_conv_vec[0], n_conv_vec[1], k_size, stride=1)
        #self.LRconv1 = LRconvBlock3D(dim[1], n_conv_vec[1], n_conv_vec[2], k_size, stride=2)
        #self.LRconv2 = LRconvBlock3D(dim[1], n_conv_vec[2], n_conv_vec[3], k_size, stride=1)
        #self.LRconv3 = LRconvBlock3D(dim[2], n_conv_vec[3], n_conv_vec[4], k_size, stride=2)
        #self.LRconv4 = LRconvBlock3D(dim[2], n_conv_vec[4], n_conv_vec[5], k_size, stride=1)
        #self.LRconv5 = LRconvBlock3D(dim[3], n_conv_vec[5], n_conv_vec[6], k_size, stride=2)
        #self.LRconv6 = LRconvBlock3D(dim[3], n_conv_vec[6], n_conv_vec[7], k_size, stride=1)

        self.flatten = nn.Flatten()
        ll_size = int(n_conv_vec[-1] * dim[-1]**3)
        self.dense0 = nn.Linear(ll_size, n_dense[0])
        self.act1 = nn.LeakyReLU(0.2, inplace=True)  # Should perhaps be nn.LeakyReLU(0.2, inplace=True)
        self.dense1 = nn.Linear(n_dense[0], n_dense[1])

        self.act_sigmoid = nn.Sigmoid()


    def forward(self, input):

        # Initial convolution and LeakyRelu activation
        x = self.act0(self.conv0(input))

        # LeakyRelu convolution block network
        for block in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x)
            else:
                x = block(x)


        #x = self.LRconv0(x)
        #x = self.LRconv1(x)
        #x = self.LRconv2(x)
        #x = self.LRconv3(x)
        #x = self.LRconv4(x)
        #x = self.LRconv5(x)
        #x = self.LRconv6(x)

        # Dense block network + LeakyRelu
        x = self.dense0(self.flatten(x))
        x = self.act1(x)
        out = self.dense1(x)

        # Final sigmoid activation (Remember to remove if BCEWithLogitsLoss() is used in training loop)
        #out = self.act_sigmoid(out)

        return out




if __name__ == "__main__":

    input_tensor = torch.randn(1, 32, 48, 48, 48)  # Batch size 1, 32 channels, 56x56 feature map
    fused_mbconv = FusedMBConv(in_channels=128, out_channels=128, expansion_factor=4, kernel_size=3, stride=1,
                               se_ratio=0.25)
    output_tensor = fused_mbconv(input_tensor)
    print(output_tensor.shape)  # Should output torch.Size([1, 32, 56, 56])


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / 10 ** 9 if torch.cuda.is_available() else 0

    n_conv_vec = [64, 64, 128, 128, 256, 256, 512, 512]
    # n_conv_vec = [32, 32, 64, 64, 128, 128, 256, 256]
    #n_conv_vec = [16, 16, 32, 32, 64, 64, 128, 128]
    #n_conv_vec = [8, 8, 16, 16, 32, 32, 64, 64]
    #n_dense = [1024, 1]
    n_dense = [512, 1]
    #n_dense = [64, 1]  # n_dense = [512, 1]
    #n_dense = [128, 1]

    in_c = 1
    patch_size = 32
    up_factor = 2
    k_size = 3

    # Create an instance of the Discriminator model
    discriminator = DiscriminatorV2(patch_size,
                                    up_factor=up_factor,
                                    in_c=in_c,
                                    n_conv_vec=n_conv_vec,
                                    n_dense=n_dense,
                                    k_size=k_size,
                                    use_checkpoint=True).to(device)

    # Create an instance of the Generator model
    generator = MultiLevelDenseNet(up_factor=up_factor,
                                   in_c=in_c,
                                   k_factor=12,
                                   k_size=k_size,
                                   num_dense_blocks=4,
                                   upsample_method="pixelshuffle3D",  # "deconv_nn_resize" "pixelshuffle3D"
                                   use_checkpoint=True).to(device)

    #monai.networks.utils.pixelshuffle()

    print("Number of parameters, G", numel(generator, only_trainable=True))
    print("Number of parameters, D", numel(discriminator, only_trainable=True))

    # Test generator on zero data
    x = torch.zeros(2, 1, patch_size, patch_size, patch_size).to(device)
    #generator_output = generator(x.to(config.DEVICE))
    ## Test discriminator
    #discriminator_output = discriminator(generator_output)

    generator.train()
    discriminator.train()

    generator_output = generator(x)
    discriminator_output = discriminator(generator_output)

    max_memory_reserved = torch.cuda.max_memory_reserved()
    print("Maximum memory reserved: %0.3f Gb / %0.3f Gb" % (max_memory_reserved / 10 ** 9, total_gpu_mem))

    # See the generator output shape
    print("Shapes")
    print("Input patch shape", x.shape)
    print("Generator output shape", generator_output.shape)
    print("Discriminator output shape", discriminator_output.shape)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(generator_output[0, 0, :, 32].detach().cpu(), cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(generator_output[0, 0, 32, :].detach().cpu(), cmap='gray')
    plt.show()

    if False:
        test_load = nib.load('3D_datasets/datasets/IXI/train/IXI002-Guys-0828-T1.nii').get_fdata()
        print("IXI sample size", test_load.shape)
        I_HR = test_load[:, :, test_load.shape[-1] // 2]
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(I_HR, cmap='gray')
        plt.subplot(1, 2, 2)
        if config.BATCH_SIZE > 1:
            I_SR = np.squeeze(generator_output.detach().cpu().numpy())[0, :, :, generator_output.shape[-1] // 2]
        else:
            I_SR = np.squeeze(generator_output.detach().cpu().numpy())[:, :, generator_output.shape[-1] // 2]
        plt.imshow(I_SR, cmap='gray')

        plt.show()


    print("Done")