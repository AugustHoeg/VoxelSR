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

        z = self.conv_first(x)

        for i, block in enumerate(self.conv_blocks):
            if self.use_checkpoint and (i + 1 % 2) == 0:
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


def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / 10 ** 9 if torch.cuda.is_available() else 0

    patch_size_hr = 64
    down_factor = 4
    print("Test DegradationModel")
    net = DegradeNet(down_factor=down_factor,
                           in_channels=1,
                           out_channels=1,
                           num_feats=32,
                           use_checkpoint=False).to(device)

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
