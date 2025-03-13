import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import time
import torch.utils.checkpoint as checkpoint
from models.models_3D import SRBlock3D
from utils.utils_3D_image import ICNR, numel


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    #return nn.Sequential(*layers)
    return nn.ModuleList(layers)


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


class RRDBNet(nn.Module):
    def __init__(self, up_factor=4, in_nc=1, out_nc=1, nf=64, nb=10, gc=32, use_checkpoint=True):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv3d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv3d(nf, nf, 3, 1, 1, bias=True)
        self.use_checkpoint = use_checkpoint

        #### upsampling
        self.up_factor = up_factor
        recon_feats = nf
        reduction = 2
        if self.up_factor >= 2:
            self.upconv1 = nn.Conv3d(nf, nf//reduction, 3, 1, 1, bias=True)
            recon_feats = nf // reduction
        if self.up_factor >= 4:
            self.upconv2 = nn.Conv3d(nf//reduction, nf//reduction, 3, 1, 1, bias=True)
            recon_feats = nf // reduction

        self.HRconv = nn.Conv3d(recon_feats, recon_feats, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv3d(recon_feats, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        #trunk = self.trunk_conv(self.RRDB_trunk(fea))
        #trunk = self.trunk_conv(checkpoint.checkpoint_sequential(self.RRDB_trunk, 6, fea))
        for i, layer in enumerate(self.RRDB_trunk):
            if i % 2 == 0 and self.use_checkpoint:  # and i != len(self.RRDB_trunk) - 1:
                fea = checkpoint.checkpoint(layer, fea)
            else:
                fea = layer(fea)
        trunk = fea

        fea = fea + trunk

        if self.up_factor == 2:
            fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
            out = self.conv_last(self.lrelu(self.HRconv(fea)))
            return out
        elif self.up_factor == 3:
            fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=3, mode='nearest')))
            out = self.conv_last(self.lrelu(self.HRconv(fea)))
            return out
        elif self.up_factor == 4:
            fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
            fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
            out = self.conv_last(self.lrelu(self.HRconv(fea)))
        else:
            out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out


def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / 10 ** 9 if torch.cuda.is_available() else 0

    up_factor = 4
    print("Test RRDBNet3D official")
    generator = RRDBNet(up_factor=up_factor,
                   in_nc=1,
                   out_nc=1,
                   nf=64,
                   nb=12,
                   gc=32).to(device)

    print("Number of parameters, G", numel(generator, only_trainable=True))

    patch_size = 32
    x = torch.randn((1, 1, patch_size, patch_size, patch_size)).to(device)

    generator.train()

    start = time.time()
    # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    gen_out = generator(x)
    stop = time.time()
    print("Time elapsed:", stop - start)
    #discriminator.train()

    x_hr = torch.randn((1, 1, patch_size * up_factor,
                        patch_size * up_factor,
                        patch_size * up_factor)).cuda()

    loss_func = nn.MSELoss()
    loss = loss_func(gen_out, x_hr)
    loss.backward()

    #disc_out = discriminator(gen_out)

    max_memory_reserved = torch.cuda.max_memory_reserved()
    print("Maximum memory reserved: %0.3f Gb / %0.3f Gb" % (max_memory_reserved / 10 ** 9, total_gpu_mem))

    print(gen_out.shape)
    #print(disc_out.shape)

if __name__ == "__main__":
    test()
