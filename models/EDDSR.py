import time

import torch
from torch import nn

from utils.utils_3D_image import ICNR, numel
from models.models_3D import SRBlock3D

class DenseBlock(nn.Module):
    def __init__(self):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv3d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv3d(in_channels=48, out_channels=24, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv3d(in_channels=72, out_channels=24, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5 = nn.Conv3d(in_channels=96, out_channels=24, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x0 = self.relu(self.conv1(x))
        x1 = self.relu(self.conv2(x0))
        x_add1 = [x0,x1]
        x_add1 = torch.cat(x_add1,1)
        x2 = self.relu(self.conv3(x_add1))
        x_add2 = [x_add1,x2]
        x_add2 = torch.cat(x_add2,1)
        x3 = self.relu(self.conv4(x_add2))
        x_add3 = [x_add2,x3]
        x_add3 = torch.cat(x_add3,1)
        x4 = self.relu(self.conv5(x_add3))

        return x4


class EDDSR(nn.Module):
    def __init__(self, up_factor=2, upsample_method="deconv_nn_resize"):
        super(EDDSR, self).__init__()

        self.up_factor = up_factor
        self.upsample_method = upsample_method

        self.conv_input = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
        )
        self.Block1 = DenseBlock()
        self.Block2 = DenseBlock()
        self.Block3 = DenseBlock()
        self.Block4 = DenseBlock()
        # Original upsampling block from paper:
        #self.deconv = nn.ConvTranspose3d(in_channels=120, out_channels=1, kernel_size=8, stride=2, padding=3, bias=False)
        # Custom SR Block
        recon_feats = 120
        if up_factor >= 2:
            self.SR0 = SRBlock3D(120, 60, k_size=6, pad=2, upsample_method="deconv_nn_resize", upscale_factor=2, use_checkpoint=False)
            recon_feats = 60
        if up_factor >= 4:
            self.SR1 = SRBlock3D(60, 60, k_size=6, pad=2, upsample_method="deconv_nn_resize", upscale_factor=2, use_checkpoint=False)
            recon_feats = 60
        self.recon = nn.Conv3d(in_channels=recon_feats, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x0 = self.conv_input(x)
        x1 = self.Block1(x0)
        x2 = self.Block2(x1)
        x3 = self.Block3(x2)
        x4 = self.Block4(x3)
        x_add = [x0,x1,x2,x3,x4]
        x_add = torch.cat(x_add, 1)
        # Original upsampling block from paper:
        #x_out = self.deconv(x_add)
        # Custom SR Block
        if self.up_factor >= 2:
            x_add = self.SR0(x_add)
        if self.up_factor >= 4:
            x_add = self.SR1(x_add)
        x_out = self.recon(x_add)
        return x_out

    def my_weights_init(self):
        """init the weight for a network"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight.data,
                    a=0,
                    mode="fan_in",
                    nonlinearity="relu"
                )
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.normal_(
                    m.weight.data,
                    mean=0,
                    std=0.001
                )
                if m.bias is not None:
                    m.bias.data.zero_()


class EDDSR_xs(nn.Module):
    def __init__(self, up_factor=2):
        super(EDDSR_xs, self).__init__()

        if up_factor > 2:
            self.pre_upsampler = nn.Upsample(scale_factor=up_factor/2, mode='trilinear', align_corners=True)
            self.up_factor = 2
            self.deconv = nn.ConvTranspose3d(in_channels=120, out_channels=1, kernel_size=8, stride=2, padding=3, bias=False)
        elif up_factor == 2:
            self.pre_upsampler = nn.Identity()
            self.up_factor = 2
            self.deconv = nn.ConvTranspose3d(in_channels=120, out_channels=1, kernel_size=8, stride=2, padding=3, bias=False)
        elif up_factor == 1:
            self.pre_upsampler = nn.Identity()
            self.up_factor = 1
            self.deconv = nn.Conv3d(in_channels=120, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)

        self.conv_input = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
        )
        self.Block1 = DenseBlock()
        self.Block2 = DenseBlock()
        self.Block3 = DenseBlock()
        self.Block4 = DenseBlock()

        # Custom SR Block
        # recon_feats = 120
        # if self.up_factor >= 2:
        #     self.SR0 = SRBlock3D(120, 60, k_size=6, pad=2, upsample_method=self.upsample_method, upscale_factor=2, use_checkpoint=False)
        #     recon_feats = 60
        # if self.up_factor >= 4:
        #     self.SR1 = SRBlock3D(60, 60, k_size=6, pad=2, upsample_method=self.upsample_method, upscale_factor=2, use_checkpoint=False)
        #     recon_feats = 60
        # self.recon = nn.Conv3d(in_channels=recon_feats, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):

        x = self.pre_upsampler(x)

        x0 = self.conv_input(x)
        x1 = self.Block1(x0)
        x2 = self.Block2(x1)
        x3 = self.Block3(x2)
        x4 = self.Block4(x3)
        x_add = [x0,x1,x2,x3,x4]
        x_add = torch.cat(x_add, 1)

        # Original upsampling block from paper:
        x_out = self.deconv(x_add)

        # Custom SR Block
        # if self.up_factor >= 2:
        #     x_add = self.SR0(x_add)
        # if self.up_factor >= 4:
        #     x_add = self.SR1(x_add)
        # x_out = self.recon(x_add)
        return x_out

    def my_weights_init(self):
        """init the weight for a network"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight.data,
                    a=0,
                    mode="fan_in",
                    nonlinearity="relu"
                )
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.normal_(
                    m.weight.data,
                    mean=0,
                    std=0.001
                )
                if m.bias is not None:
                    m.bias.data.zero_()


def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / 10 ** 9 if torch.cuda.is_available() else 0

    patch_size = 64
    x = torch.randn((1, 1, patch_size, patch_size, patch_size)).to(device)

    print("Test EDDSR")
    # generator = EDDSR(up_factor=4, upsample_method="deconv_nn_resize").to(device)
    generator = EDDSR_xs(up_factor=1).to(device)

    print("Input patch size:", patch_size)
    print("Number of parameters, G", numel(generator, only_trainable=True))

    #generator.train()
    #generator.eval()

    start = time.time()
    #with torch.inference_mode():
    gen_out = generator(x)
    stop = time.time()
    print("Time elapsed:", stop - start)

    max_memory_reserved = torch.cuda.max_memory_reserved()
    print("Maximum memory reserved: %0.3f Gb / %0.3f Gb" % (max_memory_reserved / 10 ** 9, total_gpu_mem))

    print(gen_out.shape)

if __name__ == "__main__":
    test()