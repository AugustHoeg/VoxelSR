import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.utils_3D_image import ICNR, numel
from models.models_3D import SRBlock3D

# -------------------------------
# RDN encoder network
# <Zhang, Yulun, et al. "Residual dense network for image super-resolution.">
# Here code is modified from: https://github.com/yjn870/RDN-pytorch/blob/master/models.py
# -------------------------------
class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(
            *[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])
        # local feature fusion
        self.lff = nn.Conv3d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning


class RDN(nn.Module):
    def __init__(self, feature_dim=128, num_features=64, growth_rate=64, num_blocks=8, num_layers=3):
        super(RDN, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers
        # shallow feature extraction
        self.sfe1 = nn.Conv3d(1, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv3d(num_features, num_features, kernel_size=3, padding=3 // 2)
        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))
        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv3d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv3d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )
        self.output = nn.Conv3d(self.G0, feature_dim, kernel_size=3, padding=3 // 2)

    def forward(self, x):
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)
        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)
        x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        x = self.output(x)
        return x


# -------------------------------
# ResCNN encoder network
# <Du, Jinglong, et al. "Super-resolution reconstruction of single
# anisotropic 3D MR images using residual convolutional neural network.">
# -------------------------------
class ResCNN(nn.Module):
    def __init__(self, feature_dim=128):
        super(ResCNN, self).__init__()
        self.conv_start = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True)
        )
        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True)
        )
        self.conv_end = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=feature_dim, kernel_size=3, padding=3 // 2),
        )

    def forward(self, x):
        in_block1 = self.conv_start(x)
        out_block1 = self.block1(in_block1)
        in_block2 = out_block1 + in_block1
        out_block2 = self.block2(in_block2)
        in_block3 = out_block2 + in_block2
        out_block3 = self.block3(in_block3)
        res_img = self.conv_end(out_block3 + in_block3)
        return x + res_img


# -------------------------------
# SRResNet
# <Ledig, Christian, et al. "Photo-realistic single image super-resolution
# using a generative adversarial network.">
# -------------------------------
def conv(ni, nf, kernel_size=3, actn=False):
    layers = [nn.Conv3d(ni, nf, kernel_size, padding=kernel_size // 2)]
    if actn: layers.append(nn.ReLU(True))
    return nn.Sequential(*layers)


class ResSequential(nn.Module):
    def __init__(self, layers, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.m = nn.Sequential(*layers)

    def forward(self, x): return x + self.m(x) * self.res_scale


def res_block(nf):
    return ResSequential(
        [conv(nf, nf, actn=True), conv(nf, nf)],
        1.0)  # this is best one


class SRResnet(nn.Module):
    def __init__(self, nf=64, feature_dim=128):
        super().__init__()
        features = [conv(1, nf)]
        for i in range(18): features.append(res_block(nf))
        features += [conv(nf, nf),
                     conv(nf, feature_dim)]
        self.features = nn.Sequential(*features)

    def forward(self, x):
        return self.features(x)


import torch.nn as nn

# -------------------------------
# decoder implemented by a simple MLP
# -------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=128 + 3, out_dim=1, depth=4, width=256):
        super(MLP, self).__init__()
        stage_one = []
        stage_two = []
        for i in range(depth):
            if i == 0:
                stage_one.append(nn.Linear(in_dim, width))
                stage_two.append(nn.Linear(in_dim, width))
            elif i == depth - 1:
                stage_one.append(nn.Linear(width, in_dim))
                stage_two.append(nn.Linear(width, out_dim))
            else:
                stage_one.append(nn.Linear(width, width))
                stage_two.append(nn.Linear(width, width))
            stage_one.append(nn.ReLU())
            stage_two.append(nn.ReLU())
        self.stage_one = nn.Sequential(*stage_one)
        self.stage_two = nn.Sequential(*stage_two)

    def forward(self, x):
        h = self.stage_one(x)
        return self.stage_two(x + h)

# -------------------------------
# ArSSR model
# -------------------------------
class ArSSR(nn.Module):
    def __init__(self, encoder_name, feature_dim, decoder_depth, decoder_width):
        super(ArSSR, self).__init__()
        if encoder_name == 'RDN':
            self.encoder = RDN(feature_dim=feature_dim)
        if encoder_name == 'SRResNet':
            self.encoder = SRResnet(feature_dim=feature_dim)
        if encoder_name == 'ResCNN':
            self.encoder = ResCNN(feature_dim=feature_dim)
        self.decoder = MLP(in_dim=feature_dim + 3, out_dim=1, depth=decoder_depth, width=decoder_width)

    def forward(self, img_lr, xyz_hr):
        """
        :param img_lr: N×1×h×w×d
        :param xyz_hr: N×K×3
        Note that,
            N: batch size  (N in Equ. 3)
            K: coordinate sample size (K in Equ. 3)
            {h,w,d}: dimensional size of LR input image
        """
        # extract feature map from LR image
        feature_map = self.encoder(img_lr)  # N×1×h×w×d
        # generate feature vector for coordinate through trilinear interpolation (Equ. 4 & Fig. 3).
        feature_vector = F.grid_sample(feature_map, xyz_hr.flip(-1).unsqueeze(1).unsqueeze(1),
                                       mode='bilinear',
                                       align_corners=False)[:, :, 0, 0, :].permute(0, 2, 1)
        # concatenate coordinate with feature vector
        feature_vector_and_xyz_hr = torch.cat([feature_vector, xyz_hr], dim=-1)  # N×K×(3+feature_dim)
        # estimate the voxel intensity at the coordinate by using decoder.
        N, K = xyz_hr.shape[:2]
        intensity_pre = self.decoder(feature_vector_and_xyz_hr.view(N * K, -1)).view(N, K, -1)
        return intensity_pre


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / 10 ** 9 if torch.cuda.is_available() else 0

    patch_size = 32
    up_factor = 3
    x = torch.randn((1, 1, patch_size, patch_size, patch_size)).to(device)

    print("Test ArSSR")
    decoder_depth = 8
    decoder_width = 256
    feature_dim = 128
    encoder_name = "ResCNN"
    generator = ArSSR(encoder_name=encoder_name,
                      feature_dim=feature_dim,
                      decoder_depth=int(decoder_depth / 2),
                      decoder_width=decoder_width).to(device)

    print("Input patch size:", patch_size)
    print("Number of parameters, G", numel(generator, only_trainable=True))

    generator.train()
    gen_out = generator(x)

    max_memory_reserved = torch.cuda.max_memory_reserved()
    print("Maximum memory reserved: %0.3f Gb / %0.3f Gb" % (max_memory_reserved / 10 ** 9, total_gpu_mem))

    print(gen_out.shape)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(x[0, 0, :, :, patch_size // 2].cpu().numpy(), cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(gen_out[0, 0, :, :, patch_size // (2 * up_factor)].detach().cpu().numpy(), cmap='gray')
    plt.show()