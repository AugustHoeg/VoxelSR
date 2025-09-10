import time
import logging
from functools import partial
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init
import torch.utils.checkpoint as checkpoint
from torch import Tensor, nn

from models.models_3D import SRBlock3D
from utils.utils_3D_image import numel


logger = logging.getLogger("dinov3")


def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None) -> None:
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.

    Source: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # (N, C, D, H, W) -> (N, D, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv3d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv3d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)


class BasicLayer(nn.Module):

    r""" A basic ConvNeXt layer. We can modify this to implement other types of layers.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop_path (float | list[float]): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, depth, dp_rates, layer_scale_init_value):
        super().__init__()

        self.blocks = nn.ModuleList(
            Block(dim, dp_rates[i], layer_scale_init_value) for i in range(depth)
        )

        #self.cab_blocks = nn.ModuleList(
        #    CAB(dim, compress_ratio=4, squeeze_factor=16) for i in range(depth)
        #)

    def forward(self, x):
        z = self.blocks[0](x)
        for blk in self.blocks[1:]:
            z = blk(z)

        return z + x


class TransitionLayer(nn.Module):
    r""" Transition layer between two stages
    Args:
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.layer = nn.Sequential(
            #LayerNorm(in_dim, eps=1e-6, data_format="channels_first"),
            nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True),
        )
    def forward(self, x):
        return self.layer(x)


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).

    Source: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


class ConvNeXtSR(nn.Module):
    r"""
    Code adapted from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.pyConvNeXt

    A PyTorch impl of : `A ConvNet for the 2020s`  -
        https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        patch_size (int | None): Pseudo patch size. Used to resize feature maps to those of a ViT with a given patch size. If None, no resizing is performed
    """

    def __init__(
        self,
        # original ConvNeXt arguments
        in_chans: int = 3,
        depths: List[int] = [3, 3, 9, 3],
        dims: List[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        # DINO arguments
        patch_size: int | None = None,
        # August SR arguments
        up_factor: int = 2,
        use_checkpoint: bool = False,
        upsample_method: str = "nearest",
        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        # ==== ConvNeXt's original init =====
        # self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        # stem = nn.Sequential(
        #     nn.Conv3d(in_chans, dims[0], kernel_size=4, stride=4),
        #     LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        # )
        # self.downsample_layers.append(stem)
        # for i in range(1):
        #     downsample_layer = nn.Sequential(
        #         LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
        #         nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
        #     )
        #     self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x for x in np.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(len(depths)):
            dim_ = dims[i + 1] if i < len(depths) - 1 else dims[0]
            dp = dp_rates[cur : cur + depths[i]]
            stage = nn.Sequential(
                *[
                    BasicLayer(dims[i], depths[i], dp, layer_scale_init_value),
                    # CAB(dims[i]),
                    TransitionLayer(dims[i], dim_)
                ]
            )

            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        # ==== End of ConvNeXt's original init =====

        # ==== DINO adaptation ====
        # self.head = nn.Identity()  # remove classification head
        # self.embed_dim = dims[-1]
        # self.embed_dims = dims  # per layer dimensions
        # self.n_blocks = len(self.downsample_layers)  # 4
        # self.chunked_blocks = False
        # self.n_storage_tokens = 0  # no registers
        #
        # self.norms = nn.ModuleList([nn.Identity() for i in range(3)])
        # self.norms.append(self.norm)
        #
        # self.patch_size = patch_size
        # self.input_pad_size = 4  # first convolution with kernel_size = 4, stride = 4

        # ==== SR init =====
        self.up_factor = up_factor
        self.upsample_method = upsample_method
        self.use_checkpoint = use_checkpoint

        # Shallow feature extraction block (SFE)
        self.sfe_blk = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=3, stride=1, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )

        # Transition layers between stages
        # self.transition_stages = nn.ModuleList(
        #     TransitionLayer(dims[i] + dims[i]*depths[i], dims[i + 1]) for i in range(len(depths) - 1)
        # )
        # self.transition_stages.append(
        #     TransitionLayer(dims[-1] + dims[-1]*depths[-1], dims[0])  # to match dims for long skip connection
        # )

        # Upsampling
        if self.up_factor == 2:
            self.SR0 = SRBlock3D(dims[0], dims[0], k_size=6, pad=2, upsample_method=upsample_method, upscale_factor=2)
        elif self.up_factor == 3:
            self.SR0 = SRBlock3D(dims[0], dims[0], k_size=6, pad=2, upsample_method=upsample_method, upscale_factor=3)
        elif self.up_factor == 4:
            self.SR0 = SRBlock3D(dims[0], dims[0], k_size=6, pad=2, upsample_method=upsample_method, upscale_factor=2)
            self.SR1 = SRBlock3D(dims[0], dims[0], k_size=6, pad=2, upsample_method=upsample_method, upscale_factor=2)

        # reconstruction
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.HRconv = nn.Conv3d(dims[0], dims[0], 3, 1, 1, bias=True)

        self.conv_last = nn.Sequential(
            nn.Conv3d(in_channels=dims[0], out_channels=in_chans, kernel_size=3, stride=1, padding=1, bias=True)
        )

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.LayerNorm):
            module.reset_parameters()
        if isinstance(module, LayerNorm):
            module.weight = nn.Parameter(torch.ones(module.normalized_shape))
            module.bias = nn.Parameter(torch.zeros(module.normalized_shape))
        if isinstance(module, (nn.Conv3d, nn.Linear)):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.constant_(module.bias, 0)

    def forward_features(self, x: Tensor | List[Tensor], masks: Optional[Tensor] = None) -> List[Dict[str, Tensor]]:
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])[0]
        else:
            return self.forward_features_list(x, masks)

    def forward(self, x):

        #d, h, w = x.shape[-3:]
        #x = torch.flatten(x, 2).transpose(1, 2)  # tokenize (N, C, D', H', W') -> (N, D'*H'*W', C)

        # shallow feature extraction
        x = self.sfe_blk(x)

        # deep feature extraction
        z = x
        for i in range(len(self.stages)):
            if i % 2 == 0 and self.use_checkpoint:
                z = checkpoint.checkpoint(self.stages[i], z)
            else:
                z = self.stages[i](z)

        # long skip-connection
        z = z + x

        # Upsampling
        if self.up_factor == 2:
            z = self.SR0(z)
        elif self.up_factor == 3:
            z = self.SR0(z)
        elif self.up_factor == 4:
            z = self.SR0(z)
            z = self.SR1(z)

        # reconstruction
        out = self.conv_last(self.lrelu(self.HRconv(z)))

        return out


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / 10 ** 9 if torch.cuda.is_available() else 0

    in_chans = 1
    up_factor = 2
    patch_size = 32

    net = ConvNeXtSR(depths=[3, 3, 3, 3],
                     dims=[96, 192, 384, 768],  # [96, 192, 288, 384, 768],
                     in_chans=in_chans,
                     drop_path_rate=0.1,
                     layer_scale_init_value=1e-6,
                     up_factor=up_factor,
                     upsample_method="nearest",
                     use_checkpoint=True).to(device)

    #net.init_weights()

    print("Number of parameters", numel(net, only_trainable=True))

    net.train()  # inference mode

    # Create a random input: batch size 1, 3 channels, 224x224 image
    x = torch.randn(1, in_chans, patch_size, patch_size, patch_size).to(device)

    x_hr = torch.randn((1, 1, up_factor * patch_size,
                        up_factor * patch_size,
                        up_factor * patch_size)).cuda()

    loss_func = nn.MSELoss()

    # Forward pass
    start = time.time()
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        out = net(x)
        loss = loss_func(out, x_hr)
    stop = time.time()
    print("Time elapsed:", stop - start)

    loss.backward()

    print("Output shape:", out.shape)

    max_memory_reserved = torch.cuda.max_memory_reserved()
    print("Maximum memory reserved: %0.3f Gb / %0.3f Gb" % (max_memory_reserved / 10 ** 9, total_gpu_mem))

    print("Done")

