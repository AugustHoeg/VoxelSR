# --------------------------------------------------------
# Swin Transformer V2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import os
import time
from datetime import datetime

import torch
import monai
from torchvision.utils import make_grid, save_image
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, to_3tuple, trunc_normal_
from timm.layers import PatchEmbed
import numpy as np
from torchvision.models.video.mvit import PositionalEncoding
from models.models_3D import SRBlock3D, ICNR3D, FusedMBConv
from models.SwinT.AugustSwin3D_clean import AugustDRCTBlock, AugustSwinV2Layer

from monai.networks.nets.swin_unetr import BasicLayer as monaibasiclayer

from utils.utils_3D_image import numel

import matplotlib.pyplot as plt

#from fft_conv_pytorch import FFTConv3d

# import deepspeed

def modulate(x, shift, scale):
    '''
    From DiT code github: https://github.com/facebookresearch/DiT/blob/main/models.py
    :param x:
    :param shift:
    :param scale:
    :return:
    '''
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def compute_mask(x_dims, window_size, shift_size):
    '''
    Computing region masks based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

    :param x_dims:
    :param ct_dims:
    :param window_size:
    :param ct_size:
    :param shift_size:
    :param ct_shift:
    :param device:
    :return:
    '''

    # Compute img mask for local tokens
    cnt = 0
    d, h, w = x_dims
    img_mask = torch.zeros((1, d, h, w, 1))  # removed device=device
    for d in slice(-window_size), slice(-window_size, -shift_size), slice(-shift_size, None):
        for h in slice(-window_size), slice(-window_size, -shift_size), slice(-shift_size, None):
            for w in slice(-window_size), slice(-window_size, -shift_size), slice(-shift_size, None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition3D(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


def window_partition3D(x, window_size):
    """
    Args:
        x: (B, Hp, Wp, Dp, C)
        window_size (int): window size

    Returns:
        windows: (nW, nP, C)
    """
    # B, C, H, W, D = x.shape
    # x = x.view(B, C, H // window_size, window_size, W // window_size, window_size, D // window_size, window_size)
    # windows = x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(-1, window_size * window_size * window_size, C)
    # return windows

    # B, C, H, W, D = x.shape
    # x = x.view(B, C, H // window_size, window_size, W // window_size, window_size, D // window_size, window_size)
    # windows = x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(-1, window_size * window_size * window_size, C)
    # return windows

    # FasterViT
    # B, C, H, W = x.shape
    # x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    # windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size * window_size, C)

    B, Hp, Wp, Dp, C = x.shape
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, Dp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size**3, C)  # Implementation used in Monai SwinUNETR
    #windows = x.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(-1, window_size * window_size * window_size, C)  # Works but own implementation
    return windows


def window_reverse3D(windows, window_size, dims):
    """
    Args:
        windows: (num_windows * B, window_size, window_size, window_size, C)
                 (4*4*4*B, 7, 7, 7, C)
        window_size (int): window size (default: 7)
        patch_size (int): patch size (default 4)
        H (int): Height of image (patch-wise)
        W (int): Width of image (patch-wise)
        D (int): Depth of images (patch-wise)

    Returns:
        x: (B, H, W, D, C)
    """
    B, H, W, D = dims
    nW, nP, C = windows.shape

    # Implementation used in Monai SwinUNETR
    x = windows.view(-1, window_size, window_size, window_size, C)
    x = x.view(B, H // window_size, W // window_size, D // window_size, window_size, window_size, window_size, C)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, H, W, D, C)

    # Works but own implementation
    #x = windows.view(-1, window_size, window_size, window_size, H // window_size, W // window_size, D // window_size, C).permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(-1, H, W, D, C)
    return x

    # x = windows.view(B, H // window_size, W // window_size, D // window_size, window_size, window_size, window_size, -1)
    # x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(B, windows.shape[2], H, W, D)

    # Get B from 8*8*B
    B = int(windows.shape[0] / (Hp * Wp * Dp / window_size / window_size / window_size))
    # B = int(windows.shape[0] / patch_size**3)

    # Convert to (B, 4, 4, 4, 7, 7, 7, C)
    x = windows.view(B, Hp // window_size, Wp // window_size, Dp // window_size, window_size, window_size, window_size,
                     -1)

    # Convert to (B, 4, 7, 4, 7, 4, 7, C)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()

    # Convert to (B, H, W, D, C)
    x = x.view(B, Hp, Wp, Dp, -1)

    return x


class TokenInitFromPatchEmbed3D(nn.Module):

    def __init__(self,
                 ct_size,
                 input_resolution,
                 patch_size,
                 window_size,
                 ct_embed_dim=1,
                 ct_pool_method="avg",
                 act_layer=nn.GELU,
                 zero_init=False):
        """
        Motivation: In FasterViT, carrier tokens are initialized by merging local tokens within each window to L^(2c) CTs,
        usually 4. However, CTs are generated after the local tokens have parsed through after a several convolutional
        layers. Thus, CTs can be interpreted as feature summaries rather than image summaries.

        In TokenInitAugust3D, CTs are generated after 1 initial convolution of the whole 3D image. This means that these
        CTs will most likely contain summaries of very low-level features.

        Ideally, we would also generate CTs after convolutional processing however, this is impractical due to the
        high memory consumption of convolutional layers used for super-resolution. Still, to make the implementation
        more in-line with FasterViT, this class initializes CTs based on the local patch embeddings instead of the 3D
        input image (after 1 conv), which should contain more higher-level features.

        Args:
            ct_size: Number of carrier tokens for each window.
            input_resolution: input image resolution.
            patch_size: patch size.
            window_size: window size.
            ct_embed_dim: spatial dimension of carrier token for each local window
            pool_method: pooling method used to create carrier tokens
        """
        super().__init__()

        # self.ct_size = ct_size
        self.ct_size = ct_size
        self.input_resolution = input_resolution
        self.ct_embed_dim = ct_embed_dim
        self.patch_size = patch_size
        self.window_size = window_size
        self.ct_pool_method = ct_pool_method
        self.act_layer = act_layer()

        self.size_p = self.input_resolution // self.patch_size

        #self.conv_block = nn.Sequential(
        #    nn.Conv3d(ct_embed_dim, ct_embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
        #    self.act_layer,
        #    nn.Conv3d(ct_embed_dim, ct_embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
        #    self.act_layer
        #)

        #if zero_init:
        #    # Zero initialize convolutions
        #    for layer in self.conv_block:
        #        if layer.__class__.__name__ == "Conv3d":
        #            nn.init.constant_(layer.weight, val=0)

        kernel_size = int(self.window_size / self.ct_size)

        if ct_pool_method == "avg":
            self.pool = nn.AvgPool3d(kernel_size=kernel_size, stride=kernel_size)
        elif ct_pool_method == "max":
            self.pool = nn.MaxPool3d(kernel_size=kernel_size, stride=kernel_size)
        elif ct_pool_method == "conv":
            self.pool = nn.Conv3d(in_channels=ct_embed_dim, out_channels=ct_embed_dim, kernel_size=kernel_size,
                                  stride=kernel_size)
        to_global_feature = nn.Sequential()
        # to_global_feature.add_module('proj_ct', self.proj_ct)
        to_global_feature.add_module('pool', self.pool)
        self.to_global_feature = to_global_feature

        # self.mlp = Mlp(128, hidden_features=256, out_features=128, act_layer=act_layer, drop=0.)

    def forward(self, x):

        '''
        :param x: embedded patch tokens
        :return: carrier tokens
        '''
        if len(x.shape) > 3:
            x_patches = x.permute(0, 4, 1, 2, 3).contiguous()  # B, C, H', W', D'
            #x_patches = self.conv_block(x_patches)
            ct = self.to_global_feature(x_patches)
            # B, C, Hct, Wct, Dct = ct.shape
            # ct = ct.permute(0, 2, 3, 4, 1).view(B, -1, C) # original
            # ct = ct.view(B, C, Hct//self.ct_size, self.ct_size, Wct//self.ct_size, self.ct_size, Dct//self.ct_size, self.ct_size).permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous().view(-1, Hct * Wct * Dct, C)  # works better
            ct = ct.flatten(2).transpose(1, 2)  # same as original
            #ct = ct.permute(0, 2, 3, 4, 1).contiguous() # same as previous
            #ct = ct.view(B, 3, 2, 3, 2, 3, 2, C).permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, 6 * 6 * 6, C) # same as previous
        else:
            B, T, C = x.shape
            x_patches = x.view(B, self.size_p, self.size_p, self.size_p, C).permute(0, 4, 1, 2, 3)  # B, C, H', W', D'
            #x_patches = self.conv_block(x_patches)
            ct = self.to_global_feature(x_patches)
            ct = ct.permute(0, 2, 3, 4, 1).view(1, -1, C)
            # ct = self.mlp(ct)

        return ct


class PatchEmbed3D(nn.Module):
    """
    Modified code from FasterViT
    """

    def __init__(self, in_channels=1, dim=96, patch_size=4, method="proj", act_layer=nn.GELU, out_format="image"):
        """
        Args:
            in_chans: number of input channels.
            dim: feature size dimension.
        """
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim
        self.patch_size = patch_size
        self.method = method
        self.act_layer = act_layer()
        self.out_format = out_format

        if patch_size < 3:
            pad = 0
        else:
            pad = 1

        if self.method == "proj":
            self.proj = nn.Sequential(
                nn.Conv3d(in_channels, dim, kernel_size=patch_size, stride=patch_size, padding=pad, bias=False),
            )
            # self.proj = nn.Sequential(
            #     nn.Conv3d(in_channels, dim, kernel_size=patch_size, stride=patch_size, padding=pad, bias=False),
            #     # nn.BatchNorm3d(dim, eps=1e-4),
            #     self.act_layer,
            #     nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            #     # nn.BatchNorm3d(dim, eps=1e-4),
            #     self.act_layer
            # )
        elif self.method == "mlp":
            in_features = self.in_channels * self.patch_size ** 3
            self.mlp = Mlp(in_features, hidden_features=in_features, out_features=self.dim, act_layer=nn.GELU, drop=0.)
            # self.fc1 = nn.Linear(self.patch_size * self.patch_size * self.patch_size, self.dim)
            # self.act1 = nn.GELU()
        elif self.method == "mlp_v2":
            in_features = self.in_channels * self.patch_size ** 3
            self.mlp = Mlp(in_features, hidden_features=4*in_features, out_features=dim, act_layer=nn.GELU, drop=0.)

    def forward(self, x):
        # (B, C, H, W, D) -> (C, num_windows, emd_dim)
        if self.method == "proj":
            x = self.proj(x)
            if self.out_format == "tokens":
                x = x.flatten(2).transpose(1, 2)  # used in swin transformer
            elif self.out_format == "image":
                x = x.permute(0, 2, 3, 4, 1).contiguous()
        elif self.method == "mlp":
            B, C, H, W, D = x.shape
            x = x.view(B, C, self.patch_size, H // self.patch_size, self.patch_size, W // self.patch_size,
                       self.patch_size, D // self.patch_size)
            x = x.permute(0, 1, 2, 4, 6, 3, 5, 7).contiguous()
            x = x.view(B, -1, H // self.patch_size * W // self.patch_size * D // self.patch_size).permute(0, 2, 1)
            # x = self.act1(self.fc1(x))
            x = self.mlp(x)
        elif self.method == "mlp_v2":
            x = x.view(1, 128, 2, 24, 2, 24, 2, 24).permute(0, 1, 2, 4, 6, 3, 5, 7).contiguous()
            x = x.view(1, 128 * 8, 24, 24, 24)
            x = self.mlp(x.permute(0, 2, 3, 4, 1))
        else:
            x = x.flatten(2).transpose(1, 2)  # used in SwinIR and Superformer, simply flatten instead of project
        return x


class WindowCrossAttention3D(nn.Module):
    r""" SwinV2 Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Modified code from official SwinTransformer V2 // August

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim, window_size, num_heads, q_bias=True, kv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.,
                 pretrained_window_size=[0, 0, 0]):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        #### changed for cross-attention
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=q_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=kv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if q_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None

        if kv_bias:
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.v_bias = None
        self.softmax = nn.Softmax(dim=-1)
        ####

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        '''
        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(3, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_d = torch.arange(-(self.window_size[2] - 1), self.window_size[2], dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w, relative_coords_d])).permute(1, 2, 3, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        #relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2


        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
            relative_coords_table[:, :, :, 2] /= (pretrained_window_size[2] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
            relative_coords_table[:, :, :, 2] /= (self.window_size[2] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_d = torch.arange(self.window_size[2])
        # coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        # coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        # relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        # relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        # relative_coords[:, :, 1] += self.window_size[1] - 1
        # relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        # relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # self.register_buffer("relative_position_index", relative_position_index)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_d]))  # 3, Wh, Ww, Wd
        coords_flatten = torch.flatten(coords, 1)  # 3, Wh*Ww*Wd
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wh*Ww*Wd, Wh*Ww*Wd
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww*Wd, Wh*Ww*Wd, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        '''

    def forward(self, x1, x2):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            ct: carrier tokens with shape of (num_windows*B, ct_per_window, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """

        B1, N1, C1 = x1.shape
        B2, N2, C2 = x2.shape

        # Compute q from x1
        q = F.linear(input=x1, weight=self.q.weight, bias=self.q_bias)
        q = q.reshape(B1, -1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3)

        kv_bias = None
        if self.v_bias is not None:
            kv_bias = torch.cat((torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))

        # Compute k and v from x2
        kv = F.linear(input=x2, weight=self.kv.weight, bias=kv_bias)
        kv = kv.reshape(B2, -1, 2, self.num_heads, C2 // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

        # cosine cross-attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01).cuda())).exp()
        attn = attn * logit_scale

        relative_position_bias = 0  # TODO: Add relative position bias or other positional encoding
        attn = attn + relative_position_bias

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B1, N1, C1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class WindowAttention3D(nn.Module):
    r""" SwinV2 Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Modified code from official SwinTransformer V2 // August

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0, 0]):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(3, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_d = torch.arange(-(self.window_size[2] - 1), self.window_size[2], dtype=torch.float32)
        relative_coords_table = (torch.stack(torch.meshgrid([relative_coords_h,
                                                             relative_coords_w,
                                                             relative_coords_d])).permute(1, 2, 3, 0).contiguous().unsqueeze(0))  # 1, 2*Wh-1, 2*Ww-1, 2
        # relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2

        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
            relative_coords_table[:, :, :, 2] /= (pretrained_window_size[2] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
            relative_coords_table[:, :, :, 2] /= (self.window_size[2] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        if False:
            ct_size = 2
            # get relative_coords_table
            relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
            relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
            relative_coords_d = torch.arange(-(self.window_size[2] - 1), self.window_size[2], dtype=torch.float32)
            relative_coords_h_ct = torch.arange(-(ct_size - 1), ct_size, dtype=torch.float32)
            relative_coords_w_ct = torch.arange(-(ct_size - 1), ct_size, dtype=torch.float32)
            relative_coords_d_ct = torch.arange(-(ct_size - 1), ct_size, dtype=torch.float32)
            relative_coords_h = torch.cat([relative_coords_h, relative_coords_h_ct])
            relative_coords_w = torch.cat([relative_coords_w, relative_coords_w_ct])
            relative_coords_d = torch.cat([relative_coords_d, relative_coords_d_ct])
            relative_coords_table = (torch.stack(torch.meshgrid([relative_coords_h,
                                                                 relative_coords_w,
                                                                 relative_coords_d])).permute(1, 2, 3, 0).contiguous().unsqueeze(0))  # 1, 2*Wh-1, 2*Ww-1, 2
            # relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2

            if pretrained_window_size[0] > 0:
                relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
                relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
                relative_coords_table[:, :, :, 2] /= (pretrained_window_size[2] - 1)
            else:
                relative_coords_table[:15, :15, :15, 0] /= (self.window_size[0] - 1)
                relative_coords_table[:15, :15, :15, 1] /= (self.window_size[1] - 1)
                relative_coords_table[:15, :15, :15, 2] /= (self.window_size[2] - 1)
                relative_coords_table[15:18, 15:18, 15:18, 0] /= (ct_size - 1)
                relative_coords_table[15:18, 15:18, 15:18, 1] /= (ct_size - 1)
                relative_coords_table[15:18, 15:18, 15:18, 2] /= (ct_size - 1)
            relative_coords_table *= 8  # normalize to -8, 8
            relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
                torch.abs(relative_coords_table) + 1.0) / np.log2(8)

            self.register_buffer("relative_coords_table", relative_coords_table)

            # Testing adding CTs to relative position bias
            coords_h = torch.arange(ct_size)
            coords_w = torch.arange(ct_size)
            coords_d = torch.arange(ct_size)
            coords_ct = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))  # 3, Wh, Ww, Wd
            coords_flatten_ct = torch.flatten(coords_ct, 1)  # 3, Wh*Ww*Wd

            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords_d = torch.arange(self.window_size[2])

            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))  # 3, Wh, Ww, Wd
            coords_flatten = torch.flatten(coords, 1)  # 3, Wh*Ww*Wd

            # Concat local and ct coords
            coords_flatten = torch.cat([coords_flatten, coords_flatten_ct], dim=1)

            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wh*Ww*Wd, Wh*Ww*Wd
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww*Wd, Wh*Ww*Wd, 3
            relative_coords[:self.window_size[0]**3, :self.window_size[0]**3, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:self.window_size[1]**3, :self.window_size[1]**3, 1] += self.window_size[1] - 1
            relative_coords[:self.window_size[2]**3, :self.window_size[2]**3, 2] += self.window_size[2] - 1
            relative_coords[:self.window_size[1]**3, :self.window_size[1]**3, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:self.window_size[2]**3, :self.window_size[2]**3, 1] *= 2 * self.window_size[2] - 1

            relative_coords[-ct_size**3:, -ct_size**3:, 0] += ct_size - 1  # shift to start from 0
            relative_coords[-ct_size**3:, -ct_size**3:, 1] += ct_size - 1
            relative_coords[-ct_size**3:, -ct_size**3:, 2] += ct_size - 1
            relative_coords[-ct_size**3:, -ct_size**3:, 0] *= (2 * ct_size - 1) * (2 * ct_size - 1)
            relative_coords[-ct_size**3:, -ct_size**3:, 1] *= 2 * ct_size - 1

            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)
        else:
            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords_d = torch.arange(self.window_size[2])

            #coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_d]))  # 3, Wh, Ww, Wd
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))  # 3, Wh, Ww, Wd
            coords_flatten = torch.flatten(coords, 1)  # 3, Wh*Ww*Wd
            #coords_flatten = torch.cat([coords_flatten, coords_flatten_ct], dim=1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wh*Ww*Wd, Wh*Ww*Wd
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww*Wd, Wh*Ww*Wd, 3
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, ct=None, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            ct: carrier tokens with shape of (num_windows*B, ct_per_window, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        if ct is not None:
            Nct = ct.shape[1]
            # Concatenate x and CTs
            x = torch.cat([x, ct], dim=1)
        else:
            Nct = 0

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))

        # x has shape [num_windows, num_patches_per_window, emb_dim]
        # The qkv for the first window can be calculated using "F.linear(input=x[0], weight=self.qkv.weight, bias=qkv_bias)"
        # if we want to do this sequentially.
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N + Nct, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        # q, k, v have the shape [num_windows, num_heads, num_patches_per_window, emb_dim/num_heads]
        # The attention for each window can be found using "(F.normalize(q[0], dim=-1) @ F.normalize(k[0], dim=-1).transpose(-2, -1))"
        # if we want to calculate attention sequentially within each window
        if False:  # Sequential attention
            a = (F.normalize(q[0:9], dim=-1) @ F.normalize(k[0:9], dim=-1).transpose(-2, -1))
            b = (F.normalize(q[9:18], dim=-1) @ F.normalize(k[9:18], dim=-1).transpose(-2, -1))
            c = (F.normalize(q[18:27], dim=-1) @ F.normalize(k[18:27], dim=-1).transpose(-2, -1))
            attn = torch.cat([a, b, c], dim=0)
        else:
            attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01).cuda())).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        #relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        # TODO: Fix relative position bias here - because of added CTs the relative_position_bias does not match the dimensions of the attention
        # TODO: score matrix. Then position biases should only be added to the window tokens of x and not the CTs
        # attn = attn + relative_position_bias.unsqueeze(0)

        if ct is not None:
            # zero pad relative_position_bias to same size as attention score matrix
            if relative_position_bias.shape != attn.shape[1:]:
                relative_position_bias = F.pad(relative_position_bias, (0, Nct, 0, Nct))
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            #attn = attn.view(B_ // nW, nW, self.num_heads, N + Nct, N + Nct) + mask[:, :N, :N].unsqueeze(1).unsqueeze(0) # For use with no CTs
            attn = attn.view(B_ // nW, nW, self.num_heads, N + Nct, N + Nct) + mask[:, :N + Nct, :N + Nct].unsqueeze(1).unsqueeze(0) # Use this when CTs are enabled
            attn = attn.view(-1, self.num_heads, N + Nct, N + Nct)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N + Nct, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchPositionalBias1D(nn.Module):

    def __init__(self,
                 dim,
                 rank=3,
                 seq_length=4,
                 level=0,
                 context_sizes=[96, 64, 32],
                 conv=False,
                 input_format="windows",
                 window_size=8,
                 method="window_relative"):

        super().__init__()
        self.rank = rank
        self.level = level
        self.conv = conv
        self.dim = dim
        self.input_format = input_format
        self.window_size = window_size
        self.seq_length = seq_length
        self.method = method

        # Use Linear for implementation of MLP as in FasterViT
        self.cpb_mlp = nn.Sequential(nn.Linear(self.rank, 512, bias=True),
                                     nn.ReLU(),
                                     nn.Linear(512, dim, bias=False))

        # NOTE: fixed assumption of level_ratio = 2
        dist = torch.tensor(context_sizes) / max(context_sizes)
        self.pos_scale = nn.Parameter(dist[self.level], requires_grad=True)

        self.grid_exists = False  # Flag to signal if relative biases have already been computed
        self.pos_emb = None  # used for storing positional embedding
        self.deploy = False  # Flag for deploying this layer
        relative_bias = torch.zeros(1, seq_length, dim)  # Placeholder for relative positional biases
        self.register_buffer("relative_bias", relative_bias)

    def forward(self, input_tensor):
        # seq_length = input_tensor.shape[1] if not self.conv else input_tensor.shape[2]
        if self.deploy:
            return input_tensor + self.relative_bias
        else:
            self.grid_exists = False
        if not self.grid_exists:
            self.grid_exists = True
            if self.rank == 1:
                relative_coords_h = torch.arange(0, self.seq_length, device=input_tensor.device,
                                                 dtype=input_tensor.dtype)
                relative_coords_h -= self.seq_length // 2
                relative_coords_h /= (self.seq_length // 2)
                relative_coords_table = relative_coords_h
                self.pos_emb = self.cpb_mlp(relative_coords_table.unsqueeze(0).unsqueeze(2))
                self.relative_bias = self.pos_emb
            elif self.rank == 2:
                seq_length = int(self.seq_length ** 0.5)
                relative_coords_h = torch.arange(0, seq_length, device=input_tensor.device, dtype=input_tensor.dtype)
                relative_coords_w = torch.arange(0, seq_length, device=input_tensor.device, dtype=input_tensor.dtype)
                relative_coords_table = torch.stack(
                    torch.meshgrid([relative_coords_h, relative_coords_w])).contiguous().unsqueeze(0)
                relative_coords_table -= seq_length // 2
                relative_coords_table /= (seq_length // 2)
                if not self.conv:
                    self.pos_emb = self.cpb_mlp(relative_coords_table.flatten(2).transpose(1, 2))
                else:
                    self.pos_emb = self.cpb_mlp(relative_coords_table.flatten(2))
                self.relative_bias = self.pos_emb
            else:  # rank == 3
                # TODO: When this function is called for patch tokens, each token is given a unique positional embedding
                # which is what we want. However, there might be a problem when the same function is used for CTs since
                # we have multiple CTs for each attention window. These should have the same positional bias, but are
                # given different ones in the code below.
                # TODO: Testing here for even sequence lengths
                if self.method == "absolute":
                    seq_length = int(np.cbrt(self.seq_length))
                elif self.method == "window_relative":
                    seq_length = self.window_size
                rel_coords_h = torch.linspace(-1, 1, steps=seq_length, device=input_tensor.device) * self.pos_scale
                rel_coords_w = torch.linspace(-1, 1, steps=seq_length, device=input_tensor.device) * self.pos_scale
                rel_coords_d = torch.linspace(-1, 1, steps=seq_length, device=input_tensor.device) * self.pos_scale
                #rel_coords = rel_coords * self.pos_scale
                rel_coords_table = torch.stack(torch.meshgrid([rel_coords_h, rel_coords_w, rel_coords_d])).contiguous().unsqueeze(0)
                if not self.conv:
                    self.pos_emb = self.cpb_mlp(rel_coords_table.flatten(2).transpose(1, 2))
                else:
                    self.pos_emb = self.cpb_mlp(rel_coords_table.flatten(2))

                if self.input_format == "windows" and self.method == "absolute":
                    self.pos_emb = self.pos_emb.view(-1, seq_length, seq_length, seq_length, self.dim)
                    self.pos_emb = window_partition3D(self.pos_emb, self.window_size)

                self.relative_bias = self.pos_emb

                # seq_length = int(np.cbrt(seq_length))
                # relative_coords_h = torch.arange(0, seq_length, device=input_tensor.device, dtype=input_tensor.dtype)
                # relative_coords_w = torch.arange(0, seq_length, device=input_tensor.device, dtype=input_tensor.dtype)
                # relative_coords_d = torch.arange(0, seq_length, device=input_tensor.device, dtype=input_tensor.dtype)
                # relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w, relative_coords_d])).contiguous().unsqueeze(0)
                # relative_coords_table -= seq_length // 2
                # relative_coords_table /= (seq_length // 2)
                # if not self.conv:
                #     self.pos_emb = self.cpb_mlp(relative_coords_table.flatten(2).transpose(1, 2))
                # else:
                #     self.pos_emb = self.cpb_mlp(relative_coords_table.flatten(2))
                # self.relative_bias = self.pos_emb
        input_tensor = input_tensor + self.pos_emb
        return input_tensor


class PosEmbMLPSwinv1D(nn.Module):
    def __init__(self,
                 dim,
                 rank=2,
                 seq_length=4,
                 conv=False):
        super().__init__()
        self.rank = rank
        if not conv:
            self.cpb_mlp = nn.Sequential(nn.Linear(self.rank, 512, bias=True),
                                         nn.ReLU(),
                                         nn.Linear(512, dim, bias=False))
        else:
            self.cpb_mlp = nn.Sequential(nn.Conv1d(self.rank, 512, 1, bias=True),
                                         nn.ReLU(),
                                         nn.Conv1d(512, dim, 1, bias=False))
        self.grid_exists = False
        self.pos_emb = None
        self.deploy = False
        relative_bias = torch.zeros(1, seq_length, dim)
        self.register_buffer("relative_bias", relative_bias)
        self.conv = conv

    def switch_to_deploy(self):
        self.deploy = True

    def forward(self, input_tensor):
        seq_length = input_tensor.shape[1] if not self.conv else input_tensor.shape[2]
        if self.deploy:
            return input_tensor + self.relative_bias
        else:
            self.grid_exists = False
        if not self.grid_exists:
            self.grid_exists = True
            if self.rank == 1:
                relative_coords_h = torch.arange(0, seq_length, device=input_tensor.device, dtype=input_tensor.dtype)
                relative_coords_h -= seq_length // 2
                relative_coords_h /= (seq_length // 2)
                relative_coords_table = relative_coords_h
                self.pos_emb = self.cpb_mlp(relative_coords_table.unsqueeze(0).unsqueeze(2))
                self.relative_bias = self.pos_emb
            else:
                seq_length = int(seq_length ** 0.5)
                relative_coords_h = torch.arange(0, seq_length, device=input_tensor.device, dtype=input_tensor.dtype)
                relative_coords_w = torch.arange(0, seq_length, device=input_tensor.device, dtype=input_tensor.dtype)
                relative_coords_table = torch.stack(
                    torch.meshgrid([relative_coords_h, relative_coords_w])).contiguous().unsqueeze(0)
                relative_coords_table -= seq_length // 2
                relative_coords_table /= (seq_length // 2)
                if not self.conv:
                    self.pos_emb = self.cpb_mlp(relative_coords_table.flatten(2).transpose(1, 2))
                else:
                    self.pos_emb = self.cpb_mlp(relative_coords_table.flatten(2))
                self.relative_bias = self.pos_emb
        input_tensor = input_tensor + self.pos_emb
        return input_tensor


class AugustBlock(nn.Module):

    def __init__(self, blk_layers, level, embed_dims, ct_embed_dims, ct_size, context_sizes, patch_sizes, sizes_p, num_heads,
                 attn_window_sizes, enable_shift, mlp_ratio, qkv_bias, drop, attn_drop, drop_path,
                 act_layer, norm_layer, pretrained_window_size, layer_type, enable_ape_ct, enable_ape_x, enable_ct_rpb,
                 enable_conv_skip, patch_pe_method):
        super().__init__()

        self.level = level
        self.blk_layers = blk_layers
        self.context_sizes = context_sizes

        self.layer_type = layer_type
        self.enable_conv_skip = enable_conv_skip
        # Enable / disable shift
        self.shift_size = attn_window_sizes[level] // 2 if enable_shift else 0

        self.attn_window_size = attn_window_sizes[level]
        self.size_p = sizes_p[level]
        self.Hp, self.Wp, self.Dp = sizes_p[level]

        self.layers = nn.ModuleList()
        self.compress_layers = nn.ModuleList()
        self.ct_compress_layers = nn.ModuleList()

        self.ct_size = ct_size
        self.ct_resolution = ct_size*(self.size_p[0] // self.attn_window_size)
        self.nW = self.Hp // self.attn_window_size  # Should (ideally) be done for Hp, Wp, Dp // August
        self.sr_ratio = self.nW
        self.cr_window = ct_size

        drct_mlp_ratio_list = torch.ones(blk_layers[level])
        drct_mlp_ratio_list[0:3] = torch.tensor([4, 4, 2])[0:3]  # Sets MLP ratio list to [4, 4, 2, 1, 1, ..., 1]

        skip_embed_dim = 64  # gc in DRCT
        embed_dim = embed_dims[level]
        for layer_idx in range(blk_layers[level]):
            channel_dim = embed_dim + skip_embed_dim * layer_idx
            compress_dim = skip_embed_dim if layer_idx < blk_layers[level] - 1 else embed_dim
            embed_dims = [channel_dim, channel_dim, channel_dim]
            ct_embed_dims = [channel_dim, channel_dim, channel_dim]
            drct_mlp_ratio = drct_mlp_ratio_list[layer_idx]

            # TODO test skipping CT attention sometimes
            skip_ct_attn = False  #False if layer_idx == 0 else True

            # Use the CSM in the first AugustLayer at all levels apart from level 0
            init_CSM = True if (layer_idx == 0) and (level > 0) else False
            shift_size = 0 if layer_idx % 2 == 0 else self.shift_size  # use shifting window attn in every other layer
            self.layers.append(
                AugustDRCTLayerV2(level, embed_dims, ct_embed_dims, ct_size, context_sizes, patch_sizes, sizes_p, num_heads=num_heads,
                              window_sizes=attn_window_sizes, shift_size=shift_size, mlp_ratio=drct_mlp_ratio,
                              ct_mlp_ratio=drct_mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                              drop_path=drop_path[layer_idx],  # drop path here
                              act_layer=act_layer, norm_layer=norm_layer, pretrained_window_size=pretrained_window_size,
                              enable_ape_ct=enable_ape_ct, enable_ape_x=enable_ape_x, enable_ct_rpb=enable_ct_rpb,
                              patch_pe_method=patch_pe_method, init_CSM=init_CSM, skip_ct_attn=skip_ct_attn)
            )
            self.compress_layers.append(
                nn.Conv3d(channel_dim, compress_dim, kernel_size=1, stride=1, padding=0)
            )
            self.ct_compress_layers.append(
                nn.Conv3d(channel_dim, compress_dim, kernel_size=1, stride=1, padding=0)
            )

            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)  # TODO try with act_layer

        # Convolutional block after each transformer block
        # TODO: Try with a DenseNet or ResNet block here?
        # TODO: Spatial Attention or Channel Attention here?
        # TODO: Skip? Dilated Conv?
        if enable_conv_skip:
            self.x_conv_block = nn.Sequential(
                nn.Conv3d(in_channels=embed_dims[level], out_channels=embed_dims[level], kernel_size=3, stride=1, padding=1),
            )

            self.ct_conv_block = nn.Sequential(
                nn.Conv3d(in_channels=ct_embed_dims[level], out_channels=ct_embed_dims[level], kernel_size=3, stride=1, padding=1),
            )

    def to_image(self, feats):
        B, T, C = feats.shape
        dim = int(np.cbrt(T))
        return feats.view(B, dim, dim, dim, C).permute(0, 4, 1, 2, 3)

    def to_tokens(self, feats):
        B, C, H, W, D = feats.shape
        T = H * W * D
        return feats.permute(0, 2, 3, 4, 1).view(B, T, C)

    def forward(self, x, ct, prev_x=None, prev_ct=None):

        x_input = x
        ct_input = ct
        next_x = x
        next_ct = ct

        #B, Tct, Cct = ct.shape
        B = int(x.shape[0] / (self.Hp * self.Wp * self.Dp / self.attn_window_size / self.attn_window_size / self.attn_window_size))
        x_dims = (B, self.size_p[0], self.size_p[1], self.size_p[2])

        ###### DRCT block structure with CTs ######
        for i in range(self.blk_layers[self.level] - 1):
            prev_x = None if i > 0 else prev_x
            prev_ct = None if i > 0 else prev_ct

            # Transformer layer
            x, ct = self.layers[i](next_x, next_ct, prev_x, prev_ct)

            # Image space convolution
            x = self.act(self.compress_layers[i](window_reverse3D(x, window_size=self.attn_window_size, dims=x_dims).permute(0, 4, 1, 2, 3)))
            #ct = self.act(self.ct_compress_layers[i](ct.view(Bct, self.ct_resolution, self.ct_resolution, self.ct_resolution, -1).permute(0, 4, 1, 2, 3).contiguous()))

            # Partition back to windows/tokens
            x = window_partition3D(x.permute(0, 2, 3, 4, 1), window_size=self.attn_window_size)
            #ct = ct.flatten(2).transpose(1, 2)

            next_x = torch.cat([next_x, x], 2)
            #next_ct = torch.cat([next_ct, ct], 2)

        # Final layer
        x, ct = self.layers[-1](next_x, next_ct, prev_x, prev_ct)
        x = self.act(self.compress_layers[-1](window_reverse3D(x, window_size=self.attn_window_size, dims=x_dims).permute(0, 4, 1, 2, 3)))
        #ct = self.act(self.ct_compress_layers[-1](ct.view(Bct, self.ct_resolution, self.ct_resolution, self.ct_resolution, -1).permute(0, 4, 1, 2, 3).contiguous()))
        x = window_partition3D(x.permute(0, 2, 3, 4, 1), window_size=self.attn_window_size)
        #ct = ct.flatten(2).transpose(1, 2)

        x = 0.2 * x + x_input
        #ct = 0.2 * ct + ct_input

        # if self.enable_conv_skip:
        #     x_image = window_reverse3D(x, window_size=self.attn_window_size, dims=(layer.B, self.size_p[0], self.size_p[1], self.size_p[2])).permute(0, 4, 1, 2, 3)
        #     x_image = self.x_conv_block(x_image)
        #     x = window_partition3D(x_image.permute(0, 2, 3, 4, 1), window_size=self.attn_window_size)
        #
        #     #x = self.to_tokens(self.x_conv_block(self.to_image(x)))
        #     #ct = self.to_tokens(self.ct_conv_block(self.to_image(ct)))
        #
        # x = x_skip + x

        return x, ct


class AugustDRCTLayerV2(nn.Module):
    r""" August Transformer Layer V2.

    Similar to suggested by FasterViT, this version of AugustLayer keeps CTs in token format (B, T, C) while keeping
    local tokens in windowed format (nW, nP, C). Shifting of attention windows using torch.roll have been removed
    as it is assumed CTs will allow for global information transfer between attention windows.
    Furthermore, we parse the concatenated x_windows and ct_windows through the MLP layers at the end without
    reshaping to token format.
    All together, this enables us to reduce the number of memory altering operations.


    Args:
        embed_dim (int): Number of channels in the token embedding.
        size_p (tuple[int]): Height, Width and Depth of input volume in terms of patches.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self, level, embed_dims, ct_embed_dims, ct_size, context_sizes, patch_sizes, sizes_p, num_heads, window_sizes,
                 shift_size=0,
                 ct_mlp_ratio=4., mlp_ratio=4., qkv_bias=True, qk_scale=None, layer_scale=1., use_layer_scale=True,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 pretrained_window_size=0, enable_ape_ct=True, enable_ape_x=False, enable_ct_rpb=True, patch_pe_method="window_relative",
                 return_x_image=False, token_upsample_method="Monaipixelshuffle", init_CSM=False, skip_ct_attn=False):
        super().__init__()
        self.level = level
        self.embed_dim = embed_dims[level]
        self.ct_embed_dim = ct_embed_dims[level]
        self.prev_ct_embed_dim = ct_embed_dims[level - 1] if level > 0 else self.ct_embed_dim
        self.prev_embed_dim = embed_dims[level - 1] if level > 0 else self.embed_dim
        self.Hp, self.Wp, self.Dp = sizes_p[level]
        self.prev_Hp, self.prev_Wp, self.prev_Dp = sizes_p[level - 1] if level > 0 else sizes_p[level]
        self.context_sizes = context_sizes
        self.num_heads = num_heads
        self.window_size = window_sizes[level]
        self.prev_window_size = window_sizes[level - 1] if level > 0 else self.window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.return_x_image = return_x_image
        self.image_space_up_factor = patch_sizes[-1]  # upscaling factor required to reach dimensionality of input image

        # Use this flag to initialize the CSM module to merge current-level and previous-level local tokens / CTs
        self.init_CSM = init_CSM

        self.nW = self.Hp // self.window_size  # Should (ideally) be done for Hp, Wp, Dp // August
        self.sr_ratio = self.nW

        self.enable_ape_ct = enable_ape_ct
        self.enable_ape_x = enable_ape_x
        self.enable_ct_rpb = enable_ct_rpb
        self.patch_pe_method = patch_pe_method

        self.token_upsample_method = token_upsample_method

        ######### CT init #########
        self.skip_ct_attn = skip_ct_attn
        # number of carrier tokens per every window
        self.cr_tokens_per_window = ct_size ** 3 if self.sr_ratio > 1 else 0
        # total number of carrier tokens
        cr_tokens_total = self.cr_tokens_per_window * (self.sr_ratio ** 3)
        self.cr_window = ct_size
        self.ct_size = ct_size
        self.ct_resolution = self.cr_window*self.sr_ratio

        if self.enable_ape_x:
            ######### CT and local token 1D position bias ##########
            # FasterViT uses both 1D positional bias for CTs and local tokens as well as relative attention
            # bias by MLP proposed in Swin V2. For 1D bias, they remove log-scale to enable better flexibility with
            # different image sizes.
            self.pos_embed = PatchPositionalBias1D(self.embed_dim, rank=3, seq_length=self.Hp * self.Wp * self.Dp,
                                                   level=level, context_sizes=self.context_sizes, conv=False, input_format="windows",
                                                   window_size=self.window_size, method=patch_pe_method)  # For local tokens, method="absolute", "window_relative"

        ######### CSM init #########
        if self.init_CSM:
            self.csm_method = "crossattn"
            q_bias = True
            kv_bias = True

            ######### Multi-level patch merge #########
            self.prev_mlp_hidden_dim = self.prev_embed_dim
            self.mlp_down = Mlp(in_features=self.prev_embed_dim, hidden_features=self.prev_mlp_hidden_dim,
                                out_features=self.embed_dim, act_layer=act_layer, drop=drop)

            self.cross_attn_func = WindowCrossAttention3D(self.embed_dim, window_size=to_3tuple(self.window_size),
                                                          num_heads=num_heads, q_bias=True, kv_bias=True, qk_scale=None,
                                                          attn_drop=attn_drop, proj_drop=drop,
                                                          pretrained_window_size=to_3tuple(pretrained_window_size))
            self.cross_norm = norm_layer(self.embed_dim)

        ######### Image space upsampling #########
        # pre-conv with ICNR3D before pixelshuffle3D to image space
        if self.return_x_image:
            # self.pre_conv_image_space = nn.Conv3d(self.embed_dim, self.embed_dim * (self.image_space_up_factor ** 3),
            #                                       kernel_size=3, stride=1, padding=1)
            # weight = ICNR3D(self.pre_conv_image_space.weight, initializer=nn.init.normal_,
            #                 upscale_factor=self.image_space_up_factor, mean=0.0, std=0.02)
            # self.pre_conv_image_space.weight.data.copy_(weight)
            # self.act_image_space = act_layer()
            if self.image_space_up_factor > 1:
                token_upsampler = SRBlock3D(self.embed_dim,
                                            self.embed_dim,
                                            k_size=6, pad=2,
                                            upsample_method=self.token_upsample_method,
                                            upscale_factor=self.image_space_up_factor,
                                            use_checkpoint=False)
            else:
                token_upsampler = nn.Identity()

            self.image_upsampler = nn.Sequential(
                token_upsampler,
                #nn.ConvTranspose3d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2),
                # Conv3d, InstanceNorm and LeakyReLU inspired by SuperFormer
                # TODO test with and without Conv3d, InstanceNorm and activation
                nn.Conv3d(self.embed_dim, self.embed_dim, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(self.embed_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

        # Upsampler to transform CTs to image space such that CT information can be propagated into the image
        self.propagate_ct_to_image = False
        if self.propagate_ct_to_image:
            #self.upsampler = nn.Upsample(size=self.window_size, mode='nearest')
            self.upsampler = monai.networks.blocks.UpSample(3,
                                               in_channels=self.ct_embed_dim,
                                               out_channels=None,
                                               scale_factor=self.window_size//self.ct_size,
                                               kernel_size=None,
                                               size=None,
                                               mode="deconv",
                                               pre_conv='default',
                                               interp_mode="linear",
                                               align_corners=True,
                                               bias=True,
                                               apply_pad_pool=True)

        # Scaling factors for windowed attention mechanism (optional)
        self.gamma3 = nn.Parameter(layer_scale * torch.ones(self.embed_dim)) if use_layer_scale else 1
        self.gamma4 = nn.Parameter(layer_scale * torch.ones(self.embed_dim)) if use_layer_scale else 1

        ########## Swin V2 init ###########
        # if min(self.input_resolution) <= self.window_size:
        #     # if window size is larger than input resolution, we don't partition windows
        #     self.shift_size = 0
        #     self.window_size = min(self.input_resolution)
        # assert 0 <= self.shift_size < self.window_size, "shift_size must between zero and window_size"

        # Layer norm 1 - after attention is Swin V2 paper
        self.norm1 = norm_layer(self.embed_dim)

        # Swin V2 Window Attention
        self.attn_func = WindowAttention3D(self.embed_dim, window_size=to_3tuple(self.window_size), num_heads=num_heads,
                                           qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                                           pretrained_window_size=to_3tuple(pretrained_window_size))

        # Timm stochastic dropout
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Layer norm 2 - after MLP in Swin V2 paper
        self.norm2 = norm_layer(self.embed_dim)

        # MLP
        mlp_hidden_dim = int(self.embed_dim * mlp_ratio)
        self.mlp = Mlp(in_features=self.embed_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # Initialization of mask for 3D shifted-window attention with CTs
        if self.shift_size > 0:
            attn_mask = compute_mask(x_dims=(self.Hp, self.Wp, self.Dp),
                                     window_size=self.window_size, shift_size=self.shift_size)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x_windows, ct=None, prev_x=None, prev_ct=None):

        """
        Instructions for positional encoding:
        We can either:
        Add absolute positional encoding to all patch tokens based on their absolution location in the whole 3D scan.
        Then before attention we add log relative positional encoding to patch tokens inside each local window.
        or:
        Add absolute positional encoding to all patch tokens based on the absolute location of the windows they are in.
        This means that all tokens in the same local window have the same positional encoding.
        There will only be a fixed number of carrier tokens for each attention window, so we can add the same positional encoding
        to the carrier tokens.
        Before attention, we can then add log 3D relative positional encoding to patch tokens inside each local window.
        Idea: Since carrier tokens only hold meaning for each attention window, skip addding of 3D log relative positional
        encoding to these.

        :param x: Token embeddings of current level input patch
        :param ct: Carrier tokens of current level input patch
        :param prev_x: attended tokens from previous level patch
        :param prev_ct: attended carrier tokens from previous level patch
        :return: x, ct
        """

        ############ Compute Shift and Scale ############
        # This uses the Adaptive Layer Norm (adaLN-Zero) approach from DiT
        #c = 0  # Conditioning input
        #shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        #x = x + gate_msa.unsqueeze(1)  # post-scale
        #x_windows = modulate(x_windows, shift_msa, scale_msa)  # shift and scale
        # order: shift-n-scale, attention, post-scale, LayerNorm

        ############ Swin V2 attention mechanism here ############
        # number of windows, number of patches in each window, number of channels for each patch token
        # (nW, nP, nC)
        nW, nP, C = x_windows.shape
        # assert T == self.Hp * self.Wp * self.Dp, "input feature has wrong size"
        B = int(x_windows.shape[0] / (self.Hp * self.Wp * self.Dp / self.window_size / self.window_size / self.window_size))

        # positional bias for patch tokens - NOTE: windowed format
        # TODO: Positional encoding of patch tokens
        if self.enable_ape_x:
            x_windows = self.pos_embed(x_windows.view(B, -1, nP, C)).view(nW, nP, C)  # 1D positional bias by FasterViT

        # Cross-Attention of previous level patch features
        if prev_x is not None:
            if self.csm_method == "adaLN":  # Adaptive Layer Norm Zero (adaLN-Zero) conditioning from DiT paper.
                pass
            elif self.csm_method == "crossattn":
                prev_x_windows = self.mlp_down(prev_x)
                #prev_x_windows = prev_x
                #prev_ratio = int(np.cbrt(nW / prev_x_windows.shape[0]))
                if nW != prev_x_windows.shape[0]:  # Repartition previous-level patches using current-level window size
                    prev_x_windows = window_reverse3D(prev_x_windows, self.prev_window_size, dims=(B, self.prev_Hp, self.prev_Wp, self.prev_Dp))
                    prev_x_windows = window_partition3D(prev_x_windows, window_size=self.window_size)
                # Window cross-attention
                cross_attn_windows = self.cross_attn_func(x_windows, prev_x_windows)  # nW*B, window_size*window_size, C
                # Norm and skip of CTs (As done in DiT paper)
                x_windows = x_windows + self.cross_norm(cross_attn_windows)
            # TODO: Experiment with adding MLP after attention here

        # concatenate windowed carrier_tokens to the windowed tokens
        # TODO: Experiment with having ct_windows first vs. last in concatenation
        # TODO: As default, FasterViT concatenates CTs to the left of x_windows (ct_windows, x_windows)

        # Register skip connection w. CTs before attention
        shortcut = x_windows

        # Shifting windows for W-MSA (optional)
        if self.shift_size > 0:
            x = window_reverse3D(x_windows, self.window_size, dims=(B, self.Hp, self.Wp, self.Dp))
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(1, 2, 3))
            x_windows = window_partition3D(x, window_size=self.window_size)

        # W-MSA/SW-MSA
        #x_ct = torch.cat([x_windows, ct_windows], dim=1)
        x = self.attn_func(x_windows, None, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        # Skip splitting of attn_windows and instead parse x and ct through MLP in windowed format
        # attn_windows, attn_ct = attn_windows.split([self.window_size**3, self.cr_window**3], dim=1)

        # Shifting windows for W-MSA (optional)
        if self.shift_size > 0:
            x = window_reverse3D(x_windows, self.window_size, dims=(B, self.Hp, self.Wp, self.Dp))
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(1, 2, 3))
            x = window_partition3D(x, window_size=self.window_size)

        # Norm and skip connection after attention - NOTE: windowed format of shortcut
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN - NOTE: x_ct is parsed through MLP in windowed format
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        # TODO: Test with and without propagation of CTs unto image in EVERY AugustLayer.
        # ct_image_space = ct.transpose(1, 2).reshape(Bct * self.sr_ratio ** 3, C, self.cr_window, self.cr_window, self.cr_window)
        # x = x + (self.gamma1 * self.upsampler(ct_image_space).flatten(2).transpose(1, 2)).reshape(B, T, C)

        if self.return_x_image:
            # TODO: Test propagation of CT information unto image in every layer vs. only if return x in image_space
            # Propagate CT information unto x via upsampling
            # TODO: Test with and without propagation of CTs unto image.
            # window_reverse3D(ct, 2, dims=(B, 4, 4, 4)) -> (1, 4, 4, 4, 128)

            # activation before Pixelshuffle3D
            # x_image_space = self.act_image_space(x_image_space)

            # x_image_space = x.view(-1, self.Hp, self.Wp, self.Dp, C) # TODO: Maybe use .view instead of window_reverse3D
            x_image_space = window_reverse3D(x, self.window_size, dims=(B, self.Hp, self.Wp, self.Dp))  # TODO: Maybe use window_reverse3D instead of .view

            # Upsample tokens if needed
            x_image_space = self.image_upsampler(x_image_space.permute(0, 4, 1, 2, 3))

            # Return x_image_space as x and None as ct as x_image_space also contains information from ct
            return x_image_space, None
        else:
            return x, None

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops





class MTVNet_no_CT(nn.Module):
    r""" August Net.

    Args:
        embed_dim (int): Number of channels in the token embedding.
        size_p (tuple[int]): Height, Width and Depth of input volume in terms of patches.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self, input_size, up_factor, num_levels, context_sizes, num_blks, blk_layers, in_chans, shallow_feats, pre_up_feats,
                 ct_embed_dims, embed_dims, ct_size, ct_pool_method, patch_sizes, num_heads, attn_window_sizes, enable_shift=True,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0., token_upsample_method="Monaipixelshuffle", upsample_method="deconv_nn_resize",
                 use_checkpoint=False, layer_type="fastervit", enable_ape_ct=True, enable_ape_x=False, enable_ct_rpb=True,
                 enable_conv_skip=True, enable_long_skip=True, patch_pe_method="window_relative"):
        super().__init__()

        # Plot counter (very rudimentary)
        #self.plot_counter = 0
        #self.plot_interval = 200
        #self.output_image_dir = 'AugustNet_output_images'
        #os.makedirs(self.output_image_dir, exist_ok=True)

        self.use_checkpoint = use_checkpoint
        self.num_blks = num_blks
        self.blk_layers = blk_layers

        self.layer_type = layer_type
        self.enable_ape_ct = enable_ape_ct
        self.enable_ape_x = enable_ape_x
        self.enable_ct_rpb = enable_ct_rpb
        self.ct_pool_method = ct_pool_method
        self.enable_conv_skip = enable_conv_skip
        self.enable_long_skip = enable_long_skip
        self.patch_pe_method = patch_pe_method
        self.enable_shift = enable_shift
        self.token_upsample_method = token_upsample_method
        print("layer_type: ", layer_type)
        print("enable_ape_ct: ", enable_ape_ct)
        print("enable_ape_x: ", enable_ape_x)
        print("enable_ct_rpb: ", enable_ct_rpb)
        print("ct_pool_method: ", ct_pool_method)
        print("enable_conv_skip: ", enable_conv_skip)
        print("enable_long_skip: ", enable_long_skip)
        print("patch_pe_method: ", patch_pe_method)
        print("enable_shift: ", enable_shift)
        print("token_upsample_method: ", token_upsample_method)

        self.up_factor = up_factor
        self.upsample_method = upsample_method
        self.input_size = input_size
        self.num_levels = num_levels
        self.context_sizes = context_sizes
        #self.img_sizes = [(torch.tensor(input_size) // level_ratio ** (level)).tolist() for level in range(num_levels)]
        self.patch_sizes = patch_sizes
        self.sizes_p = [[context_size // patch_size, context_size // patch_size, context_size // patch_size] for context_size, patch_size in zip(self.context_sizes, patch_sizes)]
        self.attn_window_sizes = attn_window_sizes
        self.shallow_feats = shallow_feats
        self.pre_up_feats = pre_up_feats

        self.embed_dims = embed_dims
        if embed_dims is None:
            self.embed_dims = [patch_size ** 3 for patch_size in patch_sizes]

        self.ct_embed_dims = ct_embed_dims
        if ct_embed_dims is None:
            self.ct_embed_dims = [patch_size ** 3 for patch_size in
                                  patch_sizes]  # [256*level for level in range(num_levels, 0, -1)]

        ####### Testing SFE with standard Conv3d + act #######
        self.sfe_blks = nn.ModuleList()
        for level in range(num_levels):
            input_feats = 1 if level == 0 else shallow_feats[level - 1]
            shallow_feat = shallow_feats[level]
            if level == num_levels - 1:  # use 2 layer sfe in final level
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

        # 3D Patch embedding blocks
        self.patch_embedding_blks = nn.ModuleList()
        for level in range(num_levels):
            self.patch_embedding_blks.append(
                PatchEmbed3D(in_channels=shallow_feats[level],  # channels of input image to patch embedding
                             dim=self.embed_dims[level],
                             patch_size=patch_sizes[level],
                             method="proj")
            )

        # Stochastic depth decay
        self.pos_drop = nn.Dropout(p=drop)

        # Transformer blocks
        self.LX_blocks = nn.ModuleList()
        for level in range(num_levels):
            self.blocks = nn.ModuleList()
            for i in range(num_blks[level]):
                depths = self.blk_layers*num_blks[level]
                dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
                self.blocks.append(
                    AugustBlock(self.blk_layers, level, self.embed_dims, self.ct_embed_dims, ct_size, self.context_sizes, self.patch_sizes,
                                self.sizes_p, num_heads=num_heads,
                                attn_window_sizes=attn_window_sizes, enable_shift=enable_shift, mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0, layer_type=self.layer_type,
                                enable_ape_ct=self.enable_ape_ct, enable_ape_x=enable_ape_x,
                                enable_ct_rpb=self.enable_ct_rpb, enable_conv_skip=self.enable_conv_skip,
                                patch_pe_method=patch_pe_method)
                )
            self.LX_blocks.append(self.blocks)

        # Final block
        self.Final_blk = AugustDRCTLayerV2(num_levels - 1, self.embed_dims, self.ct_embed_dims, ct_size, self.context_sizes,
                                           self.patch_sizes, self.sizes_p,
                                           num_heads=num_heads,
                                           window_sizes=attn_window_sizes, shift_size=0,  # TODO: test shift_size > 0 of last block or not
                                           mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                                           drop_path=0.0, # use zero for last layer
                                           act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0,
                                           enable_ape_ct=self.enable_ape_ct, enable_ape_x=enable_ape_x,
                                           enable_ct_rpb=self.enable_ct_rpb, patch_pe_method=patch_pe_method, return_x_image=True,
                                           token_upsample_method=token_upsample_method)


        # Image space convolutional layer
        in_chans_last_conv = self.embed_dims[-1]  # self.embed_dims[-1]//patch_sizes[-1]**3
        self.conv_image = nn.Conv3d(in_channels=in_chans_last_conv,
                                    out_channels=shallow_feats[-1],
                                    kernel_size=3, stride=1, padding=1, bias=True)
        self.act_image = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if self.layer_type == "swin" or self.layer_type == "fastervit_without_ct":
            self.up_final_feats = SRBlock3D(embed_dims[-1], embed_dims[-1], k_size=6, pad=2,
                                     upsample_method=upsample_method, upscale_factor=2, use_checkpoint=False)

        recon_feats = shallow_feats[-1]

        # Upsampling
        if self.up_factor == 2:
            feats_2x = pre_up_feats[0]
            self.SR0 = SRBlock3D(shallow_feats[-1], feats_2x, k_size=6, pad=2, upsample_method=upsample_method, upscale_factor=2, use_checkpoint=False)
            recon_feats = feats_2x
        elif self.up_factor == 3:
            feats_3x = pre_up_feats[0]
            self.SR0 = SRBlock3D(shallow_feats[-1], feats_3x, k_size=6, pad=2, upsample_method=upsample_method, upscale_factor=3, use_checkpoint=False)
            recon_feats = feats_3x
        elif self.up_factor == 4:
            feats_2x = pre_up_feats[0]
            feats_4x = pre_up_feats[1]
            self.SR0 = SRBlock3D(shallow_feats[-1], feats_2x, k_size=6, pad=2, upsample_method=upsample_method, upscale_factor=2, use_checkpoint=False)
            self.SR1 = SRBlock3D(feats_2x, feats_4x, k_size=6, pad=2, upsample_method=upsample_method, upscale_factor=2, use_checkpoint=False)
            recon_feats = feats_4x

        # recon_feats = shallow_feats
        # if self.up_factor >= 2:
        #     feats_2x = pre_up_feats[0]
        #     self.SR0 = SRBlock3D(shallow_feats, feats_2x, k_size=6, pad=2,
        #                          upsample_method=upsample_method, upscale_factor=2, use_checkpoint=False)
        #     recon_feats = feats_2x
        #
        # if self.up_factor >= 4:
        #     feats_4x = pre_up_feats[1]
        #     self.SR1 = SRBlock3D(feats_2x, feats_4x, k_size=6, pad=2,
        #                          upsample_method=upsample_method, upscale_factor=2, use_checkpoint=False)
        #     recon_feats = feats_4x

        self.conv_last = nn.Sequential(
            #nn.Conv3d(in_channels=recon_feats, out_channels=recon_feats//2, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.PReLU(),
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

        # Crop 3D image
        return x[:, :, start_h:end_h, start_w:end_w, start_d:end_d]

    def forward(self, x):

        LX_img = x

        prev_x = None
        prev_ct = None

        for level in range(0, self.num_levels):
            ######## SFE module ########
            LX_img = self.sfe_blks[level](LX_img)  # TODO: test effect of shallow-feature-extraction module

            # Patch embedding
            LX_x = self.patch_embedding_blks[level](LX_img)

            # CT initialization
            LX_ct = None #self.test_ct_tokenizer_blks[level](LX_x)  # TODO: test creating CTs based on local token embeddings

            # Window partition local tokens for AugustBlocks
            LX_x = window_partition3D(LX_x, window_size=self.attn_window_sizes[level])

            for i, blk in enumerate(self.LX_blocks[level]):
                if self.use_checkpoint:
                    x_skip, ct_skip = checkpoint.checkpoint(blk, LX_x, LX_ct, prev_x, prev_ct)
                    LX_x = LX_x + x_skip
                    #LX_ct = LX_ct + ct_skip
                else:
                    x_skip, ct_skip = blk(LX_x, LX_ct, prev_x=prev_x, prev_ct=prev_ct)
                    LX_x = LX_x + x_skip                    
                    #LX_ct = LX_ct + ct_skip

                if i == self.num_blks[level] - 1 and level != self.num_levels - 1:
                    prev_x = LX_x
                    #prev_ct = LX_ct

            if level < self.num_levels - 1:
                LX_img = self.crop_next(LX_img, level + 1)

        if self.use_checkpoint:
            final_feats, _ = checkpoint.checkpoint(self.Final_blk, LX_x, LX_ct)
        else:
            final_feats, _ = self.Final_blk(LX_x, LX_ct)

        # Long skip connection of highest-level image
        # out = LX_img + self.conv_image(final_feats)  # Without activation with skip without sfe-module
        # final_feats = self.conv_image(final_feats)  # Without activation
        final_feats = self.act_image(self.conv_image(final_feats))  # With activation

        if self.enable_long_skip:
            out = LX_img + final_feats  # Works great
        else:
            out = final_feats  # without long skip connection
        #out = self.act_image(self.conv_image(final_feats))  # Use to test if features from transformers actually produce anything
        #out = LX_img  # Use to see effect of only convolutional layers

        #out = LX_img + self.act_image(self.conv_image(final_feats))  # With activation with skip without sfe-module
        # out = self.act_image(self.conv_image(final_feats)) # with activation without skip

        # Upsampling
        if self.up_factor == 2:
            out = self.SR0(out)
        elif self.up_factor == 3:
            out = self.SR0(out)
        elif self.up_factor == 4:
            out = self.SR0(out)
            out = self.SR1(out)

        out = self.conv_last(out)

        return out


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / 10 ** 9 if torch.cuda.is_available() else 0

    img_size = 32  # 48*2  # 48  # Should ideally be divisible by the patch_size*window_size
    print("image size: ", img_size)
    x = torch.randn((1, 1, img_size, img_size, img_size)).cuda()
    B, C, H, W, D = x.shape

    patch_size = 8  # size of each tokenized image patch. Standard is 4x4 so we try 4x4x4 for 3D
    attn_window_size = 4  # 4  # The size of the window to perform local attention within, M in the swin papers. standard is 7
    embed_dim = patch_size ** 3  # set to patch_size**3 to keep same amount of information in embedding, emb_dim 96 is default for 4x4x3 = 48 (number of features in 1 4x4 patch)

    #qm_test = torch.randn((1, 64, img_size, img_size, img_size)).cuda()
    #qm = QueueModuleV2(S=64, R=32, T=64).cuda()
    #x = qm(qm_test)

    # windows = window_partition3D(x, window_size=patch_size).to("cuda")

    # Patch embedding
    # patch_embed = PatchEmbed3D(in_chans=1, dim=embed_dim, patch_size=patch_size).to("cuda")
    # emb_windows = patch_embed(x)
    # emb_windows = emb_windows.view(-1, attn_window_size**3, embed_dim)  # (nW*B, window_size*window_size, C)

    # Window attention
    # window_attn = WindowAttention3D(dim=embed_dim, window_size=[attn_window_size, attn_window_size, attn_window_size], num_heads=4, qkv_bias=True, attn_drop=0., proj_drop=0., pretrained_window_size=[0, 0]).to("cuda")
    # emb_windows_attn = window_attn(emb_windows)
    # print("emb_windows_attn: ", emb_windows_attn.shape)

    # merge windows
    # attn_windows = emb_windows_attn.view(-1, attn_window_size, attn_window_size, attn_window_size, embed_dim)
    # x_out = window_reverse3D(attn_windows, attn_window_size, int(H/patch_size), int(W/patch_size), int(D/patch_size)).to("cuda")  # B H' W' C
    # print("x_out: ", x_out.shape)

    context_sizes = [32]
    num_levels = len(context_sizes)  # 3
    shallow_feats = 128  # 128 normally
    pre_up_feats = [64, 64]
    num_blks = [3]  # [1, 1, 3]  # [6, 6, 6]
    blk_layers = [6]  # Number of transformer layers per block in each level
    patch_sizes = [2]  # [16, 8, 2]
    ct_size = 4
    ct_pool_method = "conv"
    ct_embed_dims = [128, 128, 128]  # 128 normally. Old model: [512, 128, 64]  # [512, 128, 64]
    embed_dims = [128, 128, 128]  # 128 normally. Old model: [512, 128, 64]  # [512, 128, 64]
    attn_window_sizes = [8, 8, 8]  # [4, 4, 4]
    num_heads = 4
    enable_ape_ct = True
    enable_ape_x = False
    enable_ct_rpb = True
    enable_conv_skip = False
    enable_long_skip = True
    enable_shift = True
    layer_type = "fastervit"  # fastervit_without_ct, swin, fastervit
    patch_pe_method = "window_relative"  # "absolute", "window_relative"
    token_upsample_method = "deconv_nn_resize"  # "deconv_nn_resize" "pixelshuffle3D" "Monaipixelshuffle" "nearest"
    upsample_method = "pixelshuffle3D"  # "deconv_nn_resize" "pixelshuffle3D" "Monaipixelshuffle" "nearest"

    # TODO: implement overlapping patches
    overlap_patches = False

    up_factor = 4

    input_size = (H, W, D)
    net = MTVNet_no_CT(input_size=input_size, up_factor=up_factor, num_levels=num_levels, context_sizes=context_sizes, num_blks=num_blks,
                    blk_layers=blk_layers, in_chans=1, shallow_feats=shallow_feats, pre_up_feats=pre_up_feats, ct_embed_dims=ct_embed_dims,
                    embed_dims=embed_dims, ct_size=ct_size, ct_pool_method=ct_pool_method, patch_sizes=patch_sizes,
                    num_heads=num_heads, attn_window_sizes=attn_window_sizes, enable_shift=enable_shift, mlp_ratio=4., qkv_bias=True,
                    drop=0., attn_drop=0., drop_path=0.1, token_upsample_method=token_upsample_method, upsample_method=upsample_method, use_checkpoint=True,
                    layer_type=layer_type, enable_ape_ct=enable_ape_ct, enable_ape_x=enable_ape_x, enable_ct_rpb=enable_ct_rpb,
                    enable_conv_skip=enable_conv_skip, enable_long_skip=enable_long_skip, patch_pe_method=patch_pe_method).to(device)  # fastervit_without_ct, swin, fastervit
    net.train()

    print("Number of parameters", numel(net, only_trainable=True))

    start = time.time()
    #with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    august_out = net(x)
    stop = time.time()
    print("Time elapsed:", stop - start)

    print("AugustNet output shape:", august_out.shape)

    x_hr = torch.randn((1, 1, img_size * up_factor // (2 ** (num_levels - 1)),
                        img_size * up_factor // (2 ** (num_levels - 1)),
                        img_size * up_factor // (2 ** (num_levels - 1)))).cuda()

    loss_func = nn.MSELoss()
    loss = loss_func(august_out, x_hr)
    loss.backward()

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(x[0, 0, :, :, patch_size // 2].cpu().numpy(), cmap='gray')
    # plt.subplot(1, 2, 2)
    # plt.imshow(august_out[0, 0, :, :, patch_size // (2 * up_factor)].detach().cpu().numpy(), cmap='gray')
    # plt.show()

    # bl = monaibasiclayer(dim=128,
    #                 depth=6,
    #                 num_heads=4,
    #                 window_size=[4,4,4],
    #                 drop_path=[0.0]*6,
    #                 mlp_ratio=4.0,
    #                 qkv_bias=False,
    #                 drop=0.0,
    #                 attn_drop=0.0,
    #                 norm_layer=nn.LayerNorm,
    #                 downsample=None,
    #                 use_checkpoint=False).cuda()
    #
    # x = torch.randn((1, 128, 32, 32, 32)).cuda()
    # test = bl(x)
    # print(test.shape)

    max_memory_reserved = torch.cuda.max_memory_reserved()
    print("Maximum memory reserved: %0.3f Gb / %0.3f Gb" % (max_memory_reserved / 10 ** 9, total_gpu_mem))

    print("Done")
