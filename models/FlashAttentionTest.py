import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_3tuple

from models.models_3D import SRBlock3D
#from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

from monai.networks.nets.swin_unetr import SwinTransformerBlock
from monai.networks.nets.swin_unetr import WindowAttention as WindowAttentionMonai

from torch.nn.attention import SDPBackend, sdpa_kernel

if False:
    class FlashAttentionLayer(torch.nn.Module):
        def __init__(self, dim, heads):
            super().__init__()
            self.heads = heads
            self.dim = dim
            self.head_dim = dim // heads
            self.qkv = torch.nn.Linear(dim, dim * 3)

        def forward(self, x):
            B, T, _ = x.size()
            qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)

            # reshape for FlashAttention
            q = q.reshape(B * self.heads, T, self.head_dim)
            k = k.reshape(B * self.heads, T, self.head_dim)
            v = v.reshape(B * self.heads, T, self.head_dim)

            out = flash_attn_func(q, k, v, dropout_p=0.0)
            out = out.reshape(B, self.heads, T, self.head_dim).transpose(1, 2)
            return out.reshape(B, T, self.dim)

    with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
        F.scaled_dot_product_attention(q, k, v, attn_mask=mask.unsqueeze(1), dropout_p=0.0, is_causal=False)


def compute_mask(x_dims, window_size, shift_size):
    '''
    Computing region masks based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

    :param x_dims:
    :param window_size:
    :param shift_size:
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

    def forward_original(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            ct: carrier tokens with shape of (num_windows*B, ct_per_window, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))

        # x has shape [num_windows, num_patches_per_window, emb_dim]
        # The qkv for the first window can be calculated using "F.linear(input=x[0], weight=self.qkv.weight, bias=qkv_bias)"
        # if we want to do this sequentially.
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        # q, k, v have the shape [num_windows, num_heads, num_patches_per_window, emb_dim/num_heads]
        # The attention for each window can be found using "(F.normalize(q[0], dim=-1) @ F.normalize(k[0], dim=-1).transpose(-2, -1))"
        # if we want to calculate attention sequentially within each window

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        # TODO: Fix relative position bias here - because of added CTs the relative_position_bias does not match the dimensions of the attention
        # TODO: score matrix. Then position biases should only be added to the window tokens of x and not the CTs
        # attn = attn + relative_position_bias.unsqueeze(0)

        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        # attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01).cuda())).exp()
        attn = attn * logit_scale

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            #attn = attn.view(B_ // nW, nW, self.num_heads, N + Nct, N + Nct) + mask[:, :N, :N].unsqueeze(1).unsqueeze(0) # For use with no CTs
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask[:, :N, :N].unsqueeze(1).unsqueeze(0) # Use this when CTs are enabled
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_flash(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            ct: carrier tokens with shape of (num_windows*B, ct_per_window, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))

        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)

        if mask is not None:
             x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.unsqueeze(1) + relative_position_bias, dropout_p=0.0, is_causal=False)
        else:
             x = F.scaled_dot_product_attention(q, k, v, attn_mask=relative_position_bias, dropout_p=0.0, is_causal=False)

        return x

    def forward(self, x, mask=None):
        return self.forward_flash(x, mask)
        #return self.forward_original(x, mask)

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


class WindowAttention3D_FAST(nn.Module):
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

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., pretrained_window_size=(0, 0, 0)):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
                num_heads,
            )
        )
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])

        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
        #coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))

        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
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
        self.attn_drop = attn_drop

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            ct: carrier tokens with shape of (num_windows*B, ct_per_window, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))

        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        relative_position_bias = relative_position_bias.unsqueeze(0)  # (1, num_heads, N, N)

        if mask is not None:
            nW = mask.shape[0]
            B = B_ // nW
            mask = mask.unsqueeze(1)  # (nW, 1, N, N)
            mask = mask.repeat(B, 1, 1, 1)  # (B_, 1, N, N)
            attn_mask = mask + relative_position_bias
        else:
            attn_mask = relative_position_bias

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)

        x = x.transpose(1, 2).reshape(B_, N, C)

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

class CA(nn.Module):
    """Channel attention.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(CA, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Linear(num_feat, num_feat // squeeze_factor),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(num_feat // squeeze_factor, num_feat),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class MultiHeadMlp(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=4,
        mlp_ratios=4.0,
        act_layer=nn.GELU,
        drop=0.0,
        squeeze_factor=16,
        res_scale=1.0,
    ):
        super().__init__()

        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.res_scale = res_scale

        if isinstance(mlp_ratios, float) or isinstance(mlp_ratios, int):
            hidden_dims = [int(self.head_dim * mlp_ratios) for _ in range(num_heads)]
        else:
            hidden_dims = [int(self.head_dim * mlp_ratio) for mlp_ratio in mlp_ratios]

        # Create one MLP per head
        self.mlps = nn.ModuleList([
            Mlp(
                in_features=self.head_dim,
                hidden_features=hidden_dims[i],
                out_features=self.head_dim,
                act_layer=act_layer,
                drop=drop
            )
            for i in range(num_heads)
        ])

        # Channel attention
        self.ca = CA(dim, squeeze_factor=squeeze_factor)

    def forward(self, x):
        """
        x: (B, N, C)
        """
        B, N, C = x.shape

        # (B, N, num_heads, head_dim)
        x = x.view(B, N, self.num_heads, self.head_dim)

        # Apply MLP to each head independently
        head_outputs = []
        for i in range(self.num_heads):
            head_outputs.append(self.mlps[i](x[:, :, i, :]))  # (B, N, head_dim)

        # Concatenate channels back
        out = torch.cat(head_outputs, dim=-1)  # (B, N, C)

        # Apply residual channel attention
        out = out + self.ca(out) * self.res_scale

        return out



class WindowCrossAttention3D_FAST(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., pretrained_window_size=(0, 0, 0),
                 query_token_size=1, key_token_size=1):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads
        self.query_token_size = query_token_size
        self.key_token_size = key_token_size

        max_rel_d = (self.window_size[0] - 1) * max(self.query_token_size, self.key_token_size)
        max_rel_h = (self.window_size[1] - 1) * max(self.query_token_size, self.key_token_size)
        max_rel_w = (self.window_size[2] - 1) * max(self.query_token_size, self.key_token_size)

        # number of discrete embeddings along each relative axis
        num_bins_d = 2 * max_rel_d + 1
        num_bins_h = 2 * max_rel_h + 1
        num_bins_w = 2 * max_rel_w + 1

        num_relative_positions = num_bins_d * num_bins_h * num_bins_w

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_relative_positions, num_heads)
        )

        coords_q_d = torch.arange(self.window_size[0]) * self.query_token_size
        coords_q_h = torch.arange(self.window_size[1]) * self.query_token_size
        coords_q_w = torch.arange(self.window_size[2]) * self.query_token_size
        coords_q = torch.stack(torch.meshgrid(coords_q_d, coords_q_h, coords_q_w, indexing="ij"))  # (3, D, H, W)
        coords_query = torch.flatten(coords_q, 1)  # (3, Nq)

        coords_k_d = torch.arange(self.window_size[0]) * self.key_token_size
        coords_k_h = torch.arange(self.window_size[1]) * self.key_token_size
        coords_k_w = torch.arange(self.window_size[2]) * self.key_token_size
        coords_k = torch.stack(torch.meshgrid(coords_k_d, coords_k_h, coords_k_w, indexing="ij"))  # (3, D, H, W)
        coords_key = torch.flatten(coords_k, 1)  # (3, Nk)

        relative_coords = (coords_query[:, :, None] - coords_key[:, None, :]).permute(1, 2, 0).contiguous()

        relative_coords[:, :, 0] += max_rel_d
        relative_coords[:, :, 1] += max_rel_h
        relative_coords[:, :, 2] += max_rel_w
        relative_coords[:, :, 0] = relative_coords[:, :, 0] * (num_bins_h * num_bins_w)
        relative_coords[:, :, 1] = relative_coords[:, :, 1] * (num_bins_w)
        relative_position_index = relative_coords.sum(-1)  # (Nq, Nk)

        # register buffer (long), it will move with model to device
        self.register_buffer("relative_position_index", relative_position_index)

        # This part should be the same
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = attn_drop

    def forward(self, x1, x2):

        B1, N1, C1 = x1.shape
        B2, N2, C2 = x2.shape

        # Compute q from x1
        q = F.linear(input=x1, weight=self.q.weight, bias=self.q_bias)
        q = q.reshape(B1, -1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3)

        v_bias = None
        if self.v_bias is not None:
            v_bias = torch.cat((torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))

        # Compute k and v from x2
        kv = F.linear(input=x2, weight=self.kv.weight, bias=v_bias)
        kv = kv.reshape(B2, -1, 2, self.num_heads, C2 // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N1, N2, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=relative_position_bias, dropout_p=self.attn_drop, is_causal=False)
        # x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop, is_causal=False)

        x = x.transpose(1, 2).reshape(B1, N1, C1)

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


def window_partition3D(x, window_size):
    """
    Args:
        x: (B, Hp, Wp, Dp, C)
        window_size (int): window size

    Returns:
        windows: (nW, nP, C)
    """

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

        input_tensor = input_tensor + self.pos_emb
        return input_tensor



class STLayer(nn.Module):

    def __init__(self, level, embed_dims, context_sizes, patch_sizes, sizes_p, num_heads, window_sizes,
                 shift_size=0, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0, enable_ape_x=False,
                 patch_pe_method="window_relative", return_x_image=False,
                 token_upsample_method="Monaipixelshuffle", init_CSM=False):
        super().__init__()
        self.level = level
        self.embed_dim = embed_dims[level]
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

        # Use this flag to initialize the CSM module to merge current-level and previous-level local tokens
        self.init_CSM = init_CSM

        self.nW = self.Hp // self.window_size  # Should (ideally) be done for Hp, Wp, Dp // August
        self.sr_ratio = self.nW

        self.enable_ape_x = enable_ape_x
        self.patch_pe_method = patch_pe_method

        self.token_upsample_method = token_upsample_method

        if self.enable_ape_x:
            self.pos_embed = PatchPositionalBias1D(self.embed_dim, rank=3, seq_length=self.Hp * self.Wp * self.Dp,
                                                   level=level, context_sizes=self.context_sizes, conv=False, input_format="windows",
                                                   window_size=self.window_size, method=patch_pe_method)  # For local tokens, method="absolute", "window_relative"

        ######### CSM init #########
        if self.init_CSM:
            self.csm_method = "crossattn"
            qkv_bias = True

            ######### Multi-level patch merge #########
            self.prev_mlp_hidden_dim = self.prev_embed_dim
            self.mlp_down = Mlp(in_features=self.prev_embed_dim, hidden_features=self.prev_mlp_hidden_dim,
                                out_features=self.embed_dim, act_layer=act_layer, drop=drop)

            self.cross_attn_func = WindowCrossAttention3D_FAST(self.embed_dim, window_size=to_3tuple(self.window_size),
                                                          num_heads=num_heads, qkv_bias=True, attn_drop=attn_drop,
                                                          pretrained_window_size=to_3tuple(pretrained_window_size))

            self.cross_norm = norm_layer(self.embed_dim)

        ######### Image space upsampling #########
        if self.return_x_image:
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
                nn.Conv3d(self.embed_dim, self.embed_dim, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(self.embed_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

        # Layer norm 1 - after attention is Swin V2 paper
        self.norm1 = norm_layer(self.embed_dim)

        # Swin V2 Window Attention
        self.attn_func = WindowAttention3D_FAST(self.embed_dim, window_size=to_3tuple(self.window_size), num_heads=num_heads,
                                           qkv_bias=qkv_bias, attn_drop=attn_drop,
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
            attn_mask = compute_mask(x_dims=(self.Hp, self.Wp, self.Dp), window_size=self.window_size, shift_size=self.shift_size)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x_windows, prev_x=None):

        # calculate batch size B
        B = int(x_windows.shape[0] // (self.Hp * self.Wp * self.Dp / self.window_size**3))

        nW, nP, C = x_windows.shape

        # positional bias for patch tokens - NOTE: windowed format
        if self.enable_ape_x:
            x_windows = self.pos_embed(x_windows.view(B, -1, nP, C)).view(nW, nP, C)  # 1D positional bias by FasterViT

        # Cross-Attention of previous level patch features
        if prev_x is not None:
            if self.csm_method == "adaLN":  # Adaptive Layer Norm Zero (adaLN-Zero) conditioning from DiT paper.
                pass
            elif self.csm_method == "crossattn":
                prev_x_windows = self.mlp_down(prev_x)
                if nW != prev_x_windows.shape[0]:  # Repartition previous-level patches using current-level window size
                    prev_x_windows = window_reverse3D(prev_x_windows, self.prev_window_size, dims=(B, self.prev_Hp, self.prev_Wp, self.prev_Dp))
                    prev_x_windows = window_partition3D(prev_x_windows, window_size=self.window_size)
                # Window cross-attention
                cross_attn_windows = self.cross_attn_func(x_windows, prev_x_windows)  # nW*B, window_size*window_size, C
                x_windows = x_windows + self.cross_norm(cross_attn_windows)

        # Register skip connection
        shortcut = x_windows

        # Norm 1
        x_windows = self.norm1(x_windows)

        # Shifting windows for W-MSA (optional)
        if self.shift_size > 0:
            x = window_reverse3D(x_windows, self.window_size, dims=(B, self.Hp, self.Wp, self.Dp))
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(1, 2, 3))
            x_windows = window_partition3D(x, window_size=self.window_size)

        # W-MSA/SW-MSA
        x_windows = self.attn_func(x_windows, mask=self.attn_mask)

        # Shifting windows for W-MSA (optional)
        if self.shift_size > 0:
            x = window_reverse3D(x_windows, self.window_size, dims=(B, self.Hp, self.Wp, self.Dp))
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(1, 2, 3))
            x_windows = window_partition3D(x, window_size=self.window_size)

        # Norm and skip connection after attention - NOTE: windowed format of shortcut
        x_windows = shortcut + self.drop_path(x_windows)

        # Second part: norm, MLP, skip connection
        x_windows = x_windows + self.drop_path(self.mlp(self.norm2(x_windows)))

        if self.return_x_image:

            # x_image_space = x.view(-1, self.Hp, self.Wp, self.Dp, C) # TODO: Maybe use .view instead of window_reverse3D
            x_image_space = window_reverse3D(x_windows, self.window_size, dims=(B, self.Hp, self.Wp, self.Dp))  # TODO: Maybe use window_reverse3D instead of .view

            # Upsample tokens if needed
            x_image_space = self.image_upsampler(x_image_space.permute(0, 4, 1, 2, 3))

            # Return x_image_space as x and None as ct as x_image_space also contains information from ct
            return x_image_space
        else:
            return x_windows

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



class STLayerV2(nn.Module):

    def __init__(self, level, embed_dims, context_sizes, patch_sizes, sizes_p, num_heads, window_sizes,
                 shift_size=0, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0, enable_ape_x=False,
                 patch_pe_method="window_relative", return_x_image=False, token_upsample_method="Monaipixelshuffle"):
        super().__init__()
        self.level = level
        self.embed_dim = embed_dims[level]
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

        self.nW = self.Hp // self.window_size  # Should (ideally) be done for Hp, Wp, Dp // August
        self.sr_ratio = self.nW

        self.enable_ape_x = enable_ape_x
        self.patch_pe_method = patch_pe_method

        self.token_upsample_method = token_upsample_method

        if self.enable_ape_x:
            self.pos_embed = PatchPositionalBias1D(self.embed_dim, rank=3, seq_length=self.Hp * self.Wp * self.Dp,
                                                   level=level, context_sizes=self.context_sizes, conv=False, input_format="windows",
                                                   window_size=self.window_size, method=patch_pe_method)  # For local tokens, method="absolute", "window_relative"

        ######### Image space upsampling #########
        if self.return_x_image:
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
                nn.Conv3d(self.embed_dim, self.embed_dim, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(self.embed_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

        # Layer norm 1 - after attention is Swin V2 paper
        self.norm1 = norm_layer(self.embed_dim)

        # Swin V2 Window Attention
        self.attn_func = WindowAttention3D_FAST(self.embed_dim, window_size=to_3tuple(self.window_size), num_heads=num_heads,
                                           qkv_bias=qkv_bias, attn_drop=attn_drop, pretrained_window_size=to_3tuple(pretrained_window_size))

        # Timm stochastic dropout
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Layer norm 2 - after MLP in Swin V2 paper
        self.norm2 = norm_layer(self.embed_dim)

        # MLP
        mlp_hidden_dim = int(self.embed_dim * mlp_ratio)
        self.mlp = Mlp(in_features=self.embed_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # Initialization of mask for 3D shifted-window attention with CTs
        if self.shift_size > 0:
            attn_mask = compute_mask(x_dims=(self.Hp, self.Wp, self.Dp), window_size=self.window_size, shift_size=self.shift_size)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):

        B, H, W, D, C = x.shape

        # # positional bias for patch tokens - NOTE: windowed format
        # if self.enable_ape_x:
        #     x_windows = self.pos_embed(x_windows.view(B, -1, nP, C)).view(nW, nP, C)  # 1D positional bias by FasterViT

        # Register skip connection
        shortcut = x

        # Norm 1
        x = self.norm1(x)

        # Shifting windows for W-MSA (optional)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(1, 2, 3))

        # Window partitioning
        x_windows = window_partition3D(x, window_size=self.window_size)

        # W-MSA/SW-MSA
        x_windows = self.attn_func(x_windows, mask=self.attn_mask)

        # Window reverse
        x = window_reverse3D(x_windows, self.window_size, dims=(B, self.Hp, self.Wp, self.Dp))

        # Shifting windows for W-MSA (optional)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(1, 2, 3))

        # Norm and skip connection after attention - NOTE: windowed format of shortcut
        x = shortcut + self.drop_path(x)

        # Second part: norm, MLP, skip connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.return_x_image:

            # Upsample tokens if needed
            x_image_space = self.image_upsampler(x.permute(0, 4, 1, 2, 3)) # B, C, H, W, D

            # Return x in image space (B, C, H, W, D)
            return x_image_space

        # Otherwise return x in (B, Hp, Wp, Dp, C) format
        return x

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



if __name__ == "__main__":

    print("Flash Attention:", torch.backends.cuda.flash_sdp_enabled())
    print("Mem Efficient  :", torch.backends.cuda.mem_efficient_sdp_enabled())
    print("Math SDP       :", torch.backends.cuda.math_sdp_enabled())

    device = torch.cuda.get_device_name()
    print("GPU:", device)
    print("PyTorch:", torch.__version__)

    batch_size = 2
    dim = 256
    window_size = (8, 8, 8)
    num_heads = 4

    patch_size = 24

    x_patches = torch.randn(batch_size, patch_size, patch_size, patch_size, dim).cuda()
    x_windows = window_partition3D(x_patches, window_size=8)

    # x = torch.randn(8, 512, 256).cuda()
    mask = torch.randn(128, 512, 512).cuda()

    model = WindowAttention3D_FAST(dim,
                              window_size,
                              num_heads,
                              qkv_bias=True,
                              attn_drop=0.).cuda()

    # model = WindowAttentionMonai(dim,
    #                              num_heads,
    #                              window_size,
    #                              qkv_bias=True,
    #                              attn_drop=0.,
    #                              proj_drop=0.).cuda()

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(10):
            out = model(x_windows, mask=None)

    start = time.time()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(100):
            out = model(x_windows, mask=None)
    stop = time.time()

    print("Test WindowAttention3D_FAST")
    print("Output shape:", out.shape)
    print("Time elapsed:", stop - start)


    layer = STLayer(level=0,
                    embed_dims=[256],
                    context_sizes=[patch_size],
                    patch_sizes=[1],
                    sizes_p=[(patch_size, patch_size, patch_size)],
                    num_heads=4,
                    window_sizes=[8],
                    shift_size=0,
                    mlp_ratio=4.,
                    qkv_bias=True,
                    drop=0.,
                    attn_drop=0.,
                    drop_path=0.,
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,
                    pretrained_window_size=0,
                    enable_ape_x=False,
                    patch_pe_method="window_relative",
                    return_x_image=False,
                    token_upsample_method="Monaipixelshuffle",
                    init_CSM=False).cuda()

    # layer_compiled = torch.compile(layer)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(10):
            out = layer(x_windows, prev_x=None)

    start = time.time()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(100):
            out = layer(x_windows, prev_x=None)
    stop = time.time()

    print("Test STLayer")
    print("Output shape:", out.shape)
    print("Time elapsed:", stop - start)

    print("STLayer output shape:", out.shape)

    monai_layer = SwinTransformerBlock(dim,
                                       num_heads,
                                       window_size,
                                       shift_size=(0, 0, 0),
                                       mlp_ratio=4.,
                                       qkv_bias=True,
                                       drop=0.0,
                                       attn_drop=0.0,
                                       drop_path=0.0,
                                       act_layer="GELU",
                                       norm_layer=nn.LayerNorm,
                                       use_checkpoint=False).cuda()

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(10):
            out = monai_layer(x_patches, mask_matrix=None)

    start = time.time()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(100):
            out = monai_layer(x_patches, mask_matrix=None)
    stop = time.time()

    print("Test Monai SwinTransformerBlock")
    print("Output shape:", out.shape)
    print("Time elapsed:", stop - start)

    print("Monai output shape:", out.shape)

    layer = STLayerV2(level=0,
                    embed_dims=[256],
                    context_sizes=[patch_size],
                    patch_sizes=[1],
                    sizes_p=[(patch_size, patch_size, patch_size)],
                    num_heads=4,
                    window_sizes=[8],
                    shift_size=0,
                    mlp_ratio=4.,
                    qkv_bias=True,
                    drop=0.,
                    attn_drop=0.,
                    drop_path=0.,
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,
                    pretrained_window_size=0,
                    enable_ape_x=False,
                    patch_pe_method="window_relative",
                    return_x_image=False,
                    token_upsample_method="Monaipixelshuffle"
                    ).cuda()

    # layer_compiled = torch.compile(layer)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(10):
            out = layer(x_patches)

    start = time.time()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(100):
            out = layer(x_patches)
    stop = time.time()

    print("Test STLayerV2")
    print("Output shape:", out.shape)
    print("Time elapsed:", stop - start)

    print("STLayerV2 output shape:", out.shape)

    from models.xcit3D import XCABlock

    layer = XCABlock(dim=128, num_heads=4, mlp_ratio=4., qkv_bias=True,
                         drop=0., attn_drop=0., drop_path=0.,
                         norm_layer=nn.LayerNorm, act_layer=nn.GELU, eta=1.0).cuda()

    # layer_compiled = torch.compile(xca_layer)

    x_patches = torch.randn(batch_size, patch_size, patch_size, patch_size, 128).cuda()
    Dp, Hp, Wp = patch_size, patch_size, patch_size
    x_flat = x_patches.reshape(batch_size, -1, 128)  # B, N, C

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(10):
            out = layer(x_flat, D=Dp, H=Hp, W=Wp)

    start = time.time()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(100):
            out = layer(x_flat, D=Dp, H=Hp, W=Wp)
    stop = time.time()

    print("Test XCA Block")
    print("Output shape:", out.shape)
    print("Time elapsed:", stop - start)
    print("XCA output shape:", out.shape)

