import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

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

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.unsqueeze(1) + relative_position_bias, dropout_p=0.0, is_causal=False)

        return x

    def forward(self, x, mask=None):
        return self.forward_flash(x, mask)

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

if __name__ == "__main__":

    print("Flash Attention:", torch.backends.cuda.flash_sdp_enabled())
    print("Mem Efficient  :", torch.backends.cuda.mem_efficient_sdp_enabled())
    print("Math SDP       :", torch.backends.cuda.math_sdp_enabled())

    device = torch.cuda.get_device_name()
    print("GPU:", device)
    print("PyTorch:", torch.__version__)

    batch_size = 1
    dim = 256
    window_size = (8, 8, 8)
    num_heads = 4


    patch_size = 32

    x_patches = torch.randn(batch_size, patch_size, patch_size, patch_size, dim).cuda()
    x_windows = window_partition3D(x_patches, window_size=8)

    # x = torch.randn(8, 512, 256).cuda()
    mask = torch.zeros(64, 512, 512).cuda()

    model = WindowAttention3D(dim,
                              window_size,
                              num_heads,
                              qkv_bias=True,
                              attn_drop=0.,
                              proj_drop=0.).cuda()

    start = time.time()
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for _ in range(100):
            out = model(x_windows, mask=mask)
    stop = time.time()

    print("Output shape:", out.shape)
    print("Time elapsed:", stop - start)
