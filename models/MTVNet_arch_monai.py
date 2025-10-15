# --------------------------------------------------------
# MTVNet (cleaned)
# Written by August Høeg
# --------------------------------------------------------
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import Tensor, nn

from models.models_3D import SRBlock3D
from utils.utils_3D_image import numel

from monai.networks.nets.swin_unetr import SwinTransformerBlock


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


class TransitionLayer(nn.Module):
    r""" Transition layer between two stages
    Args:
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.layer = nn.Sequential(
            LayerNorm(in_dim, eps=1e-6, data_format="channels_first"),
            nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True),
        )
    def forward(self, x):
        return self.layer(x)


def window_partition3D(x, window_size):
    """
    Partition 5D image tensor into windows.

    Args:
        x: (B, Hp, Wp, Dp, C)
        window_size (int): window size

    Returns:
        windows: (nW, window_size**3, C)
    """
    B, Hp, Wp, Dp, C = x.shape
    x = x.view(B, Hp // window_size, window_size,
               Wp // window_size, window_size,
               Dp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size**3, C)
    return windows


def window_reverse3D(windows, window_size, dims):
    """
    Reverse windows back to image space.

    Args:
        windows: (num_windows * B, window_size**3, C)
        dims: (B, H, W, D)
    Returns:
        x: (B, H, W, D, C)
    """
    B, H, W, D = dims
    nW, nP, C = windows.shape
    x = windows.view(-1, window_size, window_size, window_size, C)
    x = x.view(B, H // window_size, W // window_size, D // window_size,
               window_size, window_size, window_size, C)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, H, W, D, C)
    return x


class PatchEmbed3D(nn.Module):
    """
    Very small 3D patch embedder: conv projection (proj) or flatten fallback.
    """

    def __init__(self, in_channels=1, dim=96, patch_size=4, method="proj", out_format="image"):
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim
        self.patch_size = patch_size
        self.method = method
        self.out_format = out_format

        pad = 1 if patch_size >= 3 else 0

        if self.method == "proj":
            self.proj = nn.Conv3d(in_channels, dim, kernel_size=patch_size, stride=patch_size, padding=pad, bias=False)
        else:
            # fallback: flatten (used rarely)
            self.proj = None

    def forward(self, x):
        # x: (B, C, H, W, D)
        if self.method == "proj":
            x = self.proj(x)  # (B, dim, Hp, Wp, Dp)
            if self.out_format == "tokens":
                x = x.flatten(2).transpose(1, 2)  # (B, N, C)
            elif self.out_format == "image":
                x = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, Hp, Wp, Dp, C)
            elif self.out_format == "same":
                pass
        else:
            x = x.flatten(2).transpose(1, 2)
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


class WindowCrossAttention3D(nn.Module):
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


    def forward(self, x1, x2):
        """
        Args:
            x1: input features with shape of (num_windows*B, N, C)
            x2: input features with shape of (num_windows*B, N, C)
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



class Group(nn.Module):

    def __init__(self, input_size, patch_size, dim, skip_dim, depth, window_size=8, num_heads=4, mlp_ratio=4, dp_rates=None, prev_dim=None):
        super().__init__()

        self.layers = nn.ModuleList()
        self.compress_layers = nn.ModuleList()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        for layer_idx in range(depth):
            channel_dim = dim + skip_dim * layer_idx
            compress_dim = skip_dim if layer_idx < depth - 1 else dim

            self.window_size = window_size
            self.shift_size = window_size // 2 if (layer_idx % 2) == 1 else 0
            self.num_heads = num_heads

            self.layers.append(
                    SwinTransformerBlock(
                        dim=channel_dim,
                        num_heads=self.num_heads,
                        window_size=[self.window_size, self.window_size, self.window_size],
                        shift_size=[self.shift_size, self.shift_size, self.shift_size],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=True,
                        drop=0.0,
                        attn_drop=0.0,
                        drop_path=dp_rates[layer_idx],
                        norm_layer=nn.LayerNorm,
                        use_checkpoint=False)
            )

            self.compress_layers.append(
                nn.Conv3d(channel_dim, compress_dim, kernel_size=1, stride=1, padding=0)
            )

            if layer_idx % 2 == 1:
                mlp_ratio = max(1, mlp_ratio // 2)

        Dp, Hp, Wp = [input_size // patch_size] * 3  # hardcoded for now
        self.mask_matrix = compute_mask((Dp, Hp, Wp), self.window_size, self.shift_size).cuda()  # hardcoded for now

        if prev_dim is not None:
            ######### Multi-level patch merge #########
            self.mlp_match = Mlp(in_features=prev_dim,
                                hidden_features=prev_dim * 1,
                                out_features=dim)

            self.cross_attn = WindowCrossAttention3D(dim,
                                                     window_size=[self.window_size, self.window_size, self.window_size],
                                                     num_heads=self.num_heads,
                                                     q_bias=True,
                                                     kv_bias=True,
                                                     qk_scale=None,
                                                     attn_drop=0.0,
                                                     proj_drop=0.0)

            self.cross_norm = nn.LayerNorm(dim)


    def forward(self, x, prev_x=None):

        B, C, D, H, W = x.shape

        x_input = x
        next_x = x

        ###### Dense-connected block structure ######
        for i in range(len(self.layers) - 1):

            x = x.permute(0, 2, 3, 4, 1).contiguous()

            prev_x = None if i > 0 else prev_x
            if prev_x is not None:
                prev_x = prev_x.permute(0, 2, 3, 4, 1).contiguous()  # B, D, H, W, C
                prev_x = self.mlp_match(prev_x)

                x = window_partition3D(x, window_size=self.window_size)
                prev_x = window_partition3D(prev_x, window_size=self.window_size)

                cross_attn_windows = self.cross_attn(x, prev_x)  # nW*B, window_size*window_size, C
                x = x + self.cross_norm(cross_attn_windows)
                x = window_reverse3D(x, window_size=self.window_size, dims=(B, D, H, W))

            # Main layer
            x = self.layers[i](x, self.mask_matrix)

            x = x.permute(0, 4, 1, 2, 3).contiguous()  # B, C, D, H, W

            # Compress
            x = self.act(self.compress_layers[i](x))

            # Concatenate
            next_x = torch.cat([next_x, x], 1)
            x = next_x

        # Final layer
        next_x = next_x.permute(0, 2, 3, 4, 1).contiguous()

        x = self.layers[-1](next_x, prev_x)

        x = x.permute(0, 4, 1, 2, 3).contiguous()

        x = self.act(self.compress_layers[-1](x))

        x = 0.2 * x + x_input

        return x


class MTVNet_monai(nn.Module):
    """
    Cleaned, minimal MTVNet that only takes used args.

    Only accepts arguments that are used by the implementation:
      - input_size: tuple (H, W, D)
      - up_factor: int (2,3,4 supported)
      - num_levels: int
      - context_sizes: list[int] (size per level)
      - num_blks: list[int] (number of "blocks" per level)
      - blk_layers: list[int] (kept for compatibility; not actively used here)
      - in_chans: number of input channels (usually 1)
      - shallow_feats: list[int] (channels after SFE per level)
      - pre_up_feats: list[int] (features for SR blocks)
      - embed_dims: list[int] (embedding dims per level -> used for patch embedding dims)
      - patch_sizes: list[int] (patch sizes per level)
      - attn_window_sizes: list[int] (window sizes used for partitioning)
      - use_checkpoint: whether to enable checkpointing (kept but not essential here)
      - upsample_method: forwarded into SRBlock3D
      - enable_long_skip: whether to add LX_img long skip
      - layer_type: kept as an attribute if you want to switch behavior later
    """

    def __init__(self,
                 input_size,
                 up_factor,
                 num_levels,
                 context_sizes,
                 num_blks,
                 blk_layers,
                 in_chans,
                 shallow_feats,
                 pre_up_feats,
                 embed_dims,
                 num_heads,
                 mlp_ratio,
                 attn_window_sizes,
                 patch_sizes,
                 skip_dims,
                 drop_path_rate=0.0,
                 use_checkpoint=False,
                 upsample_method="nearest"):
        super().__init__()

        # minimal retained attributes
        self.use_checkpoint = use_checkpoint
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
            input_feats = in_chans if level == 0 else shallow_feats[level-1]
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

        self.transition_layers = nn.ModuleList()
        for level in range(num_levels):
            self.transition_layers.append(
                TransitionLayer(in_dim=self.shallow_feats[level], out_dim=self.embed_dims[level])
            )

        self.LX_blocks = nn.ModuleList()
        cur = 0
        for level in range(num_levels):
            blocks = nn.ModuleList()
            dp_rates = [x for x in np.linspace(0, drop_path_rate, self.num_blks[level] * self.blk_layers[level])]
            dp = dp_rates[cur: cur + self.blk_layers[level]]
            for _ in range(self.num_blks[level]):
                blocks.append(
                    Group(input_size=context_sizes[level],
                          patch_size=patch_sizes[level],
                          dim=self.embed_dims[level],
                          skip_dim=skip_dims[level],
                          depth=self.blk_layers[level],
                          window_size=self.attn_window_sizes[level],
                          num_heads=self.num_heads,
                          mlp_ratio=self.mlp_ratio,
                          dp_rates=dp,
                          prev_dim=self.embed_dims[level-1] if level > 0 else None)
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


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / 10 ** 9 if torch.cuda.is_available() else 0

    batch_size = 1
    img_size = 64
    print("image size: ", img_size)
    x = torch.randn((batch_size, 1, img_size, img_size, img_size)).to(device)
    B, C, H, W, D = x.shape

    # small config for quick run
    context_sizes = [64, 48, 32]
    num_levels = len(context_sizes)
    shallow_feats = [32, 64, 128]
    pre_up_feats = [64, 64]
    num_blks = [1, 2, 3]
    blk_layers = [6, 6, 6]
    patch_sizes = [4, 3, 2]
    skip_dims = [60, 60, 60]
    embed_dims = [120, 180, 240]
    num_heads = 4
    mlp_ratio = 4
    attn_window_sizes = [8, 8, 8]
    up_factor = 2
    input_size = (H, W, D)
    use_checkpoint = True

    # TODO: Add merge of features from previous network levels.
    net = MTVNet_monai(input_size=input_size,
                  up_factor=up_factor,
                  num_levels=num_levels,
                  context_sizes=context_sizes,
                  num_blks=num_blks,
                  blk_layers=blk_layers,
                  in_chans=1,
                  shallow_feats=shallow_feats,
                  pre_up_feats=pre_up_feats,
                  embed_dims=embed_dims,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  attn_window_sizes=attn_window_sizes,
                  patch_sizes=patch_sizes,
                  skip_dims=skip_dims,
                  drop_path_rate=0.1,
                  use_checkpoint=use_checkpoint,
                  upsample_method="nearest",).to(device)
    net.train()

    print("Number of parameters", numel(net, only_trainable=True))

    loss_func = nn.MSELoss()

    x_hr = torch.randn((batch_size, 1, up_factor * context_sizes[-1],
                              up_factor * context_sizes[-1],
                              up_factor * context_sizes[-1])).to(device)

    start = time.time()
    out = net(x)
    loss = loss_func(out, x_hr)
    stop = time.time()
    print("Time elapsed:", stop - start)

    print("output shape:", out.shape)
    loss.backward()

    if torch.cuda.is_available():
        max_memory_reserved = torch.cuda.max_memory_reserved()
        print("Maximum memory reserved: %0.3f Gb / %0.3f Gb" % (max_memory_reserved / 10 ** 9, total_gpu_mem))
    print("Done")
