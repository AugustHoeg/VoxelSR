# --------------------------------------------------------
# MTVNet (cleaned)
# Written by August Høeg
# --------------------------------------------------------
import time
import torch
import torch.nn as nn

from models.models_3D import SRBlock3D
from utils.utils_3D_image import numel


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
            else:
                x = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, Hp, Wp, Dp, C)
        else:
            x = x.flatten(2).transpose(1, 2)
        return x


class MTVNet(nn.Module):
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
                 patch_sizes,
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
                             out_format="image")
            )

        # dropout (kept for API)
        self.pos_drop = nn.Dropout(p=0.0)

        # Lightweight LX blocks: use Identity placeholders so the model runs.
        # Replace nn.Identity() with your real block implementation.
        self.LX_blocks = nn.ModuleList()
        for level in range(num_levels):
            blocks = nn.ModuleList()
            for _ in range(self.num_blks[level]):
                blocks.append(
                    nn.Identity()  # placeholder block
                )
            self.LX_blocks.append(blocks)

        # convolution to map final features (image space)
        self.conv_image = nn.Conv3d(in_channels=shallow_feats[-1],
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

        LX_img = x  # image-space features (B, C_img, H, W, D)
        for level in range(0, self.num_levels):
            # SFE
            LX_img = self.sfe_blks[level](LX_img)  # (B, shallow_feats[level], H, W, D)

            # Patch embedding -> image format (B, H_p, W_p, D_p, C_emb)
            LX_x = self.patch_embedding_blks[level](LX_img)

            # Pass through LX blocks (placeholder identity blocks)
            for i, blk in enumerate(self.LX_blocks[level]):
                LX_x = blk(LX_x)  # currently nn.Identity(); replace with real block

            # Optionally crop next input image for the next level
            if level < self.num_levels - 1:
                LX_img = self.crop_next(LX_img, level + 1)

        # final_feats: use highest-level image features
        final_feats = self.lrelu(self.conv_image(LX_img))  # image-space conv + act

        # Long skip connection
        out = LX_img + final_feats

        # Upsampling
        if self.up_factor == 2:
            out = self.SR0(out)
        elif self.up_factor == 3:
            out = self.SR0(out)
        elif self.up_factor == 4:
            out = self.SR0(out)
            out = self.SR1(out)

        out = self.conv_last(self.lrelu(self.HRconv(out)))
        return out


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / 10 ** 9 if torch.cuda.is_available() else 0

    batch_size = 1
    img_size = 32
    print("image size: ", img_size)
    x = torch.randn((batch_size, 1, img_size, img_size, img_size)).to(device)
    B, C, H, W, D = x.shape

    # small config for quick run
    context_sizes = [32]
    num_levels = len(context_sizes)
    shallow_feats = [32]
    pre_up_feats = [32, 32]
    num_blks = [1]
    blk_layers = [1]
    patch_sizes = [2]
    embed_dims = [32]
    up_factor = 2
    input_size = (H, W, D)

    net = MTVNet(input_size=input_size, up_factor=up_factor, num_levels=num_levels,
                 context_sizes=context_sizes, num_blks=num_blks, blk_layers=blk_layers,
                 in_chans=1, shallow_feats=shallow_feats, pre_up_feats=pre_up_feats,
                 embed_dims=embed_dims, patch_sizes=patch_sizes, use_checkpoint=False,
                 upsample_method="nearest",).to(device)
    net.train()

    print("Number of parameters", numel(net, only_trainable=True))

    loss_func = nn.MSELoss()

    x_hr = torch.randn((batch_size, 1, up_factor * context_sizes[-1],
                              up_factor * context_sizes[-1],
                              up_factor * context_sizes[-1])).to(device)

    start = time.time()
    august_out = net(x)
    loss = loss_func(august_out, x_hr)
    stop = time.time()
    print("Time elapsed:", stop - start)

    print("AugustNet output shape:", august_out.shape)
    loss.backward()

    if torch.cuda.is_available():
        max_memory_reserved = torch.cuda.max_memory_reserved()
        print("Maximum memory reserved: %0.3f Gb / %0.3f Gb" % (max_memory_reserved / 10 ** 9, total_gpu_mem))
    print("Done")
