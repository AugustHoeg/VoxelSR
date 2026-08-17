import torch
import torch.nn as nn

# ── 3D axial RoPE ──
# Adapted from https://github.com/YuZheng9/DVAR/blob/main/models/rope.py (2D → 3D).

def _grid_coords(shape):
    """(dz,dy,dx) → (dz*dy*dx, 3) float coords in raster order; columns are (z, y, x)."""
    dz, dy, dx = shape
    zz, yy, xx = torch.meshgrid(
        torch.arange(dz), torch.arange(dy), torch.arange(dx), indexing="ij"
    )
    return torch.stack([zz.reshape(-1), yy.reshape(-1), xx.reshape(-1)], dim=-1).float()


def compute_axial_cis(dim, hr_shape, lr_shape=None, theta=100.0,
                      norm_coeff_x=1.0, norm_coeff_y=1.0, norm_coeff_z=1.0):
    """3D axial rotary frequencies as complex phasors.

    Coordinate frame:
      * HR tokens use their integer grid coordinates.
      * LR (prefix) tokens are mapped onto the HR grid via per-axis upscale, so an
        LR voxel sits at the centre of the HR voxels it covers (aligned frame).

    Returns:
        freqs_cis: (T, 3 * (dim // 6)) complex, T = (N_lr) + N_hr.
    """
    n = dim // 6
    base = torch.arange(0, dim, 6)[:n].float() / dim  # (n,)
    freqs_x = norm_coeff_x / (theta ** base)
    freqs_y = norm_coeff_y / (theta ** base)
    freqs_z = norm_coeff_z / (theta ** base)

    hr_coords = _grid_coords(hr_shape)  # (N_hr, 3)
    if lr_shape is not None:
        lr_coords = _grid_coords(lr_shape)  # (N_lr, 3)
        scale = torch.tensor([hr_shape[a] / lr_shape[a] for a in range(3)])
        lr_coords = lr_coords * scale + (scale - 1) / 2  # HR-scaled, centred
        coords = torch.cat([lr_coords, hr_coords], dim=0)  # (T, 3)
    else:
        coords = hr_coords

    t_z, t_y, t_x = coords[:, 0], coords[:, 1], coords[:, 2]  # unpack (z,y,x)

    freqs_x = torch.outer(t_x, freqs_x)  # (T, n)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_z = torch.outer(t_z, freqs_z)

    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    freqs_cis_z = torch.polar(torch.ones_like(freqs_z), freqs_z)
    return torch.cat([freqs_cis_x, freqs_cis_y, freqs_cis_z], dim=-1)  # (T, 3n) complex


def apply_rotary_emb(x, freqs_cis):
    """Rotate the last dim of x by complex phasors freqs_cis.

    x:         (B, H, T, rot_dim) real   — rot_dim == 2 * freqs_cis.shape[-1]
    freqs_cis: (T, rot_dim // 2) complex
    """
    with torch.autocast(device_type=x.device.type, enabled=False):
        xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))  # (B,H,T,rot_dim/2)
        out = torch.view_as_real(xc * freqs_cis[None, None]).flatten(-2)     # (B,H,T,rot_dim)
    return out.type_as(x)


class Rope3D(nn.Module):
    """3D axial RoPE using precomputed complex freqs_cis (DVAR-style).

    Builds the phasor table for the full [ LR_prefix | HR ] sequence once. Only the
    first `rot_dim = 6 * (head_dim // 6)` channels of each head are rotated; any
    remainder (head_dim not divisible by 6) is passed through unrotated. `offset`
    selects the HR sub-table for the unconditional (HR-only) body pass.
    """

    def __init__(self, head_dim, hr_shape, lr_shape=None, theta=100.0,
                 norm_coeffs=(1.0, 1.0, 1.0)):
        super().__init__()
        assert head_dim // 6 > 0, f"head_dim={head_dim} too small for 3D RoPE (need >= 6)"
        self.rot_dim = 6 * (head_dim // 6)
        freqs_cis = compute_axial_cis(head_dim, hr_shape, lr_shape, theta, *norm_coeffs)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)  # (T, rot_dim/2) complex

    def forward(self, x, offset=0):
        """x: (B, H, T, head_dim) → same, first rot_dim channels rotated."""
        T = x.shape[2]
        freqs = self.freqs_cis[offset:offset + T]                       # (T, rot_dim/2)
        rot = apply_rotary_emb(x[..., :self.rot_dim], freqs)
        if self.rot_dim < x.shape[-1]:
            rot = torch.cat([rot, x[..., self.rot_dim:]], dim=-1)
        return rot
