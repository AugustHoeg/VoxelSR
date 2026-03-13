import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import torch

from utils.fourier_ring_correlation import radial_power_spectrum_2d


def crop_image_at_location(image: np.ndarray, size: int, location: tuple):
    """Crop square region from image while respecting boundaries."""
    H, W, _ = image.shape
    y, x = location

    crop_h, crop_w = min(size, H), min(size, W)

    if y + crop_h > H:
        y = H - crop_h
    if x + crop_w > W:
        x = W - crop_w

    y, x = max(0, y), max(0, x)

    cropped = image[y:y + crop_h, x:x + crop_w]

    return cropped


def get_comparison_dict(image_paths, img_idx, model_names, crop_size, crop_location):
    comp_dict = {}

    for model_name, paths in zip(model_names, image_paths):
        comp_img = np.array(Image.open(paths[img_idx]))
        H, W, _ = comp_img.shape

        sections = {
            'L': comp_img[:, 0 * W // 3:1 * W // 3, :],
            'E': comp_img[:, 1 * W // 3:2 * W // 3, :],
            'H': comp_img[:, 2 * W // 3:3 * W // 3, :],
        }

        for key in sections:
            sections[key] = crop_image_at_location(sections[key], crop_size, crop_location)

        comp_dict[model_name] = sections

    return comp_dict


def load_panel(base_dir, dataset, group_dir, model, section, img_idx, crop_size, crop_loc):
    """Helper function removing repeated loading code."""

    model_dirs = [f"{base_dir}/{model}/{dataset}/{group_dir}/"]
    model_dirs = [os.path.join(d, "*.png") for d in model_dirs]

    image_paths = [glob.glob(path) for path in model_dirs]

    comp_dict = get_comparison_dict(image_paths, img_idx, [model], crop_size, crop_loc)

    return comp_dict[model][section]


if __name__ == '__main__':

    spectrum_axes = []

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["text.usetex"] = True

    base_dir = "../../downloaded_data/VoDaSuRe/Visual_comparisons/"

    fig = plt.figure(figsize=(13.5, 6), constrained_layout=True)
    gs = gridspec.GridSpec(2, 5, figure=fig)

    large_img_size = 512
    red_box_coords = (150, 150)
    red_box_size = 256

    img_idx = 27
    large_img_location = (650, 650)

    panels = [
        ("VoDaSuRe_DOWN", "HR0_HR2", "RRDBNet3D", "H", r"HR reference"),
        ("VoDaSuRe_DOWN", "HR0_HR2", "RRDBNet3D", "E", r"Downsampled LR ($\times 4$)"),
        ("VoDaSuRe", "HR0_REG0", "RRDBNet3D", "E", r"Registered LR ($\times 4$)")
    ]

    radial_profiles = []
    im = None

    for j, (dataset, group_dir, model, section, title) in enumerate(panels):

        img_box = load_panel(
            base_dir,
            dataset,
            group_dir,
            model,
            section,
            img_idx,
            large_img_size,
            large_img_location
        )

        img_box = img_box / np.max(img_box)
        img = crop_image_at_location(img_box, red_box_size, red_box_coords)

        # ----- IMAGE ROW -----
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(img.transpose(1, 0, 2))
        ax.set_title(title, fontsize=16, y=1.01)
        ax.set_xticks([])
        ax.set_yticks([])

        # ----- SPECTRUM ROW -----
        ax = fig.add_subplot(gs[1, j])
        spectrum_axes.append(ax)

        img_gray = np.sum(img.astype(np.float64), axis=2)

        radial_profile, power_spec = radial_power_spectrum_2d(
            torch.from_numpy(img_gray),
            apply_window=True
        )

        radial_profile = np.array(radial_profile)
        radial_profiles.append(radial_profile)

        log2_power_spec = np.log2(power_spec + 1e-9)
        log2_power_spec = np.clip(log2_power_spec, a_min=-8, a_max=None)

        im = ax.imshow(log2_power_spec, cmap='twilight_shifted', vmax=28, vmin=-8)

        ax.set_xticks([])
        ax.set_yticks([])

    # ----- RADIAL PROFILE PLOT -----
    ax = fig.add_subplot(gs[:, 3:])

    freq = np.linspace(0, 1, len(radial_profiles[0]))

    labels = [
        "HR reference",
        "Downsampled LR",
        "Registered LR"
    ]

    for profile, label in zip(radial_profiles, labels):
        profile = profile / profile[0]
        ax.semilogy(freq, profile, label=label, linewidth=2)

    ax.set_title("Radially averaged power spectrum", fontsize=16)

    ax.set_xlabel("Normalized spatial frequency", fontsize=16)
    ax.set_ylabel("Normalized power spectrum", fontsize=16)

    # add legend with no frame
    ax.legend(fontsize=14, frameon=False)
    ax.yaxis.tick_right()
    ax.tick_params(labelsize=14)

    # ----- PRECISE COLORBAR PLACEMENT -----

    # Get positions of the three spectrum axes
    pos_left = spectrum_axes[0].get_position()
    pos_right = spectrum_axes[-1].get_position()

    # Define colorbar position
    cbar_left = pos_left.x0 - 0.11
    cbar_width = pos_right.x1 - pos_left.x0 + 0.095
    cbar_bottom = pos_left.y0 - 0.075  # small offset below images
    cbar_height = 0.010  # thin colorbar

    # Create axis for colorbar
    cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])

    # Create colorbar
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')

    # ----- uniformly spaced ticks -----
    ticks = [-8, 4, 16, 28]

    cbar.set_ticks(ticks)
    cbar.ax.tick_params(labelsize=14, length=2)

    cbar.set_label(r'$\log_2$ magnitude', fontsize=16)

    # Move label slightly up and left
    cbar.ax.xaxis.set_label_coords(0.50, -1.8)

    save_path = f"../figures/power_spectrum_{img_idx}_{red_box_size}.pdf"

    fig.savefig(save_path, format="pdf", bbox_inches="tight")

    plt.show()