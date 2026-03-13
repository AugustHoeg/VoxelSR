import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from PIL import Image
from skimage import exposure
import torch
import lpips

from utils.fourier_ring_correlation import radial_power_spectrum_2d

def crop_image_at_location(image: np.ndarray, size: int, location: tuple, return_location=True):
    """
    Crop an image to a specified size from a given location, adjusting the location
    if necessary to keep the crop within the bounds of the image.

    Parameters:test_gridspec_3x3_test.py
        image (np.ndarray): Input image array of shape (H, W, C).
        size (int): Size of the square crop (size x size).
        location (tuple): (y, x) coordinates for the top-left corner of the crop.
        return_location (bool): Whether to also return the adjusted crop location.

    Returns:
        np.ndarray: Cropped image of shape (size, size, C).
        tuple (optional): Adjusted (y, x) coordinates.
    """
    H, W, _ = image.shape
    y, x = location

    crop_h, crop_w = min(size, H), min(size, W)

    # Adjust crop if it exceeds image bounds
    if y + crop_h > H:
        y = H - crop_h
    if x + crop_w > W:
        x = W - crop_w

    y, x = max(0, y), max(0, x)
    cropped = image[x:x + crop_w, y:y + crop_h]

    return (cropped, (y, x)) if return_location else cropped


def get_comparison_dict(image_paths, img_idx, model_names, crop_size, crop_location):
    """
    Load comparison images for each model and crop corresponding L, E, and H regions.

    Returns:
        dict: {model_name: {'L': ..., 'E': ..., 'H': ...}}
    """
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
            sections[key] = crop_image_at_location(sections[key], crop_size, crop_location, return_location=False)

        comp_dict[model_name] = sections

    return comp_dict

def total_variation(img, mode="L2"):
    c, h, w = img.shape
    if mode == "sum_of_squares":
        tv_x = np.pow(img[:, 1:, :] - img[:, :-1, :], 2).sum()
        tv_y = np.pow(img[:, :, 1:] - img[:, :, :-1], 2).sum()
        return (tv_x + tv_y)/(c*h*w)
    elif mode == "L2":
        tv_x = np.pow(img[:, 1:, :] - img[:, :-1, :], 2)
        tv_y = np.pow(img[:, :, 1:] - img[:, :, :-1], 2)
        return np.sum(np.sqrt(tv_x.flatten() + tv_y.flatten()))/(c*h*w)


if __name__ == '__main__':
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["text.usetex"] = True

    base_dir = "../../downloaded_data/VoDaSuRe/Visual_comparisons/"

    row = 2
    col = 5

    large_image_string = r"examples_LPIPS" # r"Registered"

    fig = plt.figure(figsize=(10*1.5, 4.2*1.5), constrained_layout=True)
    gs = gridspec.GridSpec(row, col, figure=fig)

    large_img_size = 512
    red_box_coords = (150, 150)
    red_box_size = 256

    img_idx = 18

    for i in range(row):
        for j in range(col):

            large_img_location = (650, 650)

            if j == 0:
                dataset = "VoDaSuRe_DOWN"
                model = "RRDBNet3D"
                group_dir = "HR0_HR2"  # Change for downsampled vs. registered data
                LR_title = r"HR reference"  # "Downsampled"

                model_dirs = [
                    f"{base_dir}/{model}/{dataset}/{group_dir}/"
                ]
                model_dirs = [os.path.join(d, "*.png") for d in model_dirs]
                image_paths = [glob.glob(path) for path in model_dirs]
                comp_dict = get_comparison_dict(image_paths, img_idx, [model], large_img_size, large_img_location)

                img_box = comp_dict[model]['H']

                ax = fig.add_subplot(gs[i, j])

                if i == 0:
                    # Make image plot
                    norm_val = np.max(img_box)
                    img_box = img_box / norm_val

                    img, _ = crop_image_at_location(img_box, red_box_size, red_box_coords)

                    ax.imshow(img.transpose(1, 0, 2))

                    ax.set_title(f"{LR_title}", fontsize=16, pad=0., y=1.03)
                    # ax.text(0.5, -0.02, f"Hej", ha='center', va='top', transform=ax.transAxes, fontsize=16)
                else:
                    # plot power spectrum image
                    img, _ = crop_image_at_location(img_box, red_box_size, red_box_coords)
                    img_gray = np.sum(img.astype(np.float64), axis=2)
                    radial_profile_HR, power_spec = radial_power_spectrum_2d(torch.from_numpy(img_gray), apply_window=True)
                    radial_profile_HR /= radial_profile_HR[0]

                    log2_power_spec = np.log2(power_spec + 1e-9)
                    print("min, max log2 power spec: ", log2_power_spec.min(), log2_power_spec.max())

                    # Clip negative values to zero
                    log2_power_spec = np.clip(log2_power_spec, a_min=-10, a_max=None)

                    im = ax.imshow(log2_power_spec, cmap='twilight_shifted') # gray
                    # ax.semilogy(radial_profile, label="Radial profile")

                ax.set_xticks([])
                ax.set_yticks([])

            elif j == 1:
                dataset = "VoDaSuRe_DOWN"
                model = "RRDBNet3D"
                group_dir = "HR0_HR2"  # Change for downsampled vs. registered data
                LR_title = r"Downsampled LR ($\times 4$)"  # "Downsampled"

                model_dirs = [
                    f"{base_dir}/{model}/{dataset}/{group_dir}/"
                ]
                model_dirs = [os.path.join(d, "*.png") for d in model_dirs]
                image_paths = [glob.glob(path) for path in model_dirs]
                comp_dict = get_comparison_dict(image_paths, img_idx, [model], large_img_size, large_img_location)

                img_box = comp_dict[model]['E']

                ax = fig.add_subplot(gs[i, j])

                if i == 0:
                    # Make image plot
                    norm_val = np.max(img_box)
                    img_box = img_box / norm_val

                    img, _ = crop_image_at_location(img_box, red_box_size, red_box_coords)

                    ax.imshow(img.transpose(1, 0, 2))

                    ax.set_title(f"{LR_title}", fontsize=16, pad=0., y=1.03)
                    # ax.text(0.5, -0.02, f"Hej", ha='center', va='top', transform=ax.transAxes, fontsize=16)
                else:
                    # plot power spectrum image
                    img, _ = crop_image_at_location(img_box, red_box_size, red_box_coords)
                    img_gray = np.sum(img.astype(np.float64), axis=2)
                    radial_profile_DOWN, power_spec = radial_power_spectrum_2d(torch.from_numpy(img_gray), apply_window=True)
                    radial_profile_DOWN /= radial_profile_DOWN[0]

                    log2_power_spec = np.log2(power_spec + 1e-9)
                    print("min, max log2 power spec: ", log2_power_spec.min(), log2_power_spec.max())

                    # Clip negative values to zero
                    log2_power_spec = np.clip(log2_power_spec, a_min=-10, a_max=None)

                    im = ax.imshow(log2_power_spec, cmap='twilight_shifted')
                    # ax.semilogy(radial_profile, label="Radial profile")

                ax.set_xticks([])
                ax.set_yticks([])

            if j == 2:

                dataset = "VoDaSuRe"
                model = "RRDBNet3D"
                group_dir = "HR0_REG0"  # Change for downsampled vs. registered data
                LR_title = r"Registered LR ($\times 4$)"  # "Downsampled"

                model_dirs = [
                    f"{base_dir}/{model}/{dataset}/{group_dir}/"
                ]
                model_dirs = [os.path.join(d, "*.png") for d in model_dirs]
                image_paths = [glob.glob(path) for path in model_dirs]
                comp_dict = get_comparison_dict(image_paths, img_idx, [model], large_img_size, large_img_location)

                img_box = comp_dict[model]['E']

                ax = fig.add_subplot(gs[i, j])

                if i == 0:
                    # Make image plot
                    norm_val = np.max(img_box)
                    img_box = img_box / norm_val

                    img, _ = crop_image_at_location(img_box, red_box_size, red_box_coords)

                    ax.imshow(img.transpose(1, 0, 2))

                    ax.set_title(f"{LR_title}", fontsize=16, pad=0., y=1.03)
                    # ax.text(0.5, -0.02, f"Hej", ha='center', va='top', transform=ax.transAxes, fontsize=16)
                else:
                    # plot power spectrum image
                    img, _ = crop_image_at_location(img_box, red_box_size, red_box_coords)
                    img_gray = np.sum(img.astype(np.float64), axis=2)
                    radial_profile_REG, power_spec = radial_power_spectrum_2d(torch.from_numpy(img_gray), apply_window=True)
                    radial_profile_REG /= radial_profile_REG[0]

                    log2_power_spec = np.log2(power_spec + 1e-9)
                    print("min, max log2 power spec: ", log2_power_spec.min(), log2_power_spec.max())

                    # Clip negative values to zero
                    log2_power_spec = np.clip(log2_power_spec, a_min=-10, a_max=None)

                    im = ax.imshow(log2_power_spec, cmap='twilight_shifted')
                    # ax.semilogy(radial_profile, label="Radial profile")

                    from mpl_toolkits.axes_grid1 import make_axes_locatable

                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('bottom', size='5%', pad=0.05)
                    fig.colorbar(im, cax=cax, orientation='horizontal')

                    #divider = make_axes_locatable(ax)
                    #cax = divider.new_horizontal(pack_start=False, size="5%", pad=0.05)
                    #fig = ax.get_figure()
                    #fig.add_axes(cax)
                    #cbar = plt.colorbar(im, cax=cax)
                    #cbar.set_label(r'$Log_2$ magnitude', rotation=270, labelpad=10)

                ax.set_xticks([])
                ax.set_yticks([])

            elif j == 3 and i == 1:

                ax = fig.add_subplot(gs[:, 3:col])

                freq = np.linspace(0, 1, len(radial_profile_HR))

                ax.semilogy(freq, radial_profile_HR, label="HR reference")
                ax.semilogy(freq, radial_profile_DOWN, label="Downsampled LR")
                ax.semilogy(freq, radial_profile_REG, label="Registered LR")

                ax.set_title("Radially averaged power spectrum", fontsize=16)

                ax.set_xlabel("Normalized spatial frequency", fontsize=12)
                ax.set_ylabel("Power spectrum magnitude", fontsize=12)

                ax.legend()

                ax.yaxis.tick_right()


    #plt.subplots_adjust(hspace=0.45)
    save_path = f"../figures/power_spectrum_{img_idx}_{red_box_size}.pdf"
    #plt.tight_layout(h_pad=0.1, w_pad=0.1)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()



