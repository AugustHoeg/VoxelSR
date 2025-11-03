import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from skimage import exposure


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


if __name__ == '__main__':
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["text.usetex"] = True

    row, col = 1, 8
    model_names = ["ArSSR", "EDDSR", "MFER", "mDCSRN", "SuperFormer", "RRDBNet3D", "MTVNet"]
    dataset = "Synthetic_2022_QIM_52_Bone_4x/"
    base_dir = "../downloaded_data/paper_comparisons"

    model_dirs = [
        f"{base_dir}/ArSSR/",
        f"{base_dir}/EDDSR/",
        f"{base_dir}/MFER/",
        f"{base_dir}/mDCSRN/",
        f"{base_dir}/SuperFormer/",
        f"{base_dir}/RRDBNet3D/",
        f"{base_dir}/AugustNet/"
    ]

    model_dirs = [os.path.join(d, dataset, "full_slice_comparisons/*") for d in model_dirs]
    image_paths = [glob.glob(path) for path in model_dirs]

    img_idx_list = range(0, 1)
    show_HR_as_large_img = False

    for img_idx in img_idx_list:
        large_img_size = 400
        large_img_location = (210, 190)
        red_box_coords = (50, 150)
        red_box_size = 64

        comp_dict = get_comparison_dict(image_paths, img_idx, model_names, large_img_size, large_img_location)

        titles = [
            r"HCP 1200 ($\times$4)",
            r"IXI ($\times 4$)",
            r"FACTS-Synth ($\times 4$)",
            r"FACTS-Real ($\times 4$)"
        ]
        large_image_string = titles[2]
        use_other_string = True
        plot_metrics = False

        fig = plt.figure(figsize=(16, 2 if plot_metrics else 4), constrained_layout=True)
        fig.suptitle(large_image_string, fontsize=26)
        gs = fig.add_gridspec(row, col)

        if use_other_string:
            header = "GT   \\hspace{2.3cm}   LR"
        else:
            header = "GT vs LR"

        subplot_text = [["HR crop", "ArSSR", "EDDSR", "MFER", "mDCSRN", "SuperFormer", "RRDBNet3D", "MTVNet"]]

        img_box_order = [["HR", "ArSSR", "EDDSR", "MFER", "mDCSRN", "SuperFormer", "RRDBNet3D", "MTVNet"]]

        for i in range(row):
            for j in range(col):
                ax = fig.add_subplot(gs[i, j])
                box_name = img_box_order[i][j]

                if box_name == "HR":
                    img_box = comp_dict["MTVNet"]['H']
                elif box_name == "LR":
                    img_box = comp_dict["MTVNet"]['L']
                else:
                    img_box = comp_dict[box_name]['E']

                norm_val = np.max(comp_dict['MTVNet']['H'])

                if (i == 0 and j == 0) and show_HR_as_large_img:
                    large_img = comp_dict['MTVNet']['H'] / norm_val
                    ax.imshow(large_img)
                    rect = patches.Rectangle(red_box_coords, red_box_size, red_box_size, linewidth=1.5, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    ax.text(0.5, -0.025, header, ha='center', va='top', fontsize=20, transform=ax.transAxes)
                else:
                    if (i == 0 and j == 0):
                        img_hr = comp_dict["MTVNet"]['H'] / norm_val
                        img_lr = comp_dict["MTVNet"]['L'] / norm_val
                        img_hr, _ = crop_image_at_location(img_hr, red_box_size, red_box_coords)
                        img_lr, _ = crop_image_at_location(img_lr, red_box_size, red_box_coords)

                        img = np.zeros_like(img_hr)
                        img[:img_hr.shape[0] // 2, :, :] = img_hr[:img_hr.shape[0] // 2, :, :]
                        img[img_hr.shape[0] // 2:, :, :] = img_lr[img_hr.shape[0] // 2:, :, :]
                        ax.imshow(img.transpose(1, 0, 2))
                        ax.vlines(red_box_size // 2 - 0.5, ymin=0, ymax=red_box_size - 0.5, colors="red")
                        ax.text(0.5, -0.025, header, ha='center', va='top', fontsize=24, transform=ax.transAxes)
                    else:
                        img_box = img_box / norm_val
                        img, _ = crop_image_at_location(img_box, red_box_size, red_box_coords)
                        ax.imshow(img.transpose(1, 0, 2))
                        ax.text(0.5, -0.03, subplot_text[i][j], ha='center', va='top', fontsize=24, transform=ax.transAxes)

                ax.set_xticks([])
                ax.set_yticks([])

        save_path = f"../downloaded_data/paper_comparisons/saved_figures/{dataset[:-1]}_without_large_img_idx_{img_idx}.pdf"
        fig.savefig(save_path, format="pdf")
        plt.show()
