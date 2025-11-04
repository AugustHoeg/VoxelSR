import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.patches as patches
from PIL import Image
import glob
from skimage import exposure


def crop_image_at_location(image: np.ndarray, size: int, location: tuple, return_location=True):
    """
    Crop an image to a specified size from a given location, adjusting the location
    if necessary to keep the crop within the bounds of the image.

    Parameters:
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

    model_names = ["RCAN", "HAT", "EDDSR", "mDCSRN", "MFER", "SuperFormer", "RRDBNet3D", "MTVNet"]
    base_dir = "../downloaded_data/VoDaSuRe/Visual_comparisons/"

    dataset = "VoDaSuRe_DOWN"
    use_registered = True  # Change to False for downsampled data

    if use_registered:
        group_dir = "HR0_REG0" # Change for downsampled vs. registered data
        LR_title = r"Registered LR ($\times 4$)"  # "Downsampled"
    else:
        group_dir = "HR0_HR2" # Change for downsampled vs. registered data
        LR_title = r"Downsampled LR ($\times 4$)"  # "Downsampled"

    model_dirs = [
        f"{base_dir}/RCAN/{dataset}/{group_dir}/",
        f"{base_dir}/HAT/{dataset}/{group_dir}/",
        f"{base_dir}/EDDSR/{dataset}/{group_dir}/",
        f"{base_dir}/mDCSRN/{dataset}/{group_dir}/",
        f"{base_dir}/MFER/{dataset}/{group_dir}/",
        f"{base_dir}/SuperFormer/{dataset}/{group_dir}/",
        f"{base_dir}/RRDBNet3D/{dataset}/{group_dir}/",
        f"{base_dir}/MTVNet/{dataset}/{group_dir}/"
    ]

    model_dirs = [os.path.join(d, "*.png") for d in model_dirs]
    image_paths = [glob.glob(path) for path in model_dirs]

    idx = 5
    img_idx_start = 9 * idx   # 0 for Cardboard, 19 for Elm,
    show_HR_as_large_img = False

    names = ["Cardboard", "Bamboo", "Elm", "Larch", "Femur 01", "MDF", "Ox bone", "Oak", "Cypress"]
    img_name = names[idx]
    HR_title = rf"{dataset.split('_')[0]}: {img_name}"

    large_img_size = 1920//3
    large_img_location = (600, 300)
    red_box_size = 256
    red_box_coords = (large_img_size//2-red_box_size//2, large_img_size//2-red_box_size//2)

    for img_idx in range(img_idx_start, img_idx_start + 3):

        comp_dict = get_comparison_dict(image_paths, img_idx, model_names, large_img_size, large_img_location)

        fig = plt.figure(figsize=(10, 7.0), constrained_layout=True)

        # Main structure: narrow left column, wide right grid
        gs_main = gridspec.GridSpec(
            1, 2,
            figure=fig,
            width_ratios=[1, 2.15],  # right much wider
            wspace=0.01,             # tiny space between left and right
            hspace=0.01
        )

        # Left column: two large stacked images
        gs_left = gridspec.GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=gs_main[0, 0],
            hspace=0.08,
            wspace=0.08
        )

        ax_full = fig.add_subplot(gs_left[0])
        ax_zoom = fig.add_subplot(gs_left[1])

        # Right column: 3×3 tightly packed grid
        gs_right = gridspec.GridSpecFromSubplotSpec(
            3, 3,
            subplot_spec=gs_main[0, 1],
            wspace=0.01,
            hspace=0.01
        )
        axes_right = [fig.add_subplot(gs_right[i, j]) for i in range(3) for j in range(3)]

        # -----------------------------
        # Example content placeholders
        # -----------------------------

        ax_full.imshow(comp_dict[model_names[0]]['H'], cmap='gray')
        #ax_full.set_title("Full HR region", fontsize=10)
        ax_full.set_xticks([])
        ax_full.set_yticks([])
        ax_full.text(0.5, -0.04, f"{HR_title}", ha='center', va='top', fontsize=16, transform=ax_full.transAxes)

        rect = patches.Rectangle(red_box_coords, red_box_size, red_box_size, linewidth=1, edgecolor='r', facecolor='none')
        rect.set_linewidth(1.5)
        ax_full.add_patch(rect)

        ax_zoom.imshow(comp_dict[model_names[0]]['L'], cmap='gray')
        #ax_zoom.set_title("Zoomed region", fontsize=10)
        ax_zoom.set_xticks([])
        ax_zoom.set_yticks([])
        ax_zoom.text(0.5, -0.04, f"{LR_title}", ha='center', va='top', fontsize=16, transform=ax_zoom.transAxes)
        rect = patches.Rectangle(red_box_coords, red_box_size, red_box_size, linewidth=1, edgecolor='b', facecolor='none')
        rect.set_linewidth(1.5)
        ax_zoom.add_patch(rect)

        # Add a red rectangle to the large image
        #rect = patches.Rectangle(red_box_coords/4, red_box_size, red_box_size, linewidth=1, edgecolor='r', facecolor='none')
        #rect.set_linewidth(1.5)
        #ax_zoom.add_patch(rect)

        for i, ax in enumerate(axes_right):
            if i == 0: # HR
                img_box = comp_dict[model_names[0]]['H']
                text = "HR crop"

                # Set spine color for the remaining subplots
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(1.5)
            else:
                img_box = comp_dict[model_names[i-1]]['E']
                text = model_names[i-1]
            img, red_box_coords = crop_image_at_location(img_box, red_box_size, location=red_box_coords, return_location=True)

            ax.imshow(img, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, -0.05, text, ha='center', va='top', fontsize=16, transform=ax.transAxes)

        # Make figure as compact as possible
        fig.set_constrained_layout_pads(
            w_pad=0.06,
            h_pad=0.06,
            wspace=0.0,
            hspace=0.0
        )

        plt.show()
