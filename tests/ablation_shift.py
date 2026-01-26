import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from skimage.util import compare_images
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

    img_idx = 10 # 0 to 4
    model_dirs = ["../../downloaded_data/VoDaSuRe/Visual_comparisons/RRDBNet3D/VoDaSuRe_DOWN/HR0_HR2/*.png"]

    image_paths = [glob.glob(path) for path in model_dirs]

    large_img_size = 1920
    large_img_location = (1920 // 2 - large_img_size // 2 +300, 1920 // 2 - large_img_size // 2 -350)

    show_HR_as_large_img = False

    comp_dict = get_comparison_dict(image_paths, img_idx, model_names=["RRDBNet3D"],
                                    crop_size=large_img_size, crop_location=large_img_location)

    row, col = 1, 5
    show_HR_as_large_img = False

    red_box_size = 512
    #red_box_coords = (1920//2 - red_box_size//2, 1920//2 - red_box_size//2)
    red_box_coords = (1920 // 2 - red_box_size // 2 +300, 1920 // 2 - red_box_size // 2 -350)
    #red_box_coords = (1600 // 2 - red_box_size // 2 + 250, 1440 // 2 - red_box_size // 2 +290)

    large_image_string = r"VoDaSuRe ($\times 4$)"
    use_other_string = True
    plot_metrics = False

    fig = plt.figure(figsize=(19, 3.9 if plot_metrics else 3.9), constrained_layout=True)
    #fig.suptitle(large_image_string, fontsize=26)
    gs = fig.add_gridspec(row, col)

    HR = comp_dict["RRDBNet3D"]['H']
    LR = comp_dict["RRDBNet3D"]['L']
    SR = comp_dict["RRDBNet3D"]['E']

    # convert to grayscale
    HR = np.array(Image.fromarray(HR).convert('L')).reshape(HR.shape[0], HR.shape[1], 1)
    LR = np.array(Image.fromarray(LR).convert('L')).reshape(LR.shape[0], LR.shape[1], 1)
    SR = np.array(Image.fromarray(SR).convert('L')).reshape(SR.shape[0], SR.shape[1], 1)

    #HR = HR[:, 160:1600]
    #LR_up = LR_up[:, 160:1600]

    for j in range(col):

        name = "Elm"

        # diff = np.abs(HR - LR_up)

        #norm_val = np.max(img)
        #img = img / norm_val

        ax = fig.add_subplot(gs[0, j])

        if j == 0:  # show full slice
            img = HR
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)
            rect = patches.Rectangle(red_box_coords, red_box_size, red_box_size, linewidth=1.5, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(0.5, -0.02, f"VoDaSuRe - {name}", ha='center', va='top', fontsize=18, transform=ax.transAxes)

        elif j == 1:  # show crop region
            img = HR
            img, _ = crop_image_at_location(img, red_box_size, red_box_coords)
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)
            ax.text(0.5, -0.02, "HR crop", ha='center', va='top', fontsize=18, transform=ax.transAxes)

        elif j == 2:  # show crop region
            img = LR
            img, _ = crop_image_at_location(img, red_box_size, red_box_coords)
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)
            ax.text(0.5, -0.02, r"Shifted LR crop $(4\times)$", ha='center', va='top', fontsize=18, transform=ax.transAxes)

        elif j == 3:  # show difference image

            HR_crop, _ = crop_image_at_location(HR, red_box_size, red_box_coords)
            LR_crop, _ = crop_image_at_location(LR, red_box_size, red_box_coords)

            HR_crop = HR_crop.astype(np.float32)
            LR_crop = LR_crop.astype(np.float32)

            ax.imshow(np.abs(LR_crop - HR_crop), cmap='gray', vmin=0, vmax=255)
            ax.text(0.5, -0.02, "Absolute difference", ha='center', va='top', fontsize=18, transform=ax.transAxes)


        else:  # show prediction image

            SR_crop, _ = crop_image_at_location(SR, red_box_size, red_box_coords)

            ax.imshow(SR_crop, cmap='gray', vmin=0, vmax=255)
            ax.text(0.5, -0.02, "Prediction", ha='center', va='top', fontsize=18, transform=ax.transAxes)

        ax.set_xticks([])
        ax.set_yticks([])

    datetime = np.datetime64('now')
    time = str(datetime).replace(":", "-").replace(" ", "_")
    save_path = f"../figures/rebuttal_registration_{img_idx}.pdf"
    fig.savefig(save_path, format="pdf")
    plt.show()
