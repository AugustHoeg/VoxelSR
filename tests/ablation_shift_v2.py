import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
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

    img_idx = 2 # 0 to 4
    model_dirs = ["../../downloaded_data/VoDaSuRe/Visual_comparisons/RRDBNet3D/VoDaSuRe_shift/HR0_HR2/*.png"]

    image_paths = [glob.glob(path) for path in model_dirs]

    large_img_size = 1920
    large_img_location = (1920 // 2 - large_img_size // 2 +300, 1920 // 2 - large_img_size // 2 -350)

    show_HR_as_large_img = False

    comp_dict = get_comparison_dict(image_paths, img_idx, model_names=["RRDBNet3D"],
                                    crop_size=large_img_size, crop_location=large_img_location)

    row, col = 1, 4
    show_HR_as_large_img = False

    red_box_size = 276 # 512
    #red_box_coords = (1920//2 - red_box_size//2, 1920//2 - red_box_size//2)
    red_box_coords = (1920 // 2 - red_box_size // 2 + 600, 1920 // 2 - red_box_size // 2 - 500)
    #red_box_coords = (1600 // 2 - red_box_size // 2 + 250, 1440 // 2 - red_box_size // 2 +290)

    large_image_string = r"VoDaSuRe ($\times 4$)"
    use_other_string = True
    plot_metrics = False

    fig = plt.figure(figsize=(7*1.5, 2.1*1.5), constrained_layout=True)
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


    # Load the non-shifted downsampled image
    img_idx = 20
    model_dirs = ["../../downloaded_data/VoDaSuRe/Visual_comparisons/RRDBNet3D/VoDaSuRe_DOWN/HR0_HR2/*.png"]

    image_paths = [glob.glob(path) for path in model_dirs]

    comp_dict = get_comparison_dict(image_paths, img_idx, model_names=["RRDBNet3D"],
                                    crop_size=large_img_size, crop_location=large_img_location)

    SR_no_shift = comp_dict["RRDBNet3D"]['E']
    SR_no_shift = np.array(Image.fromarray(SR_no_shift).convert('L')).reshape(SR_no_shift.shape[0],
                                                                              SR_no_shift.shape[1], 1)

    LR_no_shift = comp_dict["RRDBNet3D"]['L']
    LR_no_shift = np.array(Image.fromarray(LR_no_shift).convert('L')).reshape(LR_no_shift.shape[0],
                                                                              LR_no_shift.shape[1], 1)

    HR_no_shift = comp_dict["RRDBNet3D"]['H']
    HR_no_shift = np.array(Image.fromarray(HR_no_shift).convert('L')).reshape(HR_no_shift.shape[0],
                                                                              HR_no_shift.shape[1], 1)

    for j in range(col):

        name = "Elm"

        # diff = np.abs(HR - LR_up)

        #norm_val = np.max(img)
        #img = img / norm_val

        ax = fig.add_subplot(gs[0, j])

        if j == 0:  # show full slice

            LR_crop = crop_image_at_location(LR, red_box_size, red_box_coords, return_location=False)
            HR_crop = crop_image_at_location(HR, red_box_size, red_box_coords, return_location=False)

            # convert to float
            LR_crop = LR_crop.astype(np.float32)
            HR_crop = HR_crop.astype(np.float32)

            difference_abs = np.abs(LR_crop - HR_crop)
            difference = LR_crop - HR_crop
            difference_full = LR.astype(np.float32) - HR.astype(np.float32)
            difference_full = (difference_full - np.min(difference_full)) / (np.max(difference_full) - np.min(difference_full)) * 255.0

            # scale difference  to 0-255
            difference = (difference - np.min(difference)) / (np.max(difference) - np.min(difference)) * 255.0
            #ax.imshow(difference, cmap='gray', vmin=0, vmax=255)

            ax.imshow(difference_full, cmap='gray')
            ax.text(0.5, -0.02, "Difference image:\n HR, LR w. shift", ha='center', va='top', fontsize=16, transform=ax.transAxes)

            rect = patches.Rectangle(red_box_coords, red_box_size, red_box_size, linewidth=1.5, edgecolor='r',
                                     facecolor='none')
            ax.add_patch(rect)

            # 2. Create an inset axes
            # [x, y, width, height] in relative axes coordinates (0 to 1)
            # This places the inset at 70% x, 70% y, taking up 25% width/height
            ax_inset = ax.inset_axes([0.5, 0.0, 0.50, 0.50])

            # 3. Display the small image in the inset
            ax_inset.imshow(difference, cmap='gray', vmin=0, vmax=255)

            for spine in ax_inset.spines.values():
                spine.set_edgecolor('red')  # Set color to red
                spine.set_linewidth(1.5)  # Make it thicker so it's visible

            ax_inset.set_xticks([])
            ax_inset.set_yticks([])

            # ... existing code ...
            ax_inset.set_yticks([])

            # --- NEW CODE STARTS HERE ---

            # 1. Define coordinates for the bottom corners of the Red Rectangle
            # (x, y) is the top-left of the rect.
            # Since images have origin='upper', y + height is the visual "bottom".
            rect_bottom_left = (red_box_coords[0], red_box_coords[1] + red_box_size)
            rect_bottom_right = (red_box_coords[0] + red_box_size, red_box_coords[1] + red_box_size)

            # 2. Create the first connector (Left side)
            # Connects Rect Bottom-Left (Data Coords) -> Inset Top-Left (Axes Fraction 0,1)
            con1 = ConnectionPatch(
                xyA=rect_bottom_left, coordsA=ax.transData,
                xyB=(0, 1), coordsB=ax_inset.transAxes,
                axesA=ax, axesB=ax_inset,
                color="red", linewidth=1.5
            )

            # 3. Create the second connector (Right side)
            # Connects Rect Bottom-Right (Data Coords) -> Inset Top-Right (Axes Fraction 1,1)
            con2 = ConnectionPatch(
                xyA=rect_bottom_right, coordsA=ax.transData,
                xyB=(1, 1), coordsB=ax_inset.transAxes,
                axesA=ax, axesB=ax_inset,
                color="red", linewidth=1.5
            )

            # 4. Add the artists to the figure
            fig.add_artist(con1)
            fig.add_artist(con2)


        elif j == 1:  # show crop region

            SR_crop, _ = crop_image_at_location(SR, red_box_size, red_box_coords)

            ax.imshow(SR_crop, cmap='gray', vmin=0, vmax=255)
            text = r"RRDBNet3D ($4 \times$)"
            ax.text(0.5, -0.02, f"Downsampled w. shift \n {text}", ha='center', va='top', fontsize=16, transform=ax.transAxes)

        elif j == 2:  # show crop region

            SR_no_shift_crop, _ = crop_image_at_location(SR_no_shift, red_box_size, red_box_coords)

            ax.imshow(SR_no_shift_crop, cmap='gray', vmin=0, vmax=255)
            text = r"RRDBNet3D ($4 \times$)"
            ax.text(0.5, -0.02, f"Downsampled w.o shift \n {text}", ha='center', va='top', fontsize=16, transform=ax.transAxes)


        elif j == 3:  # show difference image

            model_dirs = ["../../downloaded_data/VoDaSuRe/Visual_comparisons/RRDBNet3D/VoDaSuRe/HR0_REG0/*.png"]

            image_paths = [glob.glob(path) for path in model_dirs]

            comp_dict = get_comparison_dict(image_paths, img_idx, model_names=["RRDBNet3D"],
                                            crop_size=large_img_size, crop_location=large_img_location)

            SR_REG = comp_dict["RRDBNet3D"]['E']
            SR_REG = np.array(Image.fromarray(SR_REG).convert('L')).reshape(SR_REG.shape[0], SR_REG.shape[1], 1)

            SR_REG_crop, _ = crop_image_at_location(SR_REG, red_box_size, red_box_coords)

            ax.imshow(SR_REG_crop, cmap='gray', vmin=0, vmax=255)
            text = r"RRDBNet3D ($4 \times$)"
            ax.text(0.5, -0.02, f"Registered w.o shift \n {text}", ha='center', va='top', fontsize=16,
                    transform=ax.transAxes)

        ax.set_xticks([])
        ax.set_yticks([])

    datetime = np.datetime64('now')
    time = str(datetime).replace(":", "-").replace(" ", "_")
    save_path = f"../figures/rebuttal_registration_{img_idx}.pdf"
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()
