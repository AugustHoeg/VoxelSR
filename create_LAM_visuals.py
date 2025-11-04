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
    """
    H, W, C = image.shape
    h, w = location

    crop_height, crop_width = min(size, H), min(size, W)

    # Adjust coordinates if crop would go out of bounds
    if h + crop_height > H:
        h = H - crop_height
    if w + crop_width > W:
        w = W - crop_width
    h, w = max(0, h), max(0, w)

    cropped_image = image[w:w + crop_width, h:h + crop_height]
    if return_location:
        return cropped_image, (h, w)
    else:
        return cropped_image


def get_comparison_dict_LAM(image_paths, img_idx, model_names, large_img_size, large_img_location):
    """
    Load and crop the LAM images for each model.
    """
    comp_dict = {}
    for model_name in model_names:
        LAM_image = [s for s in image_paths if model_name in s]
        if len(LAM_image) == 0:
            continue
        comp_img = np.array(Image.open(LAM_image[0]))
        LAM_image_crop = crop_image_at_location(comp_img, large_img_size, large_img_location, return_location=False)
        comp_dict[model_name] = LAM_image_crop
    return comp_dict


if __name__ == '__main__':
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["text.usetex"] = True

    # Configuration
    model_names = ["RCAN", "HAT", "EDDSR", "ArSSR", "MFER", "mDCSRN", "SuperFormer", "RRDBNet3D", "MTVNet"]
    #dataset = "C:/Users/aulho/OneDrive - Danmarks Tekniske Universitet/Dokumenter/Github/downloaded_data/paper_comparisons/LAM/HCP_1200_cube_027_win48_h40-w40-d40_new/"

    cube_no = "027"
    dataset_name = "CTSpine1K"
    dataset = f"LAM_3d/Results/{dataset_name}_cube_{cube_no}_win48_h38-w38-d38_new/"

    row, col = 1, len(model_names) + 1
    large_img_size = 369
    large_img_location = (0, 0)
    red_box_coords = (138, 138)
    red_box_size = 92
    show_HR_as_large_img = False

    # Load LAM images
    image_paths = glob.glob(dataset + "*_full_log_blend.png")
    comp_dict = get_comparison_dict_LAM(image_paths, 0, model_names, large_img_size, large_img_location)

    # Prepare subplot labels
    DI_values = {
        "HAT": 8.48, "RCAN": 7.73, "ArSSR": 8.37, "EDDSR": 6.65,
        "MTVNet": 39.62, "MFER": 15.15, "mDCSRN": 15.49,
        "RRDBNet3D": 27.52, "SuperFormer": 13.92,
    }

    subplot_text = ["HR crop"]
    for model in model_names:
        if model == "MTVNet":
            subplot_text.append(f"{model}\n DI: $\\mathbf{{{DI_values[model]:.2f}}}$")
        else:
            subplot_text.append(f"{model}\n DI: {DI_values[model]:.2f}")

    # Create figure
    fig = plt.figure(figsize=(30, 4.0), constrained_layout=True)
    gs = fig.add_gridspec(row, col)
    img_box_order = [["HR"] + model_names]

    # Plot images
    for i in range(row):
        for j in range(col):
            ax = fig.add_subplot(gs[i, j])
            box_name = img_box_order[i][j]

            # Leftmost image: input LR
            if i == 0 and j == 0:
                lr_image_path = glob.glob(dataset + "lr_full*.png")[0]
                img_box = np.array(Image.open(lr_image_path)).astype(np.float32) / 255.0
                ax.imshow(img_box, cmap="Reds")
            else:
                img_box = comp_dict.get(box_name)
                if img_box is None:
                    ax.axis("off")
                    continue
                img_box = img_box.astype(np.float32) / 255.0
                ax.imshow(img_box, cmap="Reds")

                rect = patches.Rectangle(
                    red_box_coords, red_box_size, red_box_size,
                    linewidth=1.5, edgecolor='b', facecolor='none'
                )
                ax.add_patch(rect)

            label_text = "Input LR" if (i == 0 and j == 0) else subplot_text[j]
            ax.text(0.5, -0.03, label_text, ha='center', va='top', fontsize=36, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

    # Save figure
    fig.savefig(dataset[:-1] + "LAM.pdf", format="pdf")
    plt.show()
