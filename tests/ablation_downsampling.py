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

    model_names = ["RRDBNet3D", "MTVNet"]
    base_dir = "../../downloaded_data/VoDaSuRe/Visual_comparisons/"

    #img_idx_list = [0, 1, 2, 9, 10, 11, 18, 19] # For medical datasets:
    row = 1
    col = 4

    #datasets = ["CTSpine1K", "LITS", "LIDC-IDRI"]

    plot_prediction = True
    use_registered = True  # Change to False for downsampled data

    large_image_string = r"Ablation_DOWN" # r"Registered"
    use_other_string = True
    plot_metrics = False

    #fig = plt.figure(figsize=(1.5 * 11, 1.5 * 4 if plot_metrics else 1.5 * 4), constrained_layout=True)
    fig = plt.figure(figsize=(7*1.5, 2.2*1.5 if plot_metrics else 2.05*1.5), constrained_layout=True)
    # fig.suptitle(large_image_string, fontsize=26)
    gs = fig.add_gridspec(row, col)

    for i in range(row):
        for j in range(col):

            if j % 2 == 0:
                img_idx = 0
                dataset = "VoDaSuRe_ablation"
                group_dir = "HR1_REG1"  # Change for downsampled vs. registered data
                LR_title = r"Registered LR ($\times 4$)"  # "Downsampled"

                #large_img_size = 400
                #large_img_location = (520, 600)
                #red_box_coords = (150, 150)
                #red_box_size = 256

                large_img_size = 512
                large_img_location = (520, 600)
                red_box_coords = (150, 150)
                red_box_size = 128

            else:
                img_idx = 19
                dataset = "VoDaSuRe"
                group_dir = "HR0_REG0"  # Change for downsampled vs. registered data
                LR_title = r"Registered LR ($\times 4$)"  # "Downsampled"

                #large_img_size = 1920
                #large_img_location = (520 * 4, 600 * 4)
                #red_box_coords = (150, 150)
                #red_box_size = 512

                large_img_size = 512
                large_img_location = (520 * 2 + 5, 600 * 2 - 150 - 3)
                red_box_coords = (150, 150)
                red_box_size = 256


            model_dirs = [
                f"{base_dir}/MTVNet/{dataset}/{group_dir}/",
                f"{base_dir}/RRDBNet3D/{dataset}/{group_dir}/"
            ]

            model_dirs = [os.path.join(d, "*.png") for d in model_dirs]
            image_paths = [glob.glob(path) for path in model_dirs]

            #for hej in [9, 10, 11, 18, 19, 20, 27, 28, 29, 36, 37, 38, 45, 46, 47, 54, 55, 56]: # [18, 19, 20, 27, 28, 29, 36, 37, 38, 45, 46, 47, 54, 55, 56]:
            #img_idx_list = [37, 29, 19]
            show_HR_as_large_img = False

            comp_dict = get_comparison_dict(image_paths, img_idx, model_names, large_img_size, large_img_location)

            ax = fig.add_subplot(gs[i, j])
            if j >= 2:
                box_name = "RRDBNet3D"
            else:
                box_name = "MTVNet" # model_names[j]

            if plot_prediction:
                img_box = comp_dict[box_name]['E']
            else:
                img_box = comp_dict[box_name]['H']

            tv_image = img_box.transpose(2, 0, 1).astype(np.float32)
            tv = total_variation(tv_image, mode="L2")

            norm_val = np.max(comp_dict[box_name]['H'])

            img_box = img_box / norm_val
            img, _ = crop_image_at_location(img_box, red_box_size, red_box_coords)
            ax.imshow(img.transpose(1, 0, 2))
            #ax.text(0.5, -0.04, f"{large_image_string}, TV: {tv:.2f}", ha='center', va='top', fontsize=10, transform=ax.transAxes)
            if j % 2 == 0:
                ax.set_title(f"W/ downsampling", fontsize=17, pad=0., y=1.03)
            else:
                ax.set_title(f"W/O downsampling", fontsize=17, pad=0., y=1.03)
            ax.text(0.5, -0.02, f"{box_name}, TV: {tv:.2f}", ha='center', va='top', fontsize=16,
                    transform=ax.transAxes)

            ax.set_xticks([])
            ax.set_yticks([])

    #plt.subplots_adjust(hspace=0.45)
    save_path = f"../figures/{large_image_string}_ablation_downsampling_{img_idx}_{red_box_size}.pdf"
    #plt.tight_layout(h_pad=0.1, w_pad=0.1)
    fig.savefig(save_path, format="pdf")
    plt.show()



