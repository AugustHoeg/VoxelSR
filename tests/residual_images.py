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

    model_names = ["RCAN", "HAT", "EDDSR", "mDCSRN", "MFER", "SuperFormer", "RRDBNet3D", "MTVNet"]
    base_dir = "../../downloaded_data/VoDaSuRe/Visual_comparisons/"

    dataset = "VoDaSuRe_DOWN"
    use_registered = True  # Change to False for downsampled data

    if use_registered:
        group_dir = "HR0_REG0"  # Change for downsampled vs. registered data
        LR_title = r"Registered LR ($\times 4$)"  # "Downsampled"
    else:
        group_dir = "HR0_HR2"  # Change for downsampled vs. registered data
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

    #for hej in [9, 10, 11, 18, 19, 20, 27, 28, 29, 36, 37, 38, 45, 46, 47, 54, 55, 56]: # [18, 19, 20, 27, 28, 29, 36, 37, 38, 45, 46, 47, 54, 55, 56]:
    #img_idx_list = [37, 29, 19]
    img_idx = 0
    row, col = 3, len(model_names)
    show_HR_as_large_img = False

    large_img_size = 400
    #large_img_location = (210, 190)
    large_img_location = (200, 200)
    red_box_coords = (50, 150)
    red_box_size = 256

    large_image_string = r"VoDaSuRe ($\times 4$)"
    use_other_string = True
    plot_metrics = False

    fig = plt.figure(figsize=(11, 4 if plot_metrics else 4), constrained_layout=True)
    #fig.suptitle(large_image_string, fontsize=26)
    gs = fig.add_gridspec(row, col)

    comp_dict = get_comparison_dict(image_paths, img_idx, model_names, large_img_size, large_img_location)

    #subplot_text = [["ArSSR", "EDDSR", "MFER", "mDCSRN", "SuperFormer", "RRDBNet3D", "MTVNet"]]

    for i in range(row):

        if i == 0:
            for j in range(col):
                ax = fig.add_subplot(gs[i, j])
                box_name = model_names[j]

                img_box = comp_dict[box_name]['H']

                tv_image = img_box.transpose(2, 0, 1).astype(np.float32)
                tv = total_variation(tv_image, mode="L2")

                norm_val = np.max(comp_dict['MTVNet']['H'])

                img_box = img_box / norm_val
                img, _ = crop_image_at_location(img_box, red_box_size, red_box_coords)
                ax.imshow(img.transpose(1, 0, 2))
                ax.text(0.5, -0.04, f"TV: {tv:.2f}", ha='center', va='top', fontsize=10, transform=ax.transAxes)

                ax.set_xticks([])
                ax.set_yticks([])
        elif i == 1:
            for j in range(col):
                ax = fig.add_subplot(gs[i, j])
                box_name = model_names[j]

                img_box = comp_dict[box_name]['E']
                tv_image = img_box.transpose(2, 0, 1).astype(np.float32)
                tv = total_variation(tv_image, mode="L2")

                norm_val = np.max(comp_dict['MTVNet']['H'])

                img_box = img_box / norm_val
                img, _ = crop_image_at_location(img_box, red_box_size, red_box_coords)
                ax.imshow(img.transpose(1, 0, 2))
                ax.text(0.5, -0.04, f"{model_names[j]} TV: {tv:0.2f}", ha='center', va='top', fontsize=10, transform=ax.transAxes)

                ax.set_xticks([])
                ax.set_yticks([])

        elif i == 2:
            for j in range(col):
                ax = fig.add_subplot(gs[i, j])
                box_name = model_names[j]

                img_box = np.abs(comp_dict[box_name]['H'] - comp_dict[box_name]['E']).astype(np.float32)
                img_box = (img_box - np.min(img_box)) / (img_box.max() - img_box.min())

                img_box = img_box
                img, _ = crop_image_at_location(img_box, red_box_size, red_box_coords)
                ax.imshow(img.transpose(1, 0, 2))
                ax.text(0.5, -0.04, "RES", ha='center', va='top', fontsize=10, transform=ax.transAxes)

                ax.set_xticks([])
                ax.set_yticks([])

    save_path = f"../figures/{dataset}_residual_{img_idx}.pdf"
    fig.savefig(save_path, format="pdf")
    plt.show()



