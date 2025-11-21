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


def minmax_scaler(image, vmin=0, vmax=1.0):

    # in-place scaling using np operations
    image_min = np.nanmin(image)
    image_max = np.nanmax(image)
    image -= image_min
    image /= (image_max - image_min)
    image *= (vmax - vmin)
    image += vmin
    return image


def clip_percentile(image, lower=1.0, upper=99.0, vmin=0, vmax=65535):

    low = np.percentile(image, lower)
    high = np.percentile(image, upper)

    if high <= low:  # avoid divide-by-zero
        return np.zeros_like(image, dtype=np.float32)

    np.clip(image, low, high, out=image)

    image = minmax_scaler(image, vmin, vmax)

    return image


if __name__ == '__main__':
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["text.usetex"] = True

    img_idx = 15  # 0 to 4
    base_dir = "../../downloaded_data/VoDaSuRe/supplementary/examples/"

    #sample_dirs = [os.path.join(base_dir, name) for name in sample_names]

    axes = ["D", "H", "W"]

    HR_paths_D = glob.glob(os.path.join(base_dir, "*HR_D.png"))
    HR_paths_H = glob.glob(os.path.join(base_dir, "*HR_H.png"))
    HR_paths_W = glob.glob(os.path.join(base_dir, "*HR_W.png"))
    LR_paths_D = glob.glob(os.path.join(base_dir, "*LR_D.png"))
    LR_paths_H = glob.glob(os.path.join(base_dir, "*LR_H.png"))
    LR_paths_W = glob.glob(os.path.join(base_dir, "*LR_W.png"))
    REG_paths_D = glob.glob(os.path.join(base_dir, "*REG_D.png"))
    REG_paths_H = glob.glob(os.path.join(base_dir, "*REG_H.png"))
    REG_paths_W = glob.glob(os.path.join(base_dir, "*REG_W.png"))

    paths = {
        "HR": (HR_paths_D, HR_paths_H, HR_paths_W),
        "REG": (REG_paths_D, REG_paths_H, REG_paths_W),
        "LR": (LR_paths_D, LR_paths_H, LR_paths_W)
    }

    row, col = 1, 9
    show_HR_as_large_img = False

    red_box_size = 256
    #red_box_coords = (1920//2 - red_box_size//2, 1920//2 - red_box_size//2)
    #red_box_coords = (1920 // 2 - red_box_size // 2 +300, 1920 // 2 - red_box_size // 2 -350)
    #red_box_coords = (1600 // 2 - red_box_size // 2 + 250, 1440 // 2 - red_box_size // 2 + 290)

    red_box_coords = (1000, 950)  # bamboo
    red_box_coords = (1000, 950)  # cardboard
    red_box_coords = (1000, 500)  # Cypress
    red_box_coords = (1000, 500)  # Elm
    red_box_coords = (200, 300)  # Femur 01, no norm
    red_box_coords = (600, 400)  # Femur 15, no norm
    red_box_coords = (600, 400)  # Femur 21, no norm
    red_box_coords = (300, 400)  # Femur 74, no norm
    red_box_coords = (1000, 500)  # Larch
    red_box_coords = (1000, 500)  # MDF, only LR norm
    red_box_coords = (1000, 500)  # Oak, only LR norm
    red_box_coords = (600, 650)  # Ox bone, only LR norm
    red_box_coords = (450, 500)  # Vertebrae A, no norm
    red_box_coords = (450, 500)  # Vertebrae B, no norm
    red_box_coords = (450, 500)  # Vertebrae C, no norm
    red_box_coords = (450, 500)  # Vertebrae D, no norm

    large_image_string = r"VoDaSuRe ($\times 4$)"
    use_other_string = True
    plot_metrics = False

    fig = plt.figure(figsize=(18, 2 if plot_metrics else 2), constrained_layout=True)
    #fig.suptitle(large_image_string, fontsize=26)
    gs = fig.add_gridspec(row, col)

    #HR_D = np.array(Image.open(HR_paths[img_idx]))
    #LR = np.array(Image.open(LR_paths[img_idx]))
    #LR_up = np.array(Image.open(LR_up_paths[img_idx]))

    #HR = HR[:, 160:1600]
    #LR_up = LR_up[:, 160:1600]

    sample_name = os.path.basename(paths['HR'][0][img_idx]).split("_")[0]

    c = 0
    for key in paths:
        for i in range(3):
            img = np.array(Image.open(paths[key][i][img_idx])).astype(np.float32)
            # normalize to +- 3 std
            #img_rescale = clip_percentile(img, lower=1.0, upper=99.0, vmin=0, vmax=65535)

            ax = fig.add_subplot(gs[0, c])
            if key == "LR":
                red_box_coords = (250, 400)
            img_crop, _ = crop_image_at_location(img.reshape([*img.shape, 1]), red_box_size, red_box_coords)

            if key == "LR":
                pass
                #img_mean = np.mean(img_crop)
                #img_std = np.std(img_crop)
                #img_crop = np.clip(img_crop, img_mean - 3 * img_std, img_mean + 3 * img_std)
                #img_crop = exposure.rescale_intensity(img_crop, in_range=(img_mean - 3 * img_std, img_mean + 3 * img_std), out_range=(0, 65535))

            ax.imshow(img_crop, cmap='gray', vmin=0, vmax=65535)

            #ax.text(0.5, -0.02, f"{key}: Depth", ha='center', va='top', fontsize=18, transform=ax.transAxes)

            c += 1

            ax.set_xticks([])
            ax.set_yticks([])

    datetime = np.datetime64('now')
    time = str(datetime).replace(":", "-").replace(" ", "_")
    save_path = f"../figures/supplementary_examples_{sample_name}_{img_idx}.pdf"
    fig.savefig(save_path, format="pdf")
    plt.show()
