
import os
import torch
import numpy as np
import tifffile

from scipy import ndimage
import matplotlib.pyplot as plt

from utils.fourier_ring_correlation import frc, smooth, find_intersect, plot_frc

import torchio.transforms as tiotransforms
#import torch.nn.functional as F


def plot_images():

    from utils.utils_2D_image import ImageComparisonTool2D as comparison_tool
    comp_tool = comparison_tool(patch_size_hr=slice_H.shape[1:],
                                upscaling_methods=["tio_nearest", "tio_linear"],
                                unnorm=False,
                                div_max=False,
                                out_dtype=np.uint8)

    grid_image = comp_tool.get_comparison_image(img_dict={'H': slice_H, 'E': slice_E, 'L': slice_L})
    height, width = grid_image.shape[:2]
    plt.figure(figsize=(4 * width / 100, 4 * height / 100), dpi=100)
    plt.imshow(grid_image, vmin=0, vmax=255)
    plt.title("Comparison image")
    plt.xticks([])
    plt.yticks([])
    plt.savefig("figures/comparison_image.png", bbox_inches='tight', pad_inches=0.1)

def FRC(img1, img2, thl_criterion='1bit', label="SR vs HR", filename_prefix="FRC"):
    # Calculate FRC for HR and SR
    corr, thl = frc(img1, img2, thl_criterion)
    smoothed = smooth(corr, 5)
    intersect = find_intersect(smoothed, thl)
    plot_frc(corr, smoothed, thl, intersect[0], p_eff, p_unit='µm', thl_label='1-bit threshold', label=label, filename_prefix=filename_prefix)


if __name__ == "__main__":

    # Load volumes
    # path = "/work3/s173944/Python/venv_srgan/3D_datasets/datasets/LUND/bamboo/results/RRDBNet3D/"
    path = "../Vedrana_master_project/3D_datasets/datasets/LUND/bamboo/results/RRDBNet3D/"

    img_H = tifffile.imread(os.path.join(path, "full_HR_sample_0.tiff"))
    img_E = tifffile.imread(os.path.join(path, "full_SR_sample_0.tiff"))
    img_L = tifffile.imread(os.path.join(path, "full_LR_sample_0.tiff"))

    # print shapes
    print(f"Shape of HR image: {img_H.shape}")
    print(f"Shape of SR image: {img_E.shape}")
    print(f"Shape of LR image: {img_L.shape}")

    up_factor = 4

    # pick ROI
    roi_loc = [220, 120, 680]
    roi_size = [700, 700]

    # define roi in LR space
    roi_loc_lr = [val // up_factor for val in roi_loc]
    roi_size_lr = [val // up_factor for val in roi_size]

    slice_H = img_H[roi_loc[0],
                    roi_loc[1]:roi_loc[1] + roi_size[0],
                    roi_loc[2]:roi_loc[2] + roi_size[1]]

    slice_E = img_E[roi_loc[0],
                    roi_loc[1]:roi_loc[1] + roi_size[0],
                    roi_loc[2]:roi_loc[2] + roi_size[1]]

    slice_L = img_L[roi_loc_lr[0],
                    roi_loc_lr[1]:roi_loc_lr[1] + roi_size_lr[0],
                    roi_loc_lr[2]:roi_loc_lr[2] + roi_size_lr[1]]

    # Convert to float32
    slice_H = slice_H.astype(np.float32)
    slice_E = slice_E.astype(np.float32)
    slice_L = slice_L.astype(np.float32)

    if False:  # For bone data
        path = "../Vedrana_master_project/3D_datasets/datasets/danmax/binning_bone/results/"
        slice_H = tifffile.imread(os.path.join(path, "bone_2_H.tiff"))[2220: -2220:, 2220: -2220:]
        slice_E = tifffile.imread(os.path.join(path, "bone_2_E.tiff"))[2220: -2220:, 2220: -2220:]
        slice_L = tifffile.imread(os.path.join(path, "bone_2_L.tiff"))[555: -555:, 555: -555:]

        # from skimage.registration import phase_cross_correlation
        # shift, error, diffphase = phase_cross_correlation(slice_E, slice_H)
        import cv2
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        (cc, warp_matrix) = cv2.findTransformECC(slice_H, slice_E, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria)
        slice_E = cv2.warpAffine(slice_E, warp_matrix, (slice_H.shape[1], slice_H.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        slice_E = slice_E[40: -40, 40: -40]
        slice_H = slice_H[40: -40, 40: -40]
        slice_L = slice_L[10: -10, 10: -10]

        #slice_H = slice_H - slice_H.min() / (slice_H.max() - slice_H.min())
        #slice_E = slice_E - slice_E.min() / (slice_E.max() - slice_E.min())
        #slice_L = slice_L - slice_L.min() / (slice_L.max() - slice_L.min())

    upscale_volume = False
    if upscale_volume:
        scaler = tiotransforms.Resize(target_shape=img_H.shape, image_interpolation='BSPLINE')
        img_L_up = scaler(torch.from_numpy(img_L).unsqueeze(0)).squeeze().numpy()  # Upscale LR image to match HR image shape
        slice_L_up = img_L_up[roi_loc[0],
                           roi_loc[1]:roi_loc[1] + roi_size[0],
                           roi_loc[2]:roi_loc[2] + roi_size[1]]
        slice_L_up = slice_L_up.astype(np.float32)

    else:
        # Interpolate using torch interpolation / ndimage zoom
        # slice_L_up = F.interpolate(torch.from_numpy(slice_L), scale_factor=up_factor, align_corners=True, mode='bilinear').numpy()
        slice_L_up = ndimage.zoom(slice_L, up_factor, order=2, mode='grid-constant')
        slice_L_up[slice_L_up < 0] = 0  # Ensure no negative values after interpolation
        slice_L_up[slice_L_up > 1] = 1  # Ensure no values above 1 after interpolation
        #slice_L_up = slice_L_up - slice_L_up.min() / (slice_L_up.max() - slice_L_up.min())

    p_eff = 0.275  # Effective pixel size in micrometers

    # Run FRC
    FRC(slice_H, slice_E, thl_criterion='1bit', label="SR vs HR", filename_prefix="FRC_SR_vs_HR")
    FRC(slice_H, slice_L_up, thl_criterion='1bit', label="LR vs HR", filename_prefix="FRC_LR_up_vs_HR")

    # Plot images
    plt.figure(figsize=(12, 4), dpi=600)
    plt.subplot(1, 3, 1)
    plt.imshow(slice_H, cmap='gray', vmin=0.0, vmax=1.0)
    plt.axis('off')
    plt.title("HR slice")
    plt.subplot(1, 3, 2)
    plt.imshow(slice_E, cmap='gray', vmin=0.0, vmax=1.0)
    plt.axis('off')
    plt.title("SR slice")
    plt.subplot(1, 3, 3)
    plt.imshow(slice_L_up, cmap='gray', vmin=0.0, vmax=1.0)
    plt.axis('off')
    plt.title("Upscaled LR slice")
    plt.tight_layout()
    plt.savefig("figures/slice_comparison.png", bbox_inches='tight', pad_inches=0.1)

    print("Done")