
import os
import torch
import numpy as np
import tifffile

from utils.fourier_ring_correlation import frc, smooth, find_intersect, plot_frc
import torch.nn.functional as F

if __name__ == "__main__":

    # Load volumes
    path = "/work3/s173944/Python/venv_srgan/3D_datasets/datasets/LUND/bamboo/results/RRDBNet3D/"

    img_H = tifffile.imread(os.path.join(path, "full_HR_sample_0.tiff"))
    img_E = tifffile.imread(os.path.join(path, "full_SR_sample_0.tiff"))
    img_L = tifffile.imread(os.path.join(path, "full_LR_sample_0.tiff"))

    up_factor = 4

    # pick ROI
    roi_loc = [250, 700, 120]
    roi_size = [1, 700, 700]

    # define roi in LR space
    roi_loc_lr = [val // up_factor for val in roi_loc]
    roi_size_lr = [val // up_factor for val in roi_size]

    slice_H = img_H[roi_loc[0]:roi_loc[0] + roi_size[0],
                    roi_loc[1]:roi_loc[1] + roi_size[1],
                    roi_loc[2]:roi_loc[2] + roi_size[2]]

    slice_E = img_E[roi_loc[0]:roi_loc[0] + roi_size[0],
                    roi_loc[1]:roi_loc[1] + roi_size[1],
                    roi_loc[2]:roi_loc[2] + roi_size[2]]

    slice_L = img_L[roi_loc_lr[0]:roi_loc_lr[0] + roi_size_lr[0],
                    roi_loc_lr[1]:roi_loc_lr[1] + roi_size_lr[1],
                    roi_loc_lr[2]:roi_loc_lr[2] + roi_size_lr[2]]

    # Convert to float32
    slice_H = slice_H.astype(np.float32)
    slice_E = slice_E.astype(np.float32)
    slice_L = slice_L.astype(np.float32)

    # Interpolate using torch interpolation
    slice_L_up = F.interpolate(slice_L, scale_factor=up_factor, align_corners=True, mode='bilinear')

    p_eff = 0.5  # Effective pixel size in micrometers

    # Calculate FRC for HR and SR
    corr, thl = frc(slice_H, slice_E, thl_criterion='1bit')
    smoothed = smooth(corr, 5)
    intersect = find_intersect(smoothed, thl)
    plot_frc(corr, smoothed, thl, intersect[-1], p_eff, p_unit='µm', thl_label='1-bit threshold', filename_prefix="FRC_SR_vs_HR")

    # Calculate FRC for HR and LR
    corr, thl = frc(slice_H, slice_L, thl_criterion='1bit')
    smoothed = smooth(corr, 5)
    intersect = find_intersect(smoothed, thl)
    plot_frc(corr, smoothed, thl, intersect[-1], p_eff, p_unit='µm', thl_label='1-bit threshold', filename_prefix="FRC_LR_vs_HR")

    print("Done")