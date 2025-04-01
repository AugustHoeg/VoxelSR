import numpy as np
import h5py
import math
import matplotlib.pyplot as plt

from utils.utils_HDF5 import crop_slices_hdf5, match_slice_histograms

if __name__ == "__main__":

    project_path = "/dtu/3d-imaging-center/projects/2024_DANFIX_130_ExtremeCT/raw_data_extern/"
    scan_path = project_path + "2024031208/brain_1_20kev_20x_16bits_30sdd/"

    # Define the input and output paths
    bin1x1_path = scan_path + "bin1x1/scan-7014-7034_recon.h5"
    bin2x2_path = scan_path + "bin2x2/scan_X.h5"
    bin4x4_path = scan_path + "bin4x4/scan-6990-7010_recon.h5"

    bin_1x1_out = scan_path + "bin1x1/scan-7014-7034_proccesed.h5"
    bin_2x2_out = scan_path + "bin2x2/scan_X_proccesed.h5"
    bin_4x4_out = scan_path + "bin4x4/scan-6990-7010_proccesed.h5"

    # Define the cropping parameters

    # Crop the HDF5 files
    # brain_1_20kev_20x_16bits_30sdd_2 has 24614 x 24614 x 2588
    bin1x1, perc_1x1 = crop_slices_hdf5(bin1x1_path, 10807, 13807, 10807, 13807, output_path=bin_1x1_out, order='HWD')
    bin2x2, perc_2x2 = crop_slices_hdf5(bin2x2_path, 5403, 6903, 5403, 6903, output_path=bin_1x1_out, order='HWD')
    bin4x4, perc_4x4 = crop_slices_hdf5(bin4x4_path, 2701, 3451, 2701, 3451, output_path=bin_1x1_out, order='HWD')

    # Match the histograms of the cropped images
    bin2x2 = match_slice_histograms(input=bin2x2, target=bin1x1, up_factor=2, order='HWD')
    bin4x4 = match_slice_histograms(input=bin4x4, target=bin1x1, up_factor=4, order='HWD')

    # Save the processed images

    print("Done")