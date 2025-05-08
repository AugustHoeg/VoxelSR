import numpy as np
import h5py
import math
import matplotlib.pyplot as plt
from time import perf_counter as time

from utils.utils_HDF5 import save_npy, save_memmap, estimate_percentiles, parallel_estimate_percentiles, match_slice_histograms, plot_slices

if __name__ == "__main__":

    project_path = "/dtu/3d-imaging-center/projects/2024_DANFIX_130_ExtremeCT/raw_data_extern/"
    scan_path = project_path + "2024031208/brain_1_20kev_20x_16bits_30sdd/"

    dataset_path = "/work3/s173944/Python/venv_srgan/3D_datasets/datasets/danmax/binning_brain/"

    # Define the input and output paths
    bin1x1_path = scan_path + "bin1x1/scan-7014-7034_recon.h5"
    bin2x2_path = scan_path + "bin2x2/scan-6966-6986_recon.h5"
    bin4x4_path = scan_path + "bin4x4/scan-6990-7010_recon.h5"

    # Crop the HDF5 files
    # brain_1_20kev_20x_16bits_30sdd_2 has 2588 x 24614 x 24614
    # brain_1_20kev_20x_16bits_30sdd_bin2x2 has 1290 x 12306 x 12306
    # brain_1_20kev_20x_16bits_30sdd_bin4x4 has 645 x 6152 x 6152


    #perc_bin1x1 = estimate_percentiles(bin1x1_path, 10807, 13807, 10807, 13807, sample_percent=0.2)
    start = time()
    perc_bin2x2 = parallel_estimate_percentiles(bin2x2_path, 5403, 6903, 5403, 6903, sample_percent=0.2, n_proc=12, step=10)
    print(f"Estimation of percentile for bin2x2 took {time() - start} sec.")

    start = time()
    perc_bin4x4 = parallel_estimate_percentiles(bin4x4_path, 2701, 3451, 2701, 3451, sample_percent=0.2, n_proc=12, step=10)
    print(f"Estimation of percentile for bin4x4 took {time() - start} sec.")

    # Save the cropped images as .npy files
    out_name = dataset_path + "bin2x2/brain_1_train.npy"
    bin2x2_train = save_npy(bin2x2_path, 5403, 6903, 5403, 6903, 0, 645, output_path=out_name, percentiles=perc_bin2x2, order='HWD')
    plot_slices(bin2x2_train, num_slices_plot=4, order='HWD')

    out_name = dataset_path + "bin2x2/brain_1_test.npy"
    bin2x2_test = save_npy(bin2x2_path, 5403, 6903, 5403, 6903, 645, 1290, output_path=out_name, percentiles=perc_bin2x2, order='HWD')
    plot_slices(bin2x2_test, num_slices_plot=4, order='HWD')

    out_name = dataset_path + "bin4x4/brain_1_train.npy"
    bin4x4_train = save_npy(bin4x4_path, 2701, 3451, 2701, 3451, 0, 322, output_path=out_name, percentiles=perc_bin4x4, order='HWD')
    plot_slices(bin4x4_train, num_slices_plot=4, order='HWD')

    out_name = dataset_path + "bin4x4/brain_1_test.npy"
    bin4x4_test = save_npy(bin4x4_path, 2701, 3451, 2701, 3451, 322, 645, output_path=out_name, percentiles=perc_bin4x4, order='HWD')
    plot_slices(bin4x4_test, num_slices_plot=4, order='HWD')

    # Match the histograms of the cropped images
    #bin2x2 = match_slice_histograms(input=bin2x2, target=bin1x1, up_factor=2, order='HWD')
    #bin4x4 = match_slice_histograms(input=bin4x4, target=bin2x2, up_factor=4, order='HWD')

    # Save the processed images
    # Divide into train/test

    print("Done")