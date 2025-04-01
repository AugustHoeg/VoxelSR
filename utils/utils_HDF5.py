
import math
import datetime
import h5py

import numpy as np
import matplotlib.pyplot as plt
import os
import skimage

from utils_3D_image import rescale_array_

# TODO: Rewrite these functions to save the image in the order (slices, rows, cols) or (depth, height, width)

def crop_slices_hdf5(hdf5_path, data_name, start_row, end_row, start_col, end_col, output_path, sample_percent=0.2, rescale=True, num_slices_plot=4, save_npy=True, order='DHW'):
    """
    Reads a 3D image from an HDF5 file slice-by-slice, crops each slice,
    and saves the result as a .npy file.

    Parameters:
        hdf5_path (str): Path to the HDF5 file.
        data_name (str): Name of the dataset in the HDF5 file.
        start_row (int): Number of rows to crop from the top.
        end_row (int): Number of rows to crop from the bottom.
        start_col (int): Number of columns to crop from the left.
        end_col (int): Number of columns to crop from the right.    
        output_path (str): Path to save the cropped array as a .npy file.
        sample_percent (float): Percentage of the slice area to sample for intensity rescaling.
        rescale (bool): Whether to rescale the intensities of the cropped data.
        num_slices_plot (int): Number of slices to plot after cropping.
    """

    min_val = math.inf
    max_val = -math.inf

    samples = []

    with h5py.File(hdf5_path, 'r') as f:
        data = f['/exchange/data']

        # Extract the shape of the dataset
        depth, height, width = data.shape
        print("HDF5 shape: {}".format(data.shape))

        # Create an empty array to hold the cropped data
        if order == "DHW":
            crop_shape = (depth, height - start_row - end_row, width - start_col - end_col)
        elif order == "HWD":
            crop_shape = (height - start_row - end_row, width - start_col - end_col, depth)

        crop_arr = np.memmap(output_path, shape=crop_shape, dtype=np.float32, mode='w+')

        # Process each slice
        for i in range(depth):
            print(f"Processing slice: {i+1}/{data.shape[0]}")
            crop_slice = data[i, start_row:-end_row, start_col:-end_col]
            if order == "HWD":
                crop_arr[:, :, i] = crop_slice.astype(np.float32)
            elif order == "DHW":
                crop_arr[i, :, :] = crop_slice.astype(np.float32)

            # update min and max
            #slice_min = crop_slice.min()
            #slice_max = crop_slice.max()
            #min_val = min(slice_min, min_val)
            #max_val = max(slice_max, max_val)

            # Sample X points randomly and append to list
            slice_area = np.prod(crop_slice.shape)  # Area of the slice, 1*H*W
            num_samples = int(sample_percent*slice_area)  # X% of the slice area
            samples.extend(np.random.choice(crop_slice.flatten(), num_samples, replace=False))

        # Stack the cropped slices back into a 3D array
        print("Shape of cropped_array:", crop_arr.shape)

    # Calculate 5th and 95th percentile
    percentiles = np.percentile(samples, [5, 95])
    p5, p95 = percentiles

    # Rescale intensities
    if rescale:
        rescale_array_(crop_arr, mina=p5, maxa=p95, new_min=0.0, new_max=1.0)

    # Save the cropped array as a .npy file
    if save_npy:
        np.save(output_path, crop_arr)
        print(f"Cropped data saved to {output_path}")

    # Plot a few slices from the cropped stack
    plt.figure(figsize=(10, num_slices_plot * 3))
    total_slices = depth
    indices_to_plot = np.linspace(0, total_slices - 1, num_slices_plot**2, dtype=int)
    for idx, slice_idx in enumerate(indices_to_plot):
        plt.subplot(num_slices_plot, num_slices_plot, idx + 1)
        if order == "HWD":
            plt.imshow(crop_arr[:, :, slice_idx], cmap='gray', vmin=0.0, vmax=1.0)
        elif order == "DHW":
            plt.imshow(crop_arr[slice_idx, :, :], cmap='gray', vmin=0.0, vmax=1.0)
        plt.title(f'Slice {slice_idx}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), f"cropped_slices_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"))
    plt.close()

    return crop_arr, percentiles,


def match_slice_histograms(input, target, up_factor, target_percentiles=None, order='DHW'):
    # Match the histograms of the input slices to the target slices
    # Assumes (D, H, W) order or (slices, rows, cols)

    if order == "DHW":
        depth, height, width = input.shape  # (D, H, W)
    elif order == "HWD":
        height, width, depth = input.shape  # (H, W, D)

    for i in range(depth):
        if i % 10 == 0:
            print(f"Processing slice: {i+1}/{depth}")
        if order == "DHW":
            input_slice = input[i, :, :]
            target_slice = np.sum(target[i * up_factor:i*up_factor + up_factor, :, :]) / up_factor
            input[i, :, :] = skimage.exposure.match_histograms(input_slice, target_slice, multichannel=False)
        elif order == "HWD":
            input_slice = input[:, :, i]
            target_slice = np.sum(target[:, :, i * up_factor:i*up_factor + up_factor]) / up_factor
            input[:, :, i] = skimage.exposure.match_histograms(input_slice, target_slice, multichannel=False)

    return input




# 63 -> 255

# 63*4 + 4 = 252 + 4 = 256