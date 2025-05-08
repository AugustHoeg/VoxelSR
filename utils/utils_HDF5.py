
import math
import datetime
import h5py

import numpy as np
import matplotlib.pyplot as plt
import os
import skimage
import zarr
#import multiprocessing
from multiprocessing.pool import ThreadPool, Pool

from utils.utils_3D_image import rescale_array_

# TODO: Rewrite these functions to save the image in the order (slices, rows, cols) or (depth, height, width)

def crop_slices(hdf5_path, idx, start_row, end_row, start_col, end_col, percentiles=None):

    print("Processing slice: {}".format(idx+1), end="\r")

    """Each worker should open the HDF5 file separately."""
    with h5py.File(hdf5_path, 'r') as f:
        data = f['/exchange/data']  # Open the dataset inside the worker
        crop_slice = data[idx, start_row:end_row, start_col:end_col]  # Read data
        if percentiles is not None:
            rescale_array_(crop_slice, mina=percentiles[0], maxa=percentiles[1], new_min=0.0, new_max=1.0)
        return crop_slice

def parallel_crop_slices(hdf5_path, start_depth, end_depth, start_row, end_row, start_col, end_col, n_proc=8, percentiles=None):
    """Estimate percentiles using multiprocessing."""

    with h5py.File(hdf5_path, 'r') as f:
        depth, height, width = f['/exchange/data'].shape
        print("HDF5 shape: {}".format((depth, height, width)))

    with ThreadPool(n_proc) as pool:
        results_async = [
            pool.apply_async(crop_slices, args=(hdf5_path, idx, start_row, end_row, start_col, end_col, percentiles))
            for idx in range(start_depth, end_depth)
        ]

        crop_arr = np.stack([r.get() for r in results_async])  # Retrieve and concatenate results

    return crop_arr


def sample_from_slice(hdf5_path, idx, N, start_row, end_row, start_col, end_col):
    """Each worker should open the HDF5 file separately."""
    with h5py.File(hdf5_path, 'r') as f:
        data = f['/exchange/data']  # Open the dataset inside the worker
        crop_slice = data[idx, start_row:end_row, start_col:end_col]  # Read data
        flat_indices = np.random.choice(crop_slice.size, N, replace=False)
        return crop_slice.ravel()[flat_indices]  # Return sampled values


def parallel_estimate_percentiles(hdf5_path, start_row, end_row, start_col, end_col, sample_percent=0.2, n_proc=8, step=10):
    """Estimate percentiles using multiprocessing."""

    with h5py.File(hdf5_path, 'r') as f:
        depth, height, width = f['/exchange/data'].shape
        print("HDF5 shape: {}".format((depth, height, width)))

    N = int(sample_percent * (end_row - start_row) * (end_col - start_col))

    with ThreadPool(n_proc) as pool:
        results_async = [
            pool.apply_async(sample_from_slice, args=(hdf5_path, idx, N, start_row, end_row, start_col, end_col))
            for idx in range(0, depth, step)
        ]

        samples = np.concatenate([r.get() for r in results_async])  # Retrieve and concatenate results

    percentiles = np.percentile(samples, [5, 95])  # Compute percentiles
    return percentiles

# def sample_from_slice(data, idx, N, start_row, end_row, start_col, end_col):
#     print("Processing slice: {}/{}".format(idx+1, data.shape[0]), end="\r")
#     #crop_slice = data[idx, start_row:end_row, start_col:end_col] # Read data
#     #return np.random.choice(crop_slice.reshape(-1), N, replace=False)
#     crop_slice = data[idx, start_row:end_row, start_col:end_col]
#     flat_indices = np.random.choice(np.prod(crop_slice.shape), N, replace=False)
#     return crop_slice.ravel()[flat_indices]

# def parallel_estimate_percentiles(hdf5_path, start_row, end_row, start_col, end_col, sample_percent=0.2, n_proc=8, step=10):
#
#     with h5py.File(hdf5_path, 'r') as f:
#         data = f['/exchange/data']
#
#         # Extract the shape of the dataset
#         depth, height, width = data.shape
#         print("HDF5 shape: {}".format(data.shape))
#
#         N = int(sample_percent * (end_row - start_row) * (end_col - start_col))
#
#         pool = multiprocessing.Pool(n_proc)
#         results_async = [pool.apply_async(sample_from_slice(data, idx, N, start_row, end_row, start_col, end_col)) for idx in range(0, depth, step)]
#
#         #samples = [r.get() for r in results_async]
#         percentiles = np.percentile(np.array(results_async), [5, 95])
#         return percentiles


def estimate_percentiles(hdf5_path, start_row, end_row, start_col, end_col, sample_percent=0.2, step=10):
    """
    Estimates the 5th and 95th percentiles of the intensities in within cropped region of an HDF5 file.

    Parameters:
        hdf5_path (str): Path to the HDF5 file.
        start_row (int): Number of rows to crop from the top.
        end_row (int): Number of rows to crop from the bottom.
        start_col (int): Number of columns to crop from the left.
        end_col (int): Number of columns to crop from the right.
        sample_percent (float): Percentage of the slice area to sample for intensity rescaling.
    """

    # N = int(3000 * 3000 * 0.2)
    # idx = np.random.randint(low=(1, 10, 100), high=(10, 100, 1000), size=(N, 3))
    # idx_sorted = idx[np.lexsort((idx[:, 2], idx[:, 1], idx[:, 0]))]

    # ranges = [[0, 3000], [0, 3000], [0, 2500]]
    # idx = np.stack([np.sort(np.random.randint(low=low, high=high, size=N)) for low, high in ranges])

    with h5py.File(hdf5_path, 'r') as f:
        data = f['/exchange/data']

        # Extract the shape of the dataset
        depth, height, width = data.shape
        print("HDF5 shape: {}".format(data.shape))

        N = int(sample_percent * (end_row - start_row) * (end_col - start_col))
        samples = np.zeros((depth//step, N))

        # Process each slice
        for i in range(depth//step):  # depth
            print(f"Processing slice: {i*step + 1}/{data.shape[0]}")
            crop_slice = data[i*step, start_row:end_row, start_col:end_col]

            flat_indices = np.random.choice(np.prod(crop_slice.shape), N, replace=False)
            samples[i] = crop_slice.ravel()[flat_indices]
            #samples[i] = np.random.choice(crop_slice.reshape(-1), N, replace=False)  # .reshape(-1) should return a view and not a copy

            # update min and max
            #slice_min = crop_slice.min()
            #slice_max = crop_slice.max()
            #min_val = min(slice_min, min_val)
            #max_val = max(slice_max, max_val)

    # Calculate 5th and 95th percentile
    percentiles = np.percentile(samples, [5, 95])
    return percentiles


def plot_slices(arr, num_slices_plot = 4, order='DHW'):

    if order == "DHW":
        depth, height, width = arr.shape
    elif order == "HWD":
        height, width, depth = arr.shape

    # Plot a few slices from the cropped stack
    plt.figure(figsize=(10, num_slices_plot * 3))
    total_slices = depth
    indices_to_plot = np.linspace(0, total_slices - 1, num_slices_plot ** 2, dtype=int)
    for idx, slice_idx in enumerate(indices_to_plot):
        plt.subplot(num_slices_plot, num_slices_plot, idx + 1)
        if order == "HWD":
            plt.imshow(arr[:, :, slice_idx], cmap='gray', vmin=0.0, vmax=1.0)
        elif order == "DHW":
            plt.imshow(arr[slice_idx, :, :], cmap='gray', vmin=0.0, vmax=1.0)
        plt.title(f'Slice {slice_idx}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), f"cropped_slices_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"))
    plt.close()


def save_memmap(hdf5_path, start_row, end_row, start_col, end_col, start_depth, end_depth, output_path, percentiles, rescale=True, order='DHW'):
    """
    Reads a 3D image from an HDF5 file slice-by-slice, crops each slice,
    and saves the result as a .npy file using memmap.

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

    p_low, p_high = percentiles

    with h5py.File(hdf5_path, 'r') as f:
        data = f['/exchange/data']

        # Extract the shape of the dataset
        depth, height, width = data.shape
        print("HDF5 shape: {}".format(data.shape))

        # Create an empty array to hold the cropped data
        if order == "DHW":
            crop_shape = (end_depth - start_depth, end_row - start_row, end_col - start_col)
        elif order == "HWD":
            crop_shape = (end_row - start_row, end_col - start_col, end_depth - start_depth)
        print("Shape of cropped_array:", crop_shape)

        crop_arr = np.memmap(output_path, shape=crop_shape, dtype=np.float32, mode='w+')

        # Process each slice
        for i in range(start_depth, end_depth):
            print(f"Processing slice: {i+1}/{end_depth}")
            crop_slice = data[i, start_row:end_row, start_col:end_col]

            if rescale:
                rescale_array_(crop_slice, mina=p_low, maxa=p_high, new_min=0.0, new_max=1.0)

            if order == "HWD":
                crop_arr[:, :, i-start_depth] = crop_slice.astype(np.float32)
            elif order == "DHW":
                crop_arr[i-start_depth, :, :] = crop_slice.astype(np.float32)

    return crop_arr


def save_npy(hdf5_path, start_row, end_row, start_col, end_col, start_depth, end_depth, output_path, percentiles, order='DHW', n_proc=12):

    crop_arr = parallel_crop_slices(hdf5_path, start_depth, end_depth, start_row, end_row, start_col, end_col, n_proc, percentiles)

    if order == "HWD":
        crop_arr = np.transpose(crop_arr, (1, 2, 0))

    print("Cropped array shape:", crop_arr.shape)

    np.save(output_path, crop_arr)

    return crop_arr


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


def save_Zarr(hdf5_path, chunk_shape, start_row, end_row, start_col, end_col, start_depth, end_depth, output_path, percentiles, rescale=True):

    p_low, p_high = percentiles

    with h5py.File(hdf5_path, 'r') as f:
        data = f['/exchange/data']

        # Extract the shape of the dataset
        depth, height, width = data.shape
        print("HDF5 shape: {}".format(data.shape))

        # Create an empty array to hold the cropped data
        crop_shape = (depth - start_depth - end_depth, height - start_row - end_row, width - start_col - end_col)
        print("Shape of cropped_array:", crop_shape, "assuming (D, H, W) order")

        crop_arr = zarr.open(output_path, mode='w', shape=crop_shape, chunks=chunk_shape, dtype=np.float32)

        n_depth_tiles = math.ceil(crop_shape[0] / chunk_shape[0])
        n_row_tiles = math.ceil(crop_shape[1] / chunk_shape[1])
        n_col_tiles = math.ceil(crop_shape[2] / chunk_shape[2])

        # Process slices in chunks
        for i in range(n_depth_tiles):
            for j in range(n_row_tiles):
                for k in range(n_col_tiles):
                    chunk = data[i * crop_shape[0]:(i + 1) * crop_shape[0],
                                 j * crop_shape[1]:(j + 1) * crop_shape[1],
                                 k * crop_shape[2]:(k + 1) * crop_shape[2]]

                    if rescale:
                        rescale_array_(chunk, mina=p_low, maxa=p_high, new_min=0.0, new_max=1.0)

                    crop_arr.blocks[i, j, k] = chunk

    return crop_arr


if __name__ == "__main__":

    N = 10000
    idx = np.random.randint(low=(1,10,100), high=(10, 100, 1000), size=(N,3))
    idx_sorted = idx[np.lexsort((idx[:,2], idx[:,1], idx[:,0]))]

    print(idx)

    print("Done")

# 63 -> 255

# 63*4 + 4 = 252 + 4 = 256