import os
import math
import zarr
from numcodecs import Blosc
import dask.array as da
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
import torchio.transforms as tiotransforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from tqdm import tqdm

import h5py
import monai.transforms as mt

def generate_patch_coords(D, H, W, stride, f):
    z_idx = np.arange(0, D, stride)
    y_idx = np.arange(0, H, stride)
    x_idx = np.arange(0, W, stride)
    # print(len(z_idx))

    zz, yy, xx = np.meshgrid(z_idx, y_idx, x_idx, indexing='ij')
    coords_lr = np.stack([zz, yy, xx], axis=-1).reshape(-1, 3)
    coords_hr = coords_lr * f
    return coords_lr, coords_hr

#from utils.utils_3D_image import run_strided_inference
#run_strided_inference(model=model, img_L=np.zeros((1, 200, 200, 200), dtype=np.float32), f=4, size_lr=20, border=0, batch_size=2)

def get_hann_window(patch_size):
    hann_window_3d = torch.as_tensor([1])
    # create a n-dim hann window
    for spatial_dim, size in enumerate(patch_size):
        window_shape = np.ones_like(patch_size)
        window_shape[spatial_dim] = size
        hann_window_1d = torch.hann_window(
            size + 2,
            periodic=False,
        )
        hann_window_1d = hann_window_1d[1:-1].view(*window_shape)
        hann_window_3d = hann_window_3d * hann_window_1d
    return hann_window_3d

def run_strided_inference(model, img_L, f, size_lr, border, batch_size, overlap_mode="hann"):
    C, D, H, W = img_L.shape
    size_hr = size_lr * f
    stride = size_lr - border

    img_E = torch.zeros((C, int(D * f), int(H * f), int(W * f)), dtype=torch.float32)
    weight = torch.zeros_like(img_E)

    coords_lr, coords_hr = generate_patch_coords(D, H, W, stride, f)
    N = coords_lr.shape[0]

    patch_batch = torch.empty((batch_size, C, size_lr, size_lr, size_lr), dtype=img_L.dtype)

    if overlap_mode == "hann":
        hann_window = get_hann_window((size_hr, size_hr, size_hr))
        hann_window = hann_window.reshape(1, size_hr, size_hr, size_hr)

    model.netG.eval()
    with torch.inference_mode():
        for i in range(0, N, batch_size):
            if i % 10 == 0:
                print("Processing batch %d-%d/%d" % (i, i+batch_size, N))
            batch_coords_lr = coords_lr[i:i+batch_size]
            batch_coords_hr = coords_hr[i:i+batch_size]

            for j, (z, y, x) in enumerate(batch_coords_lr):
                patch = torch.zeros((C, size_lr, size_lr, size_lr))  # reinitialize patch
                data_L = img_L[:, z:z+size_lr, y:y+size_lr, x:x+size_lr]  # Extract data
                patch[:, :data_L.shape[1], :data_L.shape[2], :data_L.shape[3]] = data_L  # Fill patch with data
                patch_batch[j] = patch  # Fill batch with patch

            #upsampled_batch = np.ones((batch_size, C, size_hr, size_hr, size_hr))  # dummy initialization
            model.L = patch_batch.to(model.device)
            model.netG_forward()
            upsampled_batch = model.E.float().cpu()  # Transfer back to CPU

            for j, (z_hr, y_hr, x_hr) in enumerate(batch_coords_hr):
                dz = min(z_hr+size_hr, D*f) - z_hr
                dy = min(y_hr+size_hr, H*f) - y_hr
                dx = min(x_hr+size_hr, W*f) - x_hr
                if overlap_mode == "hann":
                    img_E[:, z_hr:z_hr+dz, y_hr:y_hr+dy, x_hr:x_hr+dx] += upsampled_batch[j, :, :dz, :dy, :dx] * hann_window[:, :dz, :dy, :dx]
                    weight[:, z_hr:z_hr+dz, y_hr:y_hr+dy, x_hr:x_hr+dx] += hann_window[:, :dz, :dy, :dx]
                elif overlap_mode == "mean":
                    img_E[:, z_hr:z_hr + dz, y_hr:y_hr + dy, x_hr:x_hr + dx] += upsampled_batch[j, :, :dz, :dy, :dx]
                    weight[:, z_hr:z_hr+dz, y_hr:y_hr+dy, x_hr:x_hr+dx] += 1
                #weight[:, z_hr:z_hr+size_hr, y_hr:y_hr+size_hr, x_hr:x_hr+size_hr] += 1

    if overlap_mode == "mean":
        weight[weight == 0] = 1
    img_E /= weight
    return img_E


def run_strided_inference_zarr(model, zarr_path, out_path, group_pair, f, size_lr, border, batch_size, overlap_mode="hann"):

    # print("TODO: Fix hardcoded global min/max values for bone_2_ome.zarr/HR/2")
    # global_min = -0.002063  # min value for bone_2_ome.zarr/HR/2
    # global_max = 0.002476  # max value for bone_2_ome.zarr/HR/2

    level_L = int(group_pair["L"].split("/")[-1])
    level_H = int(group_pair["H"].split("/")[-1])

    # Open input Zarr
    z = zarr.open(zarr_path, mode='r')
    img_L = z[group_pair["L"]]
    img_H = z[group_pair["H"]]
    D, H, W = img_L.shape
    size_hr = size_lr * f
    stride = size_lr - border

    chunks_L = img_L.chunks
    chunks_H = img_H.chunks  # tuple(int(c * f) for c in chunks_L)

    # Prepare output Zarr store
    compressor = Blosc(cname='lz4', clevel=3, shuffle=Blosc.BITSHUFFLE)
    root_out = zarr.open(out_path, mode='w')
    grp_out = root_out.create_group("temp")
    grp_weight = root_out.create_group("weight")

    grp_out.create_dataset(level_H, shape=(D*f, H*f, W*f), chunks=chunks_H, dtype=np.float32, compressor=compressor)
    grp_weight.create_dataset(level_H, shape=(D*f, H*f, W*f), chunks=chunks_H, dtype=np.float32, compressor=compressor)

    z_E = grp_out[level_H]
    z_W = grp_weight[level_H]

    coords_lr, coords_hr = generate_patch_coords(D, H, W, stride, f)
    N = coords_lr.shape[0]
    patch_batch = torch.empty((batch_size, 1, size_lr, size_lr, size_lr), dtype=torch.float32)

    if overlap_mode == "hann":
        hann_window = get_hann_window((size_hr, size_hr, size_hr))

    model.netG.eval()
    with torch.inference_mode():
        for i in range(0, N, batch_size):
            print(f"Processing batch {i}-{min(i+batch_size, N)}/{N}")
            batch_coords_lr = coords_lr[i:i+batch_size]
            batch_coords_hr = coords_hr[i:i+batch_size]

            for j, (z0, y0, x0) in enumerate(batch_coords_lr):
                patch = torch.zeros((1, size_lr, size_lr, size_lr), dtype=torch.float32)
                data_L = img_L[z0:z0+size_lr, y0:y0+size_lr, x0:x0+size_lr]

                # Only for binning bone...
                data_L = data_L.astype(np.float32)  # Ensure data is float32
                # data_L = (data_L - global_min) / (global_max - global_min)

                patch[:, :data_L.shape[0], :data_L.shape[1], :data_L.shape[2]] = torch.from_numpy(data_L)
                patch_batch[j] = patch

            model.L = patch_batch.to(model.device)
            model.netG_forward()
            upsampled_batch = model.E.float().cpu()

            for j, (z_hr, y_hr, x_hr) in enumerate(batch_coords_hr):
                dz = min(z_hr+size_hr, D*f) - z_hr
                dy = min(y_hr+size_hr, H*f) - y_hr
                dx = min(x_hr+size_hr, W*f) - x_hr
                patch_E = upsampled_batch[j, 0, :dz, :dy, :dx].numpy()

                if overlap_mode == "hann":
                    window = hann_window[:dz, :dy, :dx].numpy()
                    z_E[z_hr:z_hr+dz, y_hr:y_hr+dy, x_hr:x_hr+dx] += patch_E * window
                    z_W[z_hr:z_hr+dz, y_hr:y_hr+dy, x_hr:x_hr+dx] += window
                else:  # mean
                    z_E[z_hr:z_hr+dz, y_hr:y_hr+dy, x_hr:x_hr+dx] += patch_E
                    z_W[z_hr:z_hr+dz, y_hr:y_hr+dy, x_hr:x_hr+dx] += 1

    # Normalize with Dask (in memory-safe way)
    print("Normalizing output by overlap weights...")

    data_E = da.from_zarr(z_E)
    data_W = da.from_zarr(z_W)

    if overlap_mode == "mean":
        data_W = da.where(data_W == 0, 1, data_W)

    data_norm = (data_E / data_W).astype(np.float16)

    # Create image pyramid using downscale_local_mean
    image_pyramid = [data_norm]
    for i in range(2):
        image_pyramid.append(da.coarsen(np.mean, image_pyramid[i], axes={0: 2, 1: 2, 2: 2}))

    # Create image group for the volume
    image_group = root_out.create_group("SR")

    from utils.utils_zarr import write_ome_pyramid
    write_ome_pyramid(
        image_group=image_group,
        image_pyramid=image_pyramid,
        label_pyramid=None,  # No labels
        chunk_size=chunks_H,
        cname='lz4'  # Compression codec
    )

    # Remove image and weight groups
    del root_out["temp"]
    del root_out["weight"]

    print(f"Done writing {out_path} to OME-Zarr format at {image_group.name}")

    # # Save normalized result to a new group
    # norm_grp = root_out.require_group("upscaled")
    # norm_grp.array(name=level_H, data=da_norm, chunks=chunks_H, dtype=np.float16, compressor=compressor)

    return 0

# def run_strided_inference_zarr(model, zarr_path, group_name, weight_name, level_L, level_H, f, size_lr, border, batch_size, overlap_mode="hann"):
#
#     # We assume img_L is a zarr array
#     z = zarr.open(zarr_path, mode='r')
#     img_L = z[group_name][level_L]
#     img_H = z[group_name][level_H]
#
#     chunks_L = img_L.chunks
#     chunks_H = img_H.chunks
#
#     C, D, H, W = img_L.shape
#     size_hr = size_lr * f
#     stride = size_lr - border
#
#     # Create zarr for super-resolution output
#     z_out = zarr.open(f"E_{path}", mode='w', shape=(C, int(D * f), int(H * f), int(W * f)), chunks=chunks_L, dtype=np.float16)
#     root = zarr.group(store=z_out)
#     image_group = root.create_group(group_name)
#     weight_group = root.create_group(weight_name)
#
#     coords_lr, coords_hr = generate_patch_coords(D, H, W, stride, f)
#     N = coords_lr.shape[0]
#
#     patch_batch = torch.empty((batch_size, C, size_lr, size_lr, size_lr), dtype=img_L.dtype)
#
#     if overlap_mode == "hann":
#         hann_window = get_hann_window((size_hr, size_hr, size_hr))
#         hann_window = hann_window.reshape(1, size_hr, size_hr, size_hr)
#
#     model.netG.eval()
#     with torch.inference_mode():
#         for i in range(0, N, batch_size):
#             if i % 10 == 0:
#                 print("Processing batch %d-%d/%d" % (i, i+batch_size, N))
#             batch_coords_lr = coords_lr[i:i+batch_size]
#             batch_coords_hr = coords_hr[i:i+batch_size]
#
#             for j, (z, y, x) in enumerate(batch_coords_lr):
#                 patch = torch.zeros((C, size_lr, size_lr, size_lr))  # reinitialize patch
#                 data_L = img_L[:, z:z+size_lr, y:y+size_lr, x:x+size_lr]  # Extract data
#                 patch[:, :data_L.shape[1], :data_L.shape[2], :data_L.shape[3]] = data_L  # Fill patch with data
#                 patch_batch[j] = patch  # Fill batch with patch
#
#             #upsampled_batch = np.ones((batch_size, C, size_hr, size_hr, size_hr))  # dummy initialization
#             model.L = patch_batch.to(model.device)
#             model.netG_forward()
#             upsampled_batch = model.E.float().cpu()  # Transfer back to CPU
#
#             for j, (z_hr, y_hr, x_hr) in enumerate(batch_coords_hr):
#                 dz = min(z_hr+size_hr, D*f) - z_hr
#                 dy = min(y_hr+size_hr, H*f) - y_hr
#                 dx = min(x_hr+size_hr, W*f) - x_hr
#                 if overlap_mode == "hann":
#                     z_out[group_name][level_H][:, z_hr:z_hr+dz, y_hr:y_hr+dy, x_hr:x_hr+dx] += upsampled_batch[j, :, :dz, :dy, :dx] * hann_window[:, :dz, :dy, :dx]
#                     z_out[weight_name][level_H][:, z_hr:z_hr+dz, y_hr:y_hr+dy, x_hr:x_hr+dx] += hann_window[:, :dz, :dy, :dx]
#                 elif overlap_mode == "mean":
#                     z_out[group_name][level_H][:, z_hr:z_hr + dz, y_hr:y_hr + dy, x_hr:x_hr + dx] += upsampled_batch[j, :, :dz, :dy, :dx]
#                     z_out[weight_name][level_H][:, z_hr:z_hr+dz, y_hr:y_hr+dy, x_hr:x_hr+dx] += 1
#                 #weight[:, z_hr:z_hr+size_hr, y_hr:y_hr+size_hr, x_hr:x_hr+size_hr] += 1
#
#     # Normalize the output by the weight
#     #for idx in z_out[group_name][level_H].chunk_grid:
#     for idx in tqdm(z_out[group_name][level_H].chunk_grid, desc="Writing SR output", mininterval=2):
#         chunk_E = z_out[group_name][level_H].get_chunk(idx)
#         chunk_W = z_out[weight_name][level_H].get_chunk(idx)
#         if overlap_mode == "mean":
#             chunk_W[chunk_W == 0] = 1
#         z_out[group_name][level_H].set_chunk(idx, chunk_E / chunk_W)
#
#     # # Normalize the output by the weight using dask
#     # img_E_da = da.from_array(z_out[group_name][level_H], chunks=chunks_H)
#     # weight_da = da.from_array(z_out[weight_name][level_H], chunks=chunks_H)
#
#     # if overlap_mode == "mean":
#     #     weight_da[weight_da == 0] = 1
#     # img_E_da = img_E_da / weight_da
#
#     return 0


def rescale_array_(arr, mina, maxa, new_min=0.0, new_max=1.0, dtype=np.float32):
    """
    Rescale the values of numpy array `arr` to be from `minv` to `maxv`.
    If either `minv` or `maxv` is None, it returns `(a - min_a) / (max_a - min_a)`.

    Args:
        arr: input array to rescale.
        minv: minimum value of target rescaled array.
        maxv: maximum value of target rescaled array.
        dtype: if not None, convert input array to dtype before computation.
    """
    #arr = arr.astype(dtype, copy=False)
    if mina is None:
        mina = arr.min()
    if maxa is None:
        maxa = arr.max()

    # print("Rescaling array...")
    # Normalize the array in-place
    arr -= mina  # Subtract min (in-place)
    arr /= (maxa - mina)  # Divide by range (in-place)

    # Rescale to the new range in-place
    if (new_min is None) or (new_max is None):
        return arr

    arr *= (new_max - new_min)  # Scale by the new range (in-place)
    arr += new_min  # Shift by the new min (in-place)

    #if mina == maxa:
    #    return arr * new_min if new_min is not None else arr

    #norm = (arr - mina) / (maxa - mina)  # normalize the array first
    #if (new_min is None) or (new_max is None):
    #    return norm
    #return (norm * (new_max - new_min)) + new_min  # rescale by minv and maxv, which is the normalized array by default





def print_hdf5_tree(file_path):
    """
    Prints the entire tree structure of an HDF5 file.

    Parameters:
        file_path (str): Path to the HDF5 file.
    """

    def visit_group(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"  Dataset: {name}, Shape: {obj.shape}, Dtype: {obj.dtype}")

    with h5py.File(file_path, 'r') as f:
        print(f"Contents of HDF5 file: {file_path}")
        f.visititems(visit_group)


def process_hdf5(hdf5_path, data_name, crop_indices, output_path, rescale=True, num_slices_plot=4):
    """
    Reads a 3D image from an HDF5 file slice-by-slice, crops each slice,
    and saves the result as a .npy file.

    Parameters:
        hdf5_path (str): Path to the HDF5 file.
        data_name (str): Name of the dataset in the HDF5 file.
        crop_indices (dict): Dictionary with 'start_row', 'end_row', 'start_col', 'end_col'.
        output_path (str): Path to save the cropped array as a .npy file.
    """

    #output_dir = os.path.dirname(output_path)
    #output_base = os.path.splitext(os.path.basename(output_dir))[0]
    #pdf_output_file = os.path.join(output_dir, f"{output_base}_cropped_slices.pdf")

    start_row, end_row = crop_indices['start_row'], crop_indices['end_row']
    start_col, end_col = crop_indices['start_col'], crop_indices['end_col']

    min_val = math.inf
    max_val = -math.inf

    samples = []

    with h5py.File(hdf5_path, 'r') as f:
        data = f['/exchange/data']

        # Extract the shape of the dataset
        depth, height, width = data.shape
        print("HDF5 shape: {}".format(data.shape))

        # Create an empty array to hold the cropped data
        cropped_array = np.zeros((height - start_row - end_row, width - start_col - end_col, depth))  #[]

        # Process each slice
        for i in range(data.shape[0]):
            print(f"Processing slice: {i+1}/{data.shape[0]}")
            cropped_slice = data[i, start_row:-end_row, start_col:-end_col]
            cropped_array[:, :, i] = cropped_slice.astype(np.float32)

            # update min and max
            slice_min = cropped_slice.min()
            slice_max = cropped_slice.max()
            min_val = min(slice_min, min_val)
            max_val = max(slice_max, max_val)

            # Sample 1000 points randomly and append to list
            slice_area = cropped_slice.shape[0] * cropped_slice.shape[1]
            num_sampels = int(0.2*slice_area)  # 20% of the slice area
            samples.extend(np.random.choice(cropped_slice.flatten(), num_sampels, replace=False))

        # Stack the cropped slices back into a 3D array
        #cropped_array = np.stack(cropped_slices)
        print("Shape of cropped_array:", cropped_array.shape)
        #cropped_array = np.transpose(cropped_array, (1, 2, 0))  # D, H, W -> H, W, D

    # Calculate 5th and 95th percentile
    p5, p95 = np.percentile(samples, [5, 95])

    # Rescale intensities
    if rescale:
        #rescale_array_(cropped_array, min_val, max_val, new_min=0.0, new_max=1.0)
        rescale_array_(cropped_array, mina=p5, maxa=p95, new_min=0.0, new_max=1.0)

    # Save the cropped array as a .npy file
    np.save(output_path, cropped_array)
    print(f"Cropped data saved to {output_path}")

    # Plot a few slices from the cropped stack
    plt.figure(figsize=(10, num_slices_plot * 3))
    total_slices = cropped_array.shape[-1]
    indices_to_plot = np.linspace(0, total_slices - 1, num_slices_plot**2, dtype=int)
    for idx, slice_idx in enumerate(indices_to_plot):
        plt.subplot(num_slices_plot, num_slices_plot, idx + 1)
        plt.imshow(cropped_array[:, :, slice_idx], cmap='gray', vmin=0.0, vmax=1.0)
        plt.title(f'Slice {slice_idx}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "cropped_slices.pdf"))
    plt.close()


def tiff_stack_to_numpy(tiff_folder_path, dtype=np.uint16):
    """
    Converts a stack of TIFF images in a folder to a 3D numpy array in a memory-efficient way.

    Args:
        tiff_folder_path (str): Path to the folder containing the TIFF images.
        dtype (type): The desired data type for the output 3D array (default: np.uint8).

    Returns:
        np.ndarray: A 3D numpy array representing the stacked images.
    """
    # List all TIFF files in the folder and sort them
    tiff_files = sorted([f for f in os.listdir(tiff_folder_path) if f.lower().endswith('.tiff') or f.lower().endswith('.tif')])

    if not tiff_files:
        raise ValueError("No TIFF files found in the specified folder.")

    # Read the first image to determine dimensions
    first_image_path = os.path.join(tiff_folder_path, tiff_files[0])
    with Image.open(first_image_path) as img:
        dtype = np.array(img).dtype if dtype is None else dtype  # infer data type from image
        img_width, img_height = img.size

    # Initialize an empty 3D numpy array
    depth = len(tiff_files)
    stacked_array = np.empty((img_height, img_width, depth), dtype=dtype)

    # Load each TIFF image into the array
    for idx, file_name in enumerate(tiff_files):
        file_path = os.path.join(tiff_folder_path, file_name)
        with Image.open(file_path) as img:
            stacked_array[:, :, idx] = np.array(img, dtype=dtype)

    return stacked_array


def crop_center(image, center_size):
    B, C, H, W, D = image.shape

    start_H = (H - center_size) // 2
    start_W = (W - center_size) // 2
    start_D = (D - center_size) // 2

    # Crop the image
    image = image[:, :, start_H:start_H + center_size, start_W:start_W + center_size, start_D:start_D + center_size]

    return image


def crop_context(image, L, level_ratio):
    """
    Crops a 3D image tensor according to the specified rules.

    Parameters:
        image (torch.Tensor): The input 3D image tensor of shape (H, W, D).
        L (int): The cropping level.
        level_ratio (int): The ratio used to calculate the new dimensions. Can be 2 or 3.

    Returns:
        torch.Tensor: The cropped 3D image tensor.
    """
    if L == 1:
        return image
    if L < 1:
        raise ValueError("L must be at least 1")
    if level_ratio not in [2, 3]:
        raise ValueError("level_ratio must be either 2 or 3")

    B, C, H, W, D = image.shape

    # Compute new dimensions based on the level_ratio
    if level_ratio == 2:
        new_H = H // (level_ratio ** (L - 1))
        new_W = W // (level_ratio ** (L - 1))
        new_D = D // (level_ratio ** (L - 1))
    elif level_ratio == 3:
        new_H = int(H * (2 / level_ratio) ** (L - 1))
        new_W = int(W * (2 / level_ratio) ** (L - 1))
        new_D = int(D * (2 / level_ratio) ** (L - 1))

    # Compute the starting indices to crop the center
    start_H = (H - new_H) // 2
    start_W = (W - new_W) // 2
    start_D = (D - new_D) // 2

    # Crop the image
    image = image[:, :, start_H:start_H + new_H, start_W:start_W + new_W, start_D:start_D + new_D]

    return image


class ImageComparisonTool3D():
    def __init__(self, patch_size_hr, upscaling_methods, unnorm=True, div_max=False, out_dtype=np.uint8, upscale_slice=False):
        self.patch_size_hr = patch_size_hr
        self.unnorm = unnorm
        self.div_max = div_max
        self.out_dtype = out_dtype
        self.upscaling_methods = upscaling_methods
        self.upscale_slice = upscale_slice

        self.upscale_func_dict = {}
        for method in upscaling_methods:
            self.upscale_func_dict[method] = self.get_upscaling_func(method=method, shape=patch_size_hr)


    def get_upscaling_func(self, method="tio_linear", shape=None):

        if method == "tio_linear":
            return tiotransforms.Resize(target_shape=shape, image_interpolation='LINEAR')
        elif method == "tio_nearest":
            return tiotransforms.Resize(target_shape=shape, image_interpolation='NEAREST')
        elif method == "tio_bspline":
            return tiotransforms.Resize(target_shape=shape, image_interpolation='BSPLINE')
        else:
            raise NotImplementedError('Upsampling method %s not implemented.' % method)

    def get_slice(self, image, slice_idx, axis=0):

        if len(image.shape) == 3:
            if axis == 0:
                return torch.from_numpy(image[slice_idx, :, :]).unsqueeze(0)
            elif axis == 1:
                return torch.from_numpy(image[:, slice_idx, :]).unsqueeze(0)
            elif axis == 2:
                return torch.from_numpy(image[:, :, slice_idx]).unsqueeze(0)
            else:
                raise ValueError(f"Slice axis must be 0, 1, or 2 but got {axis}")
        elif len(image.shape) == 4:
            if axis == 0:
                return torch.from_numpy(image[:, slice_idx, :, :])
            elif axis == 1:
                return torch.from_numpy(image[:, :, slice_idx, :])
            elif axis == 2:
                return torch.from_numpy(image[:, :, :, slice_idx])
            else:
                raise ValueError(f"Slice axis must be 0, 1, or 2 but got {axis}")
        else:
            raise ValueError(f"Length of image shape must 3 or 4, got {len(image.shape)}")

    def get_comparison_image(self, img_dict, slice_idx=None, axis=2):
        if slice_idx is None:
            slice_idx = img_dict['H'].shape[1 + axis] // 2

        # Upscale LR volumes and extract slice
        img_list = []
        for key, func in self.upscale_func_dict.items():

            for key in img_dict:
                if isinstance(img_dict[key], torch.Tensor):
                    img_dict[key] = img_dict[key].cpu()

            if img_dict['H'].shape != img_dict['L'].shape:  # upscale LR volume to match HR
                if self.upscale_slice:
                    img = self.get_slice(img_dict['L'], slice_idx // (img_dict['H'].shape[-1] // img_dict['L'].shape[-1]), axis).unsqueeze(0)
                    up_lr_slice = func(img).squeeze(0)  # index slice, then 2D upscale
                else:
                    img = func(img_dict['L'])
                    up_lr_slice = self.get_slice(img, slice_idx, axis)  # 3D upscale, then index slice

            else:
                up_lr_slice = self.get_slice(img_dict['L'], slice_idx, axis)

            img_list.append(up_lr_slice)

        hr_slice = self.get_slice(img_dict['H'], slice_idx, axis)  # C, H, W, D -> C, H, W
        sr_slice = self.get_slice(img_dict['E'], slice_idx, axis)

        img_list.append(sr_slice)
        img_list.append(hr_slice)

        row = torch.stack(img_list)
        grid = make_grid(row, nrow=len(row), padding=0).permute(1, 2, 0)  # make grid, then permute to H, W, C because WandB assumes channel last
        grid_image = self.unnorm_and_rescale(grid, self.out_dtype)

        return grid_image

    def unnorm_and_rescale(self, img, out_dtype=np.uint8):

        if self.unnorm:
            img = (img/2) + 0.5  # unnormalize from [-1; 1] to [0; 1]

        if self.div_max:
            img = img / torch.max(img)  # Divide by max to ensure output is between 0 and 1
            img[img < 0] = 0

        img = torch.clamp(img, min=0.0, max=1.0)  # Clip values
        if out_dtype == torch.uint8:
            img = torch.squeeze((torch.round(img * 255)).type(torch.uint8))  # Convert to uint8
        elif out_dtype == np.uint8:
            img = (np.round(img.numpy() * 255)).astype(np.uint8).squeeze()  # Convert to numpy uint8
        elif out_dtype == np.uint16:
            img = (np.round(img.numpy() * 65535)).astype(np.uint16).squeeze()  # Convert to numpy uint16 (unsupported in torch)

        return img


def rescale255(images):

    images = images / np.max(images)  # Ensure output is between 0 and 1
    images[images < 0] = 0
    images = (images * 255).astype(np.uint8)  # Convert to uint8 in range [0; 255]

    return images

def unnormalize_image(img):
    """
    If the input images are normalized, which is generally recommended, they should be unnormalized before visualization
    This function unnormalizes an image by assuming zero mean and unit standard deviation.
    :param img: Input image (normalized)
    :return: unnormalized output image
    """
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if len(img.size()) > 2:
        return np.transpose(npimg, (1, 2, 0))
    else:
        return npimg

def numel(m: torch.nn.Module, only_trainable: bool = False):
    """
    Returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)

def toggle_grad(model, on_or_off):
    # https://github.com/ajbrock/BigGAN-PyTorch/blob/master/utils.py#L674
    for param in model.parameters():
        param.requires_grad = on_or_off

def ICNR(tensor, initializer, upscale_factor=2, *args, **kwargs):
    "tensor: the 2-dimensional Tensor or more"
    upscale_factor_squared = upscale_factor * upscale_factor
    assert tensor.shape[0] % upscale_factor_squared == 0, \
        ("The size of the first dimension: "
         f"tensor.shape[0] = {tensor.shape[0]}"
         " is not divisible by square of upscale_factor: "
         f"upscale_factor = {upscale_factor}")
    sub_kernel = torch.empty(tensor.shape[0] // upscale_factor_squared, *tensor.shape[1:])
    sub_kernel = initializer(sub_kernel, *args, **kwargs)
    return sub_kernel.repeat_interleave(upscale_factor_squared, dim=0)

def ICNR3D(tensor, initializer, upscale_factor=2, *args, **kwargs):
    "tensor: the 2-dimensional Tensor or more"
    upscale_factor_cubed = upscale_factor * upscale_factor * upscale_factor
    assert tensor.shape[0] % upscale_factor_cubed == 0, \
        ("The size of the first dimension: "
         f"tensor.shape[0] = {tensor.shape[0]}"
         " is not divisible by upscale_factor^3: "
         f"upscale_factor = {upscale_factor}")
    sub_kernel = torch.empty(tensor.shape[0] // upscale_factor_cubed, *tensor.shape[1:])
    sub_kernel = initializer(sub_kernel, *args, **kwargs)
    return sub_kernel.repeat_interleave(upscale_factor_cubed, dim=0)

def deconvICNR(tensor, initializer, upscale_factor=2, *args, **kwargs):
    "tensor: the 2-dimensional Tensor or more"
    kernel_size = torch.tensor(tensor.shape[2:])
    upscale_factor_squared = upscale_factor * upscale_factor
    assert tensor.shape[-1] % upscale_factor == 0, \
        ("The size of the kernel: "
         f"tensor.shape[0] = {tensor.shape[-1]}"
         " is not divisible by the upscale_factor: "
         f"upscale_factor = {upscale_factor}")
    sub_kernel_size = kernel_size // upscale_factor
    sub_kernel = torch.empty(*tensor.shape[0:2], *sub_kernel_size)  # assumes 3D kernel
    sub_kernel = initializer(sub_kernel, *args, **kwargs)
    return torch.nn.functional.interpolate(sub_kernel, mode='nearest', scale_factor=upscale_factor)


def read_parameters(file_path):
    parameters = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=')
                parameters[key.strip()] = value.strip()
    return parameters

def gauss_dev_1D(t):
    s = np.sqrt(t)
    x = np.array(range(int(-3*s), int(3*s+1)))
    return -(x/t*np.sqrt(2*np.pi*t))*np.exp(-x*x/2*t)

def get_gaussian_kernel(sigma, also_dg=False, also_ddg=False, radius=None):
    # order only 0 or 1

    if radius is None:
        radius = max(int(4 * sigma + 0.5), 1)  # similar to scipy _gaussian_kernel1d but never smaller than 1
    x = torch.arange(-radius, radius + 1)

    sigma2 = sigma * sigma
    phi_x = torch.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    if also_dg:
        return phi_x, phi_x * -x / sigma2
    elif also_ddg:
        return phi_x, phi_x * -x / sigma2, phi_x * ((x**2/sigma2**2) - (1/sigma2))
    else:
        return phi_x

def test_3d_gaussian_blur(vol, ks, blur_sigma):

    t = blur_sigma**2
    radius = math.ceil(3 * blur_sigma)
    x = np.arange(-radius, radius + 1)
    if vol.dtype == torch.double:
        k = torch.from_numpy(1.0 / np.sqrt(2.0 * np.pi * t) * np.exp((-x * x) / (2.0 * t))).double()
    elif vol.dtype == torch.float:
        k = torch.from_numpy(1.0 / np.sqrt(2.0 * np.pi * t) * np.exp((-x * x) / (2.0 * t))).float()
    elif vol.dtype == torch.short:
        k = torch.from_numpy(1.0 / np.sqrt(2.0 * np.pi * t) * np.exp((-x * x) / (2.0 * t))).short()
    elif vol.dtype == torch.bfloat16:
        k = torch.from_numpy(1.0 / np.sqrt(2.0 * np.pi * t) * np.exp((-x * x) / (2.0 * t))).bfloat16()

    # Separable 1D convolution in 3 directions
    k1d = k.view(1, 1, -1)
    for _ in range(3):
        vol = F.conv1d(vol.reshape(-1, 1, vol.size(2)), k1d, padding=ks // 2).view(*vol.shape)
        vol = vol.permute(2, 0, 1)

    vol = vol.reshape(1, *vol.shape)
    return vol


if __name__ == "__main__":

    # Example usage
    path = '/work3/s173944/Python/venv_srgan/3D_datasets/datasets/danmax/binning_bone'

    hdf5_path = f"{path}/bin4x4/bone_1/scan-6858-6870_recon.h5"  # bone_1 bin4x4
    #hdf5_path = f"{path}/bin2x2/bone_1/scan-6842-6854_recon.h5"  # bone_1 bin2x2
    #hdf5_path = f"{path}/bin4x4/bone_2/scan-7212-7221_recon.h5"  # bone_2 bin4x4
    #hdf5_path = f"{path}/bin2x2/bone_2/scan-7200-7209_recon.h5"  # bone_2 bin2x2

    crop_val = 1651 if hdf5_path.find("bin2x2") > 0 else 825

    data_name = 'dataset'  # Name of the dataset inside the HDF5 file
    crop_indices = {
        'start_row': crop_val, # 1151 for bin2x2 and 575 for bin4x4
        'end_row': crop_val,
        'start_col': crop_val,
        'end_col': crop_val
    }  # Define cropping dimensions
    output_path = 'bone_1_crop_norm.npy'  # Path to save the output .npy file

    # print HDF5 tree
    print_hdf5_tree(hdf5_path)

    # Process and crop HDF5
    process_hdf5(hdf5_path, data_name, crop_indices, output_path, rescale=True, num_slices_plot=4)

    if False:
        # Convert TIFF stack to Numpy
        import matplotlib.pyplot as plt
        path = "../../Vedrana_master_project/3D_datasets/datasets/nanoCT/Scan1_test"
        image = tiff_stack_to_numpy(f"{path}/Recon_Pag_ram-lak_ringnorm9")
        crop = image[130:-130, 140:-120, :]
        np.save(os.path.join(path, "recon_pag_crop.npy"), crop)