import os
import math
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
import torchio.transforms as tiotransforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

import h5py
import monai.transforms as mt

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

    print("Rescaling array...")
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
            slice_ = data[i, :, :]
            cropped_slice = slice_[start_row:-end_row, start_col:-end_col]
            #cropped_slices.append(cropped_slice)
            cropped_array[:, :, i] = cropped_slice.astype(np.float32)

            # update min and max
            slice_min = cropped_slice.min()
            slice_max = cropped_slice.max()
            min_val = min(slice_min, min_val)
            max_val = max(slice_max, max_val)

            # Sample 1000 points randomly and append to list
            samples.extend(np.random.choice(cropped_slice.flatten(), 1000, replace=False))

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
    def __init__(self, patch_size_hr, upscaling_methods, unnorm=True, div_max=False, out_dtype=np.uint8):
        self.patch_size_hr = patch_size_hr
        self.unnorm = unnorm
        self.div_max = div_max
        self.out_dtype = out_dtype
        self.upscaling_methods = upscaling_methods

        self.upscale_func_dict = {}
        for method in upscaling_methods:
            self.upscale_func_dict[method] = self.get_upscaling_func(method=method, size=patch_size_hr)


    def get_upscaling_func(self, method="tio_linear", size=None):

        if method == "tio_linear":
            return tiotransforms.Resize(target_shape=size, image_interpolation='LINEAR')
        elif method == "tio_nearest":
            return tiotransforms.Resize(target_shape=size, image_interpolation='NEAREST')
        elif method == "tio_bspline":
            return tiotransforms.Resize(target_shape=size, image_interpolation='BSPLINE')
        else:
            raise NotImplementedError('Upsampling method %s not implemented.' % method)


    def get_comparison_image(self, img_dict, slice_idx=None):
        if slice_idx is None:
            slice_idx = img_dict['H'].shape[-1] // 2

        # Upscale LR volumes and extract slice
        img_list = []
        for key, func in self.upscale_func_dict.items():
            if img_dict['H'].shape != img_dict['L'].shape:  # upscale LR volume to match HR
                up_lr_slice = func(img_dict['L'].cpu())[:, :, :, slice_idx]

            else:
                up_lr_slice = img_dict['L'].cpu()[:, :, :, slice_idx]

            img_list.append(up_lr_slice)

        hr_slice = img_dict['H'][:, :, :, slice_idx].cpu()  # C, H, W, D -> C, H, W
        sr_slice = img_dict['E'][:, :, :, slice_idx].cpu()

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