from typing import Hashable, Mapping
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.fft as fft
import torch.nn.functional as F
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms import Transform, MapTransform
from monai.utils.enums import TransformBackends
from scipy.fft import fftn, ifftn, fftshift, ifftshift
from scipy.ndimage import zoom


def kspace_trunc_old(data_tensor, Factor_truncate=4.0, norm_val=4095.0, slice_dim=1, keep_shape=True):
    '''
    K-space truncation function.
    :param data_tensor: input tensor containing volumetric data
    :param Factor_truncate: Inverse portion of K-space to set to zero. Factor_truncate = 4.0 is equivalent to setting a
    4th of the K-space equal to zero. Must be greater than 2.0, otherwise the output will contain only NaN
    :param norm_val: Value to normalize input image with, as input must be in range [0; 1]
    :param scale_output: Does not work right now
    :param slice_dim: Dimension of the slice direction in the tomographic volume eg. (B, Z, X, Y) has slide_dim=1.
    We truncate the k-space only along X, Y and not Z.
    :return:
    '''

    if len(data_tensor.shape) > 3:  # squeeze if 4D tensor
        data_float = data_tensor.squeeze().float()
    else:
        data_float = data_tensor.float()
    #data_float = data_tensor.float()
    #print("Max before norm", torch.max(data_float))
    #print("Min before norm", torch.min(data_float))
    #norm_val = torch.max(data_float)
    if norm_val == None:  # if not provided, normalize with max value
        norm_val = torch.max(data_float)

    data_norm = data_float / norm_val
    kData = fft.fftshift(fft.fftn(fft.ifftshift(data_norm), norm="forward"))
    kData_truncate = kData.clone()

    if slice_dim == 1:  # Z, X, Y
        x_range = round(kData.shape[1] / Factor_truncate)
        y_range = round(kData.shape[2] / Factor_truncate)
        kData_truncate[:, :x_range, :] = 0
        kData_truncate[:, :, :y_range] = 0
        kData_truncate[:, -x_range:, :] = 0
        kData_truncate[:, :, -y_range:] = 0
        nz, nx, ny = data_norm.shape

        kData_crop = kData[:, x_range:nx - x_range:, y_range:ny - y_range:]

    elif slice_dim == 2:  # X, Z, Y
        x_range = round(kData.shape[0] / Factor_truncate)
        y_range = round(kData.shape[2] / Factor_truncate)
        kData_truncate[:x_range, :, :] = 0
        kData_truncate[:, :, :y_range] = 0
        kData_truncate[-x_range:, :, :] = 0
        kData_truncate[:, :, -y_range:] = 0
        nx, nz, ny = data_norm.shape

        kData_crop = kData[x_range:nx - x_range:, :, y_range:ny - y_range:]

    elif slice_dim == 3:  # X, Y, Z
        x_range = round(kData.shape[0] / Factor_truncate)
        y_range = round(kData.shape[1] / Factor_truncate)
        kData_truncate[:x_range, :, :] = 0
        kData_truncate[:, :y_range, :] = 0
        kData_truncate[-x_range:, :, :] = 0
        kData_truncate[:, -y_range:, :] = 0
        nx, ny, nz = data_norm.shape

        kData_crop = kData[x_range:nx - x_range:, y_range:ny - y_range:, :]

    if keep_shape == False:
        # Adjust network or input such we can deal with 32x32x64 shapes for instance
        # Or just interpolate back up to 64x64x64. We could also just take every other slice.
        out_crop = fft.fftshift(fft.ifftn(fft.ifftshift(kData_crop), norm="backward"))
        out_real = torch.abs(out_crop)
        out_norm = out_real / torch.max(out_real)
        out_final = out_norm * norm_val
        return out_final.unsqueeze(0)

    kout = kData_truncate
    out = fft.fftshift(fft.ifftn(fft.ifftshift(kout), norm="backward"))
    out_real = torch.abs(out)

    out_norm = out_real / torch.max(out_real)
    out_float = torch.zeros_like(out_real)
    if slice_dim == 1:
        for j in range(nz):
            out_float[j, :, :] = F.interpolate(out_norm[j, :, :].unsqueeze(0).unsqueeze(0), size=(nx, ny), mode='bilinear').squeeze()
    elif slice_dim == 2:
        for j in range(nz):
            out_float[:, j, :] = F.interpolate(out_norm[:, j, :].unsqueeze(0).unsqueeze(0), size=(nx, ny), mode='bilinear').squeeze()
    elif slice_dim == 3:
        for j in range(nz):
            out_float[:, :, j] = F.interpolate(out_norm[:, :, j].unsqueeze(0).unsqueeze(0), size=(nx, ny), mode='bilinear').squeeze()

    #out_final = torch.round(out_float * norm_val).to(torch.int16)
    out_final = out_float * norm_val
    return out_final.unsqueeze(0)  # ensure channel dimension first

def kspace_trunc_numpy(data_tensor, Factor_truncate=4.0, norm_val=4095.0):

    data_float = data_tensor.astype(float)
    data_norm = data_float / norm_val
    kData = fftshift(fftn(ifftshift(data_norm)))
    x_range = round(kData.shape[1] / Factor_truncate)
    y_range = round(kData.shape[2] / Factor_truncate)
    kData_truncate = kData.copy()
    kData_truncate[:, :x_range, :] = 0
    kData_truncate[:, :, :y_range] = 0
    kData_truncate[:, -x_range:, :] = 0
    kData_truncate[:, :, -y_range:] = 0

    kout = kData_truncate
    out = fftshift(ifftn(ifftshift(kout)))
    out_real = np.abs(out)

    nz, nx, ny = data_norm.shape
    out_norm = out_real / np.max(out_real)
    out_float = np.zeros_like(out_real)
    for j in range(nz):
        out_float[j, :, :] = zoom(out_norm[j, :, :], (nx / ny, ny / nx), order=1)

    out_final = np.round(out_float * norm_val).astype(np.int16)
    return out_final

def plot_histograms(tensor1, tensor2, bins=100, title='Histogram of Tensors'):
    flattened_tensor1 = tensor1.reshape(-1)
    flattened_tensor2 = tensor2.reshape(-1)

    plt.hist(flattened_tensor1.numpy(), bins=bins, alpha=0.5, label='Tensor 1')
    plt.hist(flattened_tensor2.numpy(), bins=bins, alpha=0.5, label='Tensor 2')
    plt.title(title)
    plt.legend()


def kspace_trunc_func(data_tensor, trunc_factor=3.0, norm_val=1.0, slice_dim=1):
    '''
    K-space truncation function.
    :param data_tensor: input tensor containing volumetric data
    :param trunc_factor: Inverse portion of K-space to set to zero. Factor_truncate = 4.0 is equivalent to setting a
    4th of the K-space equal to zero on both sides. Must be greater than 2.0, otherwise the output will contain only NaN
    :param norm_val: Value to normalize input image with, as input must be in range [0; 1]
    :param scale_output: Does not work right now
    :param slice_dim: Dimension of the slice direction in the tomographic volume eg. (B, Z, X, Y) has slide_dim=1.
    We truncate the k-space only along X, Y and not Z.
    :return:
    '''

    # Ensure float32
    if data_tensor.dtype is not torch.float32:
        data_tensor = data_tensor.float()

    # Squeeze if 4D tensor
    if len(data_tensor.shape) > 3:
        data_tensor = data_tensor.squeeze()

    # if not provided, compute max value
    if norm_val == None:
        norm_val = torch.max(data_tensor)

    data_tensor /= norm_val
    kData = fft.fftshift(fft.fftn(fft.ifftshift(data_tensor), norm="forward"))
    kData_truncate = kData.clone()

    if slice_dim == 1:  # Z, X, Y
        x_range = round(kData.shape[1] / trunc_factor)
        y_range = round(kData.shape[2] / trunc_factor)
        kData_truncate[:, :x_range, :] = 0
        kData_truncate[:, :, :y_range] = 0
        kData_truncate[:, -x_range:, :] = 0
        kData_truncate[:, :, -y_range:] = 0
        nz, nx, ny = data_tensor.shape

    elif slice_dim == 2:  # X, Z, Y
        x_range = round(kData.shape[0] / trunc_factor)
        y_range = round(kData.shape[2] / trunc_factor)
        kData_truncate[:x_range, :, :] = 0
        kData_truncate[:, :, :y_range] = 0
        kData_truncate[-x_range:, :, :] = 0
        kData_truncate[:, :, -y_range:] = 0
        nx, nz, ny = data_tensor.shape

    elif slice_dim == 3:  # X, Y, Z
        x_range = round(kData.shape[0] / trunc_factor)
        y_range = round(kData.shape[1] / trunc_factor)
        kData_truncate[:x_range, :, :] = 0
        kData_truncate[:, :y_range, :] = 0
        kData_truncate[-x_range:, :, :] = 0
        kData_truncate[:, -y_range:, :] = 0
        nx, ny, nz = data_tensor.shape

    # Perform inverse FFT
    data_out = fft.fftshift(fft.ifftn(fft.ifftshift(kData_truncate), norm="backward"))
    data_out = torch.abs(data_out)

    # Normalize and interpolate
    data_out /= torch.max(data_out)

    # Rearrange slice dimension to batch dimension
    if slice_dim == 1:
        data_out = data_out.unsqueeze(0).permute(1, 0, 2, 3).contiguous() # 1, Z, X, Y -> Z, 1, X, Y
    elif slice_dim == 2:
        data_out = data_out.unsqueeze(0).permute(2, 0, 1, 3).contiguous() # 1, X, Z, Y -> Z, 1, X, Y
    elif slice_dim == 3:
        data_out = data_out.unsqueeze(0).permute(3, 0, 1, 2).contiguous() # 1, X, Y, Z -> Z, 1, X, Y

    # Interpolate slices to original size
    data_out = F.interpolate(data_out, size=(nx, ny), mode='bilinear')

    # Rearrange back to channel dimension first
    if slice_dim == 1:
        data_out = data_out.permute(1, 0, 2, 3).contiguous() # Z, 1, X, Y -> 1, Z, X, Y
    elif slice_dim == 2:
        data_out = data_out.permute(1, 2, 0, 3).contiguous() # Z, 1, X, Y -> 1, X, Z, Y
    elif slice_dim == 3:
        data_out = data_out.permute(1, 2, 3, 0).contiguous() # Z, 1, X, Y -> 1, X, Y, Z

    return data_out * norm_val  # ensure channel dimension first



class KspaceTrunc(Transform):
    """
    K-space truncation transform.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, trunc_factor, norm_val, slice_dim) -> None:
        self.trunc_factor = trunc_factor
        self.norm_val = norm_val
        self.slice_dim = slice_dim

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        out = kspace_trunc_func(img, self.trunc_factor, self.norm_val, self.slice_dim)
        return out


class KspaceTruncd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`KspaceTrunc`.
    """

    backend = KspaceTrunc.backend

    def __init__(
        self,
        keys: KeysCollection,
        trunc_factor: float = 3.0,
        norm_val: float = 1.0,
        slice_dim: int = 1,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.converter = KspaceTrunc(trunc_factor, norm_val, slice_dim)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d



if __name__ == "__main__":

    data = np.load("../../Vedrana_master_project/3D_datasets/datasets/2022_QIM_52_Bone/femur_001/micro/volume/f_001.npy")
    data = torch.tensor(data).unsqueeze(0)

    #img = nib.load('139839_3T_T1w_MPR1.nii')
    #img = nib.load('BraTS-GLI-00002-000-t1c.nii.gz')
    #img = nib.load('IXI002-Guys-0828-T1.nii')
    #data = torch.Tensor(img.get_fdata()).unsqueeze(0)


    print(data.shape)
    #data = data.permute(0, 3, 1, 2)  # z-direction first
    #print(data.shape)
    print("Max and Min data", torch.max(data), torch.min(data))
    data = data / torch.max(data)

    norm_val = 1.0
    out_tf_4 = kspace_trunc_func(data, 4.0, norm_val=norm_val, slice_dim=3)
    print("Max and Min out", torch.max(out_tf_4), torch.min(out_tf_4))

    out_tf_3 = kspace_trunc_func(data, 3.0, norm_val=norm_val, slice_dim=3)
    out_tf_2p5 = kspace_trunc_func(data, 2.5, norm_val=norm_val, slice_dim=3)

    plt.figure(figsize=(18,5))
    plt.subplot(1, 4, 1)
    plt.imshow(data[0, :, :, 200], cmap='gray', vmin=0.0, vmax=norm_val)
    plt.title("Reference")

    plt.subplot(1, 4, 2)
    plt.imshow(out_tf_4[0, :, :, 200], cmap='gray', vmin=0.0, vmax=norm_val)
    plt.title("k-space truncation: %d%%" % 50)

    plt.subplot(1, 4, 3)
    plt.imshow(out_tf_3[0, :, :, 200], cmap='gray', vmin=0.0, vmax=norm_val)
    plt.title("k-space truncation: %d%%" % 67)

    plt.subplot(1, 4, 4)
    plt.imshow(out_tf_2p5[0, :, :, 200], cmap='gray', vmin=0.0, vmax=norm_val)
    plt.title("k-space truncation: %d%%" % 80)

    plt.figure()
    plot_histograms(out_tf_4, data)

    plt.show()


    print("Done")

