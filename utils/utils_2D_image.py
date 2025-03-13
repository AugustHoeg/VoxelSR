import os
import math
from PIL import Image

import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
import torchio.transforms as tiotransforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def upscale_slices_upfactor(model, img_L, img_H, batch_size_2D, up_factor=2):

    B, C, H, W, D = img_H.shape
    b, c, h, w, d = img_L.shape

    img_H_p = img_H.permute(0, 4, 1, 2, 3).reshape(B * D, C, H, W).as_tensor()
    img_L_p = torch.zeros((b, c, h, w, D))
    img_L_p[..., ::up_factor] = img_L
    img_L_p = img_L_p.permute(0, 4, 1, 2, 3).reshape(B * D, c, h, w)

    output_tensor = torch.zeros_like(img_H_p)
    # Loop over the first dimension in sizes of batch_size
    for i in tqdm(range(0, img_L_p.shape[0], batch_size_2D*up_factor), desc='Upscaling slices', mininterval=2):
        batch_L = img_L_p[i:i + batch_size_2D*up_factor][::up_factor]
        model.feed_data({'L': batch_L}, need_H=False)
        model.netG_forward()
        output_tensor[i:i + batch_size_2D*up_factor][::up_factor] = model.E

    # Reshape the tensor back to (B, C, H, W, D)
    output_tensor = output_tensor.reshape(B, D, C, H, W).permute(0, 2, 3, 4, 1).contiguous()
    return output_tensor


def upscale_slices(model, img_L, img_H, batch_size_2D):
    # Reshape the tensor from (B, C, H, W, D) to (B*D, C, H, W)
    B, C, H, W, D = img_H.shape
    b, c, h, w, d = img_L.shape
    img_H_p = img_H.permute(0, 4, 1, 2, 3).reshape(B * D, C, H, W).as_tensor()
    img_L_p = img_L.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w).as_tensor()

    output_tensor = torch.zeros_like(img_H_p)
    # Loop over the first dimension in sizes of batch_size
    for i in tqdm(range(0, img_L_p.shape[0], batch_size_2D), desc='Upscaling slices', mininterval=2):
        batch_L = img_L_p[i:i + batch_size_2D]
        model.feed_data({'L': batch_L}, need_H=False)
        model.netG_forward()
        output_tensor[i:i + batch_size_2D] = model.E

    # Reshape the tensor back to (B, C, H, W, D)
    output_tensor = output_tensor.reshape(B, D, C, H, W).permute(0, 2, 3, 4, 1).contiguous()
    return output_tensor


class ImageComparisonTool2D():
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
            return tiotransforms.Resize(target_shape=(size, size, 1), image_interpolation='LINEAR')
        elif method == "tio_nearest":
            return tiotransforms.Resize(target_shape=(size, size, 1), image_interpolation='NEAREST')
        elif method == "tio_bspline":
            return tiotransforms.Resize(target_shape=(size, size, 1), image_interpolation='BSPLINE')
        else:
            raise NotImplementedError('Upsampling method %s not implemented.' % method)


    def get_comparison_image(self, img_dict):

        # Upscale LR images
        img_list = []
        for key, func in self.upscale_func_dict.items():
            if img_dict['H'].shape != img_dict['L'].shape:  # upscale LR image to match HR image
                up_lr_slice = func(img_dict['L'].cpu().unsqueeze(-1)).squeeze(-1)  # Add z dim for TorchIO resize
            else:
                up_lr_slice = img_dict['L'].cpu()
            img_list.append(up_lr_slice)

        hr_slice = img_dict['H'].cpu() # HR image
        sr_slice = img_dict['E'].cpu() # SR image

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
