import numpy as np
import torch
import torch.nn.functional as F
import torchio.transforms as tiotransforms
from torchvision.utils import make_grid
from skimage.transform import downscale_local_mean

class DegradationComparisonTool3D():
    def __init__(self, patch_size_lr, upscaling_methods, unnorm=True, div_max=False, out_dtype=np.uint8):
        self.patch_size_lr = patch_size_lr
        self.unnorm = unnorm
        self.div_max = div_max
        self.out_dtype = out_dtype
        self.upscaling_methods = upscaling_methods

        self.upscale_func_dict = {}
        for method in upscaling_methods:
            self.upscale_func_dict[method] = self.get_downscaling_func(method=method, shape=patch_size_lr)


    def get_downscaling_func(self, method="tio_linear", shape=None):

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
                im = image[slice_idx, :, :]
            elif axis == 1:
                im = image[:, slice_idx, :]
            elif axis == 2:
                im = image[:, :, slice_idx]
            else:
                raise ValueError(f"Slice axis must be 0, 1, or 2 but got {axis}")
        elif len(image.shape) == 4:
            if axis == 0:
                im = image[:, slice_idx, :, :]
            elif axis == 1:
                im = image[:, :, slice_idx, :]
            elif axis == 2:
                im = image[:, :, :, slice_idx]
            else:
                raise ValueError(f"Slice axis must be 0, 1, or 2 but got {axis}")
        else:
            raise ValueError(f"Length of image shape must 3 or 4, got {len(image.shape)}")

        if not isinstance(im, torch.Tensor):
            im = torch.from_numpy(im)

        if len(image.shape) == 3:
            im = im.unsqueeze(0)

        return im


    def get_comparison_image(self, img_dict, slice_idx=None, axis=2):
        if slice_idx is None:
            slice_idx = img_dict['L'].shape[1 + axis] // 2

        # Downscale HR volumes and extract slice
        img_list = []

        img_list.append(self.get_slice(img_dict['L'], slice_idx, axis).float())
        img_list.append(self.get_slice(img_dict['E'], slice_idx, axis).float())

        for key, func in self.upscale_func_dict.items():

            for key in img_dict:
                if isinstance(img_dict[key], torch.Tensor):
                    img_dict[key] = img_dict[key].cpu()

            down_hr_slice = self.get_slice(func(img_dict['H']), slice_idx, axis)  # 3D downscale, then index slice
            img_list.append(down_hr_slice.float())

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



class DegradationComparisonTool3D_V2():
    def __init__(self, patch_size_hr, down_factor, upscaling_methods, unnorm=True, div_max=False, out_dtype=np.uint8, plot_synth_LR=False):
        self.patch_size_hr = patch_size_hr
        self.down_factor = down_factor
        self.unnorm = unnorm
        self.div_max = div_max
        self.out_dtype = out_dtype
        self.upscaling_methods = upscaling_methods
        self.plot_synth_LR = plot_synth_LR  # Whether to include the synthetic LR image
        self.synth_up_func = tiotransforms.Resize(target_shape=patch_size_hr, image_interpolation='NEAREST')
        self.synth_down_func = tiotransforms.Resize(target_shape=patch_size_hr // down_factor, image_interpolation='NEAREST')

        self.upscale_func_dict = {}
        for method in upscaling_methods:
            self.upscale_func_dict[method] = self.get_downscaling_func(method=method, shape=patch_size_hr)


    def get_downscaling_func(self, method="tio_linear", shape=None):

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
                im = image[slice_idx, :, :]
            elif axis == 1:
                im = image[:, slice_idx, :]
            elif axis == 2:
                im = image[:, :, slice_idx]
            else:
                raise ValueError(f"Slice axis must be 0, 1, or 2 but got {axis}")
        elif len(image.shape) == 4:
            if axis == 0:
                im = image[:, slice_idx, :, :]
            elif axis == 1:
                im = image[:, :, slice_idx, :]
            elif axis == 2:
                im = image[:, :, :, slice_idx]
            else:
                raise ValueError(f"Slice axis must be 0, 1, or 2 but got {axis}")
        else:
            raise ValueError(f"Length of image shape must 3 or 4, got {len(image.shape)}")

        if not isinstance(im, torch.Tensor):
            im = torch.from_numpy(im)

        if len(image.shape) == 3:
            im = im.unsqueeze(0)

        return im


    def get_comparison_image(self, img_dict, slice_idx=None, axis=2):
        if slice_idx is None:
            slice_idx = img_dict['L'].shape[1 + axis] // 2

        # Downscale HR volumes and extract slice
        img_list = []

        img_list.append(self.get_slice(self.synth_up_func(img_dict['L']), slice_idx, axis).float())

        for key, func in self.upscale_func_dict.items():

            for key in img_dict:
                if isinstance(img_dict[key], torch.Tensor):
                    img_dict[key] = img_dict[key].cpu()

            img_list.append(self.get_slice(func(img_dict['E']), slice_idx, axis).float())

        if self.plot_synth_LR:
            synth_down = img_dict['H'].squeeze().numpy()
            if self.down_factor >= 2:
                synth_down = downscale_local_mean(synth_down, factors=(2, 2, 2))
            if self.down_factor >= 4:
                synth_down = downscale_local_mean(synth_down, factors=(2, 2, 2))
            if self.down_factor >= 8:
                synth_down = downscale_local_mean(synth_down, factors=(2, 2, 2))

            synth_down = self.synth_up_func(torch.from_numpy(synth_down).unsqueeze(0))
            img_list.append(self.get_slice(synth_down, slice_idx, axis).float())

        img_list.append(self.get_slice(img_dict['H'], slice_idx, axis).float())

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