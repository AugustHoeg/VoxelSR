import math

import numpy as np
import torch
import monai.transforms as mt
from monai.transforms import Randomizable, Transform, MapTransform

from data.kspace import KspaceTruncd
from utils.utils_3D_image import test_3d_gaussian_blur
from utils.utils_ARSSR import make_coord

from data.train_transforms_implicit import RandomCropPairImplicitd


# from monai.transforms import RandomizableTransform, Rand3DElastic
# from collections.abc import Hashable, Mapping, Sequence
# from monai.config import DtypeLike, KeysCollection, SequenceStr

# class CustomRand3DElasticd(RandomizableTransform, MapTransform):
#     """
#     Dictionary-based wrapper of :py:class:`monai.transforms.Rand3DElastic`.
#     """
#
#     backend = Rand3DElastic.backend
#
#     def __init__(
#         self,
#         keys: KeysCollection,
#         sigma_range: tuple[float, float],
#         magnitude_range: tuple[float, float],
#         spatial_size: tuple[int, int, int] | int | None = None,
#         prob: float = 0.1,
#         rotate_range: Sequence[tuple[float, float] | float] | float | None = None,
#         shear_range: Sequence[tuple[float, float] | float] | float | None = None,
#         translate_range: Sequence[tuple[float, float] | float] | float | None = None,
#         scale_range: Sequence[tuple[float, float] | float] | float | None = None,
#         mode: SequenceStr = GridSampleMode.BILINEAR,
#         padding_mode: SequenceStr = GridSamplePadMode.REFLECTION,
#         device: torch.device | None = None,
#         allow_missing_keys: bool = False,
#     ) -> None:
#         """
#         Args:
#             keys: keys of the corresponding items to be transformed.
#             sigma_range: a Gaussian kernel with standard deviation sampled from
#                 ``uniform[sigma_range[0], sigma_range[1])`` will be used to smooth the random offset grid.
#             magnitude_range: the random offsets on the grid will be generated from
#                 ``uniform[magnitude[0], magnitude[1])``.
#             spatial_size: specifying output image spatial size [h, w, d].
#                 if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
#                 the transform will use the spatial size of `img`.
#                 if some components of the `spatial_size` are non-positive values, the transform will use the
#                 corresponding components of img size. For example, `spatial_size=(32, 32, -1)` will be adapted
#                 to `(32, 32, 64)` if the third spatial dimension size of img is `64`.
#             prob: probability of returning a randomized affine grid.
#                 defaults to 0.1, with 10% chance returns a randomized grid,
#                 otherwise returns a ``spatial_size`` centered area extracted from the input image.
#             rotate_range: angle range in radians. If element `i` is a pair of (min, max) values, then
#                 `uniform[-rotate_range[i][0], rotate_range[i][1])` will be used to generate the rotation parameter
#                 for the `i`th spatial dimension. If not, `uniform[-rotate_range[i], rotate_range[i])` will be used.
#                 This can be altered on a per-dimension basis. E.g., `((0,3), 1, ...)`: for dim0, rotation will be
#                 in range `[0, 3]`, and for dim1 `[-1, 1]` will be used. Setting a single value will use `[-x, x]`
#                 for dim0 and nothing for the remaining dimensions.
#             shear_range: shear range with format matching `rotate_range`, it defines the range to randomly select
#                 shearing factors(a tuple of 6 floats for 3D) for affine matrix, take a 3D affine as example::
#
#                     [
#                         [1.0, params[0], params[1], 0.0],
#                         [params[2], 1.0, params[3], 0.0],
#                         [params[4], params[5], 1.0, 0.0],
#                         [0.0, 0.0, 0.0, 1.0],
#                     ]
#
#             translate_range: translate range with format matching `rotate_range`, it defines the range to randomly
#                 select voxel to translate for every spatial dims.
#             scale_range: scaling range with format matching `rotate_range`. it defines the range to randomly select
#                 the scale factor to translate for every spatial dims. A value of 1.0 is added to the result.
#                 This allows 0 to correspond to no change (i.e., a scaling of 1.0).
#             mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
#                 Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
#                 See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
#                 When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
#                 and the value represents the order of the spline interpolation.
#                 See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
#                 It also can be a sequence, each element corresponds to a key in ``keys``.
#             padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
#                 Padding mode for outside grid values. Defaults to ``"reflection"``.
#                 See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
#                 When `mode` is an integer, using numpy/cupy backends, this argument accepts
#                 {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
#                 See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
#                 It also can be a sequence, each element corresponds to a key in ``keys``.
#             device: device on which the tensor will be allocated.
#             allow_missing_keys: don't raise exception if key is missing.
#
#         See also:
#             - :py:class:`RandAffineGrid` for the random affine parameters configurations.
#             - :py:class:`Affine` for the affine transformation parameters configurations.
#
#         """
#         MapTransform.__init__(self, keys, allow_missing_keys)
#         RandomizableTransform.__init__(self, prob)
#         self.rand_3d_elastic = Rand3DElastic(
#             sigma_range=sigma_range,
#             magnitude_range=magnitude_range,
#             prob=1.0,  # because probability controlled by this class
#             rotate_range=rotate_range,
#             shear_range=shear_range,
#             translate_range=translate_range,
#             scale_range=scale_range,
#             spatial_size=spatial_size,
#             device=device,
#         )
#         self.mode = ensure_tuple_rep(mode, len(self.keys))
#         self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))
#
#     def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> Rand3DElasticd:
#         self.rand_3d_elastic.set_random_state(seed, state)
#         super().set_random_state(seed, state)
#         return self
#
#     def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
#         """
#         Args:
#             data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
#                 in this dictionary must be tensor like arrays that are channel first and have at most
#                 three spatial dimensions
#
#         Returns:
#             a dictionary containing the transformed data, as well as any other data present in the dictionary
#         """
#         d = dict(data)
#         first_key: Hashable = self.first_key(d)
#
#         if first_key == ():
#             out: dict[Hashable, torch.Tensor] = convert_to_tensor(d, track_meta=get_track_meta())
#             return out
#
#         self.randomize(None)
#         if isinstance(d[first_key], MetaTensor) and d[first_key].pending_operations:  # type: ignore
#             warnings.warn(f"data['{first_key}'] has pending operations, transform may return incorrect results.")
#         sp_size = fall_back_tuple(self.rand_3d_elastic.spatial_size, d[first_key].shape[1:])
#
#         # all the keys share the same random elastic factor
#         self.rand_3d_elastic.randomize(sp_size)
#
#         device = self.rand_3d_elastic.device
#         if device is None and isinstance(d[first_key], torch.Tensor):
#             device = d[first_key].device
#             self.rand_3d_elastic.set_device(device)
#         grid = create_grid(spatial_size=sp_size, device=device, backend="torch")
#         if self._do_transform:
#             gaussian = GaussianFilter(spatial_dims=3, sigma=self.rand_3d_elastic.sigma, truncated=3.0).to(device)
#             offset = torch.as_tensor(self.rand_3d_elastic.rand_offset, device=device).unsqueeze(0)
#             grid[:3] += gaussian(offset)[0] * self.rand_3d_elastic.magnitude
#             grid = self.rand_3d_elastic.rand_affine_grid(grid=grid)
#
#         for key, mode, padding_mode in self.key_iterator(d, self.mode, self.padding_mode):
#             d[key] = self.rand_3d_elastic.resampler(d[key], grid, mode=mode, padding_mode=padding_mode)  # type: ignore
#         return d


def get_context_pad_size(opt):
    if 'context_sizes' in opt.model_opt.netG:
        center_size = opt.model_opt.netG.context_sizes[-1]
        pad_size = (opt.dataset_opt.patch_size - center_size) // 2  # pad half of the context on all sides
    else:
        pad_size = 0
    return pad_size


class GlobalScaleIntensityd(MapTransform):
    def __init__(self, keys, global_min, global_max):
        super().__init__(keys)
        self.global_min = global_min
        self.global_max = global_max

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = (d[key] - self.global_min) / (self.global_max - self.global_min)
        return d

class RandomCropOld(Randomizable):
    """ Randomly crops a uniform region from both LR and HR images (supports 2D & 3D). """

    def __init__(self, patch_size_lr, up_factor, pad_size=0, input_type="3D"):
        super().__init__()
        self.size_lr = patch_size_lr
        self.up_factor = up_factor
        self.size_hr = patch_size_lr * up_factor
        self.pad_size = pad_size

        if pad_size > 0:
            self.size_hr -= 2 * up_factor * pad_size  # Adjust HR patch size

        self.spatial_dims = 3 if input_type == "3D" else 2  # 2D or 3D

        ''' # For debugging
        import matplotlib.pyplot as plt
        import torch.nn.functional as F
        slicehr = img_dict['H'][0, 380//2, :, :]
        slicelr = img_dict['L'][0, 95//2, :, :]
        slicelrup = F.interpolate(img_dict['L'].unsqueeze(0), scale_factor=self.up_factor, mode='trilinear', align_corners=True)[0, 0, 380//2, :, :]
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(slicehr, cmap='gray')
        plt.subplot(1,3,2)
        plt.imshow(slicelr, cmap='gray')
        plt.subplot(1,3,3)
        plt.imshow(slicehr-slicelrup, cmap='gray')
        plt.show()
        '''

    def __call__(self, img_dict):
        return self.crop(img_dict)

    def crop(self, img_dict):
        img_L = img_dict['L']
        img_H = img_dict['H']

        # Compute valid cropping range
        valid_range_lr = torch.tensor(img_L.shape[1:]) - self.size_lr

        # Sample a random crop position
        crop_start_lr = np.random.randint(0, valid_range_lr[:self.spatial_dims] + 1, (self.spatial_dims,))
        crop_start_hr = crop_start_lr * self.up_factor

        # Extract patches
        if self.spatial_dims == 2:  # 2D (H, W)
            # TODO: update so that slices are limit to within cube and not whole 3D image, see RandomCropLabel
            slice_idx_lr = np.random.randint(0, img_L.shape[-1])  # Random slice index
            L = img_L[:, crop_start_lr[0]:crop_start_lr[0] + self.size_lr,
                         crop_start_lr[1]:crop_start_lr[1] + self.size_lr,
                         slice_idx_lr]

            slice_idx_hr = slice_idx_lr * self.up_factor
            H = img_H[:, crop_start_hr[0]:crop_start_hr[0] + self.size_hr,
                         crop_start_hr[1]:crop_start_hr[1] + self.size_hr,
                         slice_idx_hr]
        else:  # 3D (H, W, D)
            L = img_L[:, crop_start_lr[0]:crop_start_lr[0] + self.size_lr,
                         crop_start_lr[1]:crop_start_lr[1] + self.size_lr,
                         crop_start_lr[2]:crop_start_lr[2] + self.size_lr]

            H = img_H[:, crop_start_hr[0]:crop_start_hr[0] + self.size_hr,
                         crop_start_hr[1]:crop_start_hr[1] + self.size_hr,
                         crop_start_hr[2]:crop_start_hr[2] + self.size_hr]

        return {'H': H.float(), 'L': L.float()}


class RandomCropUniform(Randomizable):
    """ Randomly crops a uniform region from both LR and HR images (supports 2D & 3D). """

    def __init__(self, patch_size_lr, up_factor, pad_size=0, input_type="3D"):
        super().__init__()
        self.size_lr = patch_size_lr
        self.up_factor = up_factor
        self.size_hr = patch_size_lr * up_factor
        self.pad_size = pad_size

        if pad_size > 0:
            self.size_hr -= 2 * up_factor * pad_size  # Adjust HR patch size

        self.spatial_dims = 3 if input_type == "3D" else 2  # 2D or 3D

    def __call__(self, img_dict):
        return self.crop(img_dict)

    def crop(self, img_dict):
        # Compute valid cropping range
        valid_range_lr = torch.tensor(img_dict['L'].shape[1:]) - self.size_lr

        # Sample a random crop position
        #crop_start_lr = np.random.randint(0, valid_range_lr[:self.spatial_dims] + 1, (self.spatial_dims,))
        crop_start_lr = np.random.randint(0, valid_range_lr + 1)
        crop_start_hr = crop_start_lr * self.up_factor

        # Get corresponding HR crop position
        crop_center_lr = crop_start_lr + self.size_lr // 2
        crop_center_hr = crop_start_hr + self.size_hr // 2

        # Extract patches
        if self.spatial_dims == 2:  # 2D (H, W)
            slice_idx_lr = np.random.randint(crop_center_lr[2] - self.size_lr // 2,
                                             crop_center_lr[2] + self.size_lr // 2)  # Random slice index within lr cube
            L = img_dict['L'][:, crop_center_lr[0] - self.size_lr // 2:crop_center_lr[0] + self.size_lr // 2,
                         crop_center_lr[1] - self.size_lr // 2:crop_center_lr[1] + self.size_lr // 2,
                         slice_idx_lr]

            slice_idx_hr = slice_idx_lr * self.up_factor
            H = img_dict['H'][:, crop_center_hr[0] - self.size_hr // 2:crop_center_hr[0] + self.size_hr // 2,
                         crop_center_hr[1] - self.size_hr // 2:crop_center_hr[1] + self.size_hr // 2,
                         slice_idx_hr]
        else:  # 3D (H, W, D)
            L = img_dict['L'][:, crop_center_lr[0] - self.size_lr // 2:crop_center_lr[0] + self.size_lr // 2,
                         crop_center_lr[1] - self.size_lr // 2:crop_center_lr[1] + self.size_lr // 2,
                         crop_center_lr[2] - self.size_lr // 2:crop_center_lr[2] + self.size_lr // 2]

            H = img_dict['H'][:, crop_center_hr[0] - self.size_hr // 2:crop_center_hr[0] + self.size_hr // 2,
                         crop_center_hr[1] - self.size_hr // 2:crop_center_hr[1] + self.size_hr // 2,
                         crop_center_hr[2] - self.size_hr // 2:crop_center_hr[2] + self.size_hr // 2]

        return {'H': H.float(), 'L': L.float()}


class RandomCropForeground(Randomizable):
    """ Crops a patch with at least a certain percentage of foreground pixels (supports 2D & 3D). """

    def __init__(self, patch_size_lr, up_factor, foreground_threshold, foreground_ratio=0.10, pad_size=0, input_type="3D"):
        super().__init__()
        self.size_lr = patch_size_lr
        self.up_factor = up_factor
        self.size_hr = patch_size_lr * up_factor
        self.foreground_threshold = foreground_threshold
        self.foreground_ratio = foreground_ratio  # Required percentage of foreground pixels
        self.pad_size = pad_size
        self.spatial_dims = 3 if input_type == "3D" else 2  # 2D or 3D

        if pad_size > 0:
            self.size_hr -= 2 * up_factor * pad_size  # Adjust HR patch size

    def __call__(self, img_dict):
        return self.crop(img_dict)

    def crop(self, img_dict):
        img_L = img_dict['L']
        img_H = img_dict['H']

        valid_range_lr = torch.tensor(img_L.shape[1:]) - self.size_lr
        foreground_threshold = self.size_lr ** self.spatial_dims * self.foreground_ratio  # Required foreground count

        # Try 10 times to find a valid patch
        for _ in range(10):
            crop_start_lr = np.random.randint(0, valid_range_lr[:self.spatial_dims] + 1, (self.spatial_dims,)).tolist()

            if self.spatial_dims == 2:  # 2D (H, W)
                # TODO: update so that slices are limit to within cube and not whole 3D image, see RandomCropLabel
                slice_idx_lr = np.random.randint(0, img_L.shape[-1])  # Random slice index
                L = img_L[:, crop_start_lr[0]:crop_start_lr[0] + self.size_lr,
                             crop_start_lr[1]:crop_start_lr[1] + self.size_lr,
                             slice_idx_lr]
            else:  # 3D (D, H, W)
                L = img_L[:, crop_start_lr[0]:crop_start_lr[0] + self.size_lr,
                             crop_start_lr[1]:crop_start_lr[1] + self.size_lr,
                             crop_start_lr[2]:crop_start_lr[2] + self.size_lr]

            # Check if the patch has enough foreground pixels
            if (L > self.foreground_threshold).sum() > foreground_threshold:
                break  # Exit early if a valid patch is found

        crop_start_hr = [x * self.up_factor for x in crop_start_lr]

        if self.spatial_dims == 2:  # 2D (H, W)
            slice_idx_hr = slice_idx_lr * self.up_factor
            H = img_H[:, crop_start_hr[0]:crop_start_hr[0] + self.size_hr,
                         crop_start_hr[1]:crop_start_hr[1] + self.size_hr,
                         slice_idx_hr]
        else:  # 3D (D, H, W)
            H = img_H[:, crop_start_hr[0]:crop_start_hr[0] + self.size_hr,
                         crop_start_hr[1]:crop_start_hr[1] + self.size_hr,
                         crop_start_hr[2]:crop_start_hr[2] + self.size_hr]

        return {'H': H.float(), 'L': L.float()}


class RandomCropLabel(Randomizable):
    """ Randomly crops a region from both LR and HR images based on segmentation label image (supports 2D & 3D). """

    def __init__(self, patch_size_lr, up_factor, pad_size=0, input_type="3D", mask_mode="HR"):
        super().__init__()
        self.size_lr = patch_size_lr
        self.up_factor = up_factor
        self.size_hr = patch_size_lr * up_factor
        self.pad_size = pad_size
        self.mask_mode = mask_mode

        if pad_size > 0:
            self.size_hr -= 2 * up_factor * pad_size  # Adjust HR patch size

        self.spatial_dims = 3 if input_type == "3D" else 2  # 2D or 3D

    def __call__(self, img_dict):
        return self.crop(img_dict)

    def get_label_coords(self, seg_coords, valid_range):

        # label sampling
        n_idx = seg_coords.shape[-1]
        # Sample randomly within list of indexes where mask == 1 until a valid position is returned
        while True:
            rand_index = np.random.randint(low=0, high=n_idx + 1, size=(1,)).item()
            # rand_index = self.R.randint(low=0, high=n_idx + 1, size=(1,)).item()
            indexes = i, j, k = seg_coords[:, rand_index]
            if np.alltrue(indexes < valid_range):
                break

        return int(i), int(j), int(k)

    def crop(self, img_dict):

        img_L = img_dict['L']
        img_H = img_dict['H']

        # Compute valid cropping range
        valid_range_lr = torch.tensor(img_L.shape[1:]) - self.size_lr
        valid_range_hr = torch.tensor(img_H.shape[1:]) - self.size_hr

        # sample uniformly within mask image
        assert 'seg_coords' in img_dict, "seg_coords must be in img_dict for RandomCropLabel transform"
        if self.mask_mode == "HR":
            crop_start_hr = np.asarray(self.get_label_coords(img_dict['seg_coords'], valid_range_hr))

            # Correct indexes to be divisible by up_factor
            crop_start_lr = crop_start_hr // self.up_factor
            crop_start_hr = crop_start_lr * self.up_factor
            crop_start_lr += self.size_lr // 2
            crop_start_hr += self.size_hr // 2
        else:
            crop_start_lr = np.asarray(self.get_label_coords(img_dict['seg_coords'], valid_range_lr))

            # Correct indexes to be divisible by up_factor
            crop_start_hr = crop_start_lr * self.up_factor
            crop_start_lr += self.size_lr // 2
            crop_start_hr += self.size_hr // 2

        # Correct for padding of LR image, if any
        if self.pad_size > 0:
            pred_area_lr = self.size_hr//self.up_factor
            crop_start_lr = [x + self.pad_size + pred_area_lr//2 for x in crop_start_lr]

        # Extract patches
        if self.spatial_dims == 2:  # 2D (H, W)
            slice_idx_lr = np.random.randint(crop_start_lr[2] - self.size_lr//2,
                                             crop_start_lr[2] + self.size_lr//2)  # Random slice index within lr cube
            L = img_L[:, crop_start_lr[0] - self.size_lr//2:crop_start_lr[0] + self.size_lr//2,
                         crop_start_lr[1] - self.size_lr//2:crop_start_lr[1] + self.size_lr//2,
                         slice_idx_lr]

            slice_idx_hr = slice_idx_lr * self.up_factor
            H = img_H[:, crop_start_hr[0] - self.size_hr//2:crop_start_hr[0] + self.size_hr//2,
                         crop_start_hr[1] - self.size_hr//2:crop_start_hr[1] + self.size_hr//2,
                         slice_idx_hr]
        else:  # 3D (H, W, D)
            L = img_L[:, crop_start_lr[0] - self.size_lr//2:crop_start_lr[0] + self.size_lr//2,
                         crop_start_lr[1] - self.size_lr//2:crop_start_lr[1] + self.size_lr//2,
                         crop_start_lr[2] - self.size_lr//2:crop_start_lr[2] + self.size_lr//2]

            H = img_H[:, crop_start_hr[0] - self.size_hr//2:crop_start_hr[0] + self.size_hr//2,
                         crop_start_hr[1] - self.size_hr//2:crop_start_hr[1] + self.size_hr//2,
                         crop_start_hr[2] - self.size_hr//2:crop_start_hr[2] + self.size_hr//2]

        return {'H': H.float(), 'L': L.float()}



class GaussianblurImaged(MapTransform):
    def __init__(self, keys, blur_sigma):
        super().__init__(keys)

        self.blur_sigma = blur_sigma
        self.radius = math.ceil(3 * self.blur_sigma)

    def __call__(self, data):

        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = test_3d_gaussian_blur(d[key].squeeze(), ks=int(2 * self.radius + 1), blur_sigma=self.blur_sigma)
        return d



class BasicSRTransforms:

    def __init__(self, opt, mode="train"):

        self.opt = opt
        self.implicit = True if opt['model_opt']['model'] == "implicit" else False
        self.mode = mode
        self.size_hr = opt['dataset_opt']['patch_size_hr']
        self.size_lr = opt['dataset_opt']['patch_size']
        self.blur_sigma = opt['dataset_opt']['blur_sigma']
        self.downsampling_method = opt['dataset_opt']['downsampling_method']
        self.patches_per_batch = opt['dataset_opt']['train_dataset_params']['patches_per_batch']
        self.channel_dim = opt['dataset_opt']['channel_dim']
        self.up_factor = opt['up_factor']

        self.patch_crop_type = opt['dataset_opt']['patch_crop_type']
        self.mask_mode = "HR"
        if 'mask_mode' in opt['dataset_opt']:
            self.mask_mode = opt['dataset_opt']['mask_mode']
        self.sample_crop_pad_type = opt['dataset_opt']['sample_crop_pad_type']
        self.foreground_thresh = self.get_foreground_threshold(opt['dataset_opt']['name'])
        self.degradation_type = opt['dataset_opt']['degradation_type']
        self.input_type = opt['input_type']

        # Define foreground cropping / divisible padding
        if self.sample_crop_pad_type == "sample_crop_foreground":
            self.sample_crop_pad_transform = mt.CropForegroundd(keys=["H"], source_key="H", margin=0, select_fn=self.foreground_threshold_func, k_divisible=4)
        elif self.sample_crop_pad_type == "sample_divisible_padding":
            self.sample_crop_pad_transform = mt.DivisiblePadd(keys=["H"], k=4, mode="constant")  # Ensure HR and LR scans have even dimensions
        # self.border_crop = CropBorderd(self.lr_foreground_threshold)

        # Ensure the HR image is padded to a minimum of the patch size
        self.min_padding = mt.SpatialPadd(keys=["H"], spatial_size=[self.size_hr, self.size_hr, self.size_hr], mode="constant", value=0)

        # Normalization and scaling
        if opt['dataset_opt']['norm_type'] == "scale_intensity":
            self.norm_transform = mt.ScaleIntensityd(keys=["H"], minv=0.0, maxv=1.0)
        elif opt['dataset_opt']['norm_type'] == "znormalization":
            self.norm_transform = mt.NormalizeIntensityd(keys=["H"])

        # Padding transform
        self.pad_transform = mt.Identityd(keys=['H'])
        self.pad_size = 0
        if opt['model_opt']['model_architecture'] == "MTVNet":
            center_size = opt['model_opt']['netG']['context_sizes'][-1]  # fixed assumption of level_ratio = 2
            self.pad_size = (self.size_lr - center_size) // 2  # pad half of the context on all sides
            if self.pad_size > 0:
                self.pad_transform = mt.BorderPadd(keys=["L"], spatial_border=[self.pad_size, self.pad_size, self.pad_size], mode='constant')  # Pad here if net is MTVNet
            else:
                self.pad_transform = mt.Identityd(keys=["L"])

        # Orientation transform
        self.orientation_transform = mt.Identityd(keys=["H"])
        if opt['dataset_opt']['name'] == "IXI":
            self.orientation_transform = mt.Orientationd(keys=["H"], axcodes="RAS")

        # Degradation
        self.blur_transform = mt.Identityd(keys=["L"])
        if self.degradation_type == "resize":
            if opt['dataset_opt']['blur_method'] == '3d_gaussian_blur':
                self.blur_transform = GaussianblurImaged(keys=["L"], blur_sigma=self.blur_sigma)
            elif opt['dataset_opt']['blur_method'] == 'monai_gaussian_blur':
                self.blur_transform = mt.GaussianSmoothd(keys=["L"], sigma=opt['dataset_opt']['blur_sigma'])

            self.degradation = mt.Zoomd(keys=["L"],
                               zoom=1 / self.up_factor,
                               mode=self.downsampling_method,
                               align_corners=True,
                               keep_size=False)

        elif self.degradation_type == "kspace_trunc":
            self.trunc_factor = opt['dataset_opt']['trunc_factor']
            self.kspace_trunc_dim = opt['dataset_opt']['kspace_trunc_dim']
            self.degradation = KspaceTruncd(keys=["L"], trunc_factor=self.trunc_factor, norm_val=1.0, slice_dim=self.kspace_trunc_dim)

        # Random crop pair
        if self.implicit:
            self.random_crop_pair = RandomCropPairImplicitd(self.size_lr, self.up_factor, self.foreground_thresh, mode)
        else:
            if self.patch_crop_type == "random_spatial":
                self.random_crop_pair = RandomCropUniform(self.size_lr, self.up_factor, self.pad_size, self.input_type)
            elif self.patch_crop_type == "random_foreground":
                self.random_crop_pair = RandomCropForeground(self.size_lr, self.up_factor, self.foreground_thresh, self.pad_size, self.input_type)
            elif self.patch_crop_type == "random_label":
                self.random_crop_pair = RandomCropLabel(self.size_lr, self.up_factor, self.pad_size, self.input_type, self.mask_mode)

        # Random augmentations
        #self.random_aug_HR = mt.RandFlipd(keys=["H"], prob=0.5, spatial_axis=[0, 1, 2])  # Randomly flip HR image
        self.random_aug_LR = mt.RandFlipd(keys=["H", "L"], prob=0.5, spatial_axis=[0, 1, 2])  # Randomly flip HR and LR images

    def foreground_threshold_func(self, img):
        # threshold foreground
        return img > self.foreground_thresh  # This is actually the HR image


    def get_foreground_threshold(self, dataset_name):
        # Fields for crop_foreground
        if dataset_name == "KIRBY21":
            foreground_thresh = 0.02  # Kirby21 is T2w and therefore we need lower threshold
        elif dataset_name == "IXI":
            foreground_thresh = 0.05
        elif dataset_name == "BRATS2023":
            foreground_thresh = 0.05
        elif dataset_name == "HCP_1200":
            foreground_thresh = 0.05
        else:
            foreground_thresh = 0.05
        return foreground_thresh


    def get_transforms(self, baseline=False):

        if baseline:  # Remove cropping for baseline transforms
            self.random_crop_pair = mt.Identityd(keys=["H", "L"])

        transforms = mt.Compose(
            [
                # Deterministic Transforms
                mt.LoadImaged(keys=["H"], dtype=None),
                mt.EnsureChannelFirstd(keys=["H"], channel_dim=self.channel_dim),
                mt.SignalFillEmptyd(keys=["H"], replacement=0),  # Remove any NaNs
                self.norm_transform,
                self.sample_crop_pad_transform,
                self.min_padding,
                self.orientation_transform,
                #self.pad_transform,  # pad HR
                mt.CopyItemsd(keys=["H"], times=1, names=["L"]),
                self.blur_transform,  # Is Identity for other degradation types than resize
                self.degradation,
                self.pad_transform,  # pad LR
                # Random transforms
                #RandomCropPaird(self.size_lr, self.up_factor)
                self.random_crop_pair,  # Random crop pair
                # Augmentations
                self.random_aug_LR,  # Randomly flip HR image

            ]
        )

        return transforms


    def get_transforms_FACTS_Synth(self, baseline=False):  # unused

        if baseline:
            self.random_crop_pair = mt.Identityd(keys=["H", "L"])

        transforms = mt.Compose(
            [
                # Deterministic Transforms
                mt.LoadImaged(keys=["H", "L", "seg_coords"], dtype=None),
                mt.EnsureChannelFirstd(keys=["H"], channel_dim=self.channel_dim),
                mt.SignalFillEmptyd(keys=["H"], replacement=0),  # Remove any NaNs
                self.sample_crop_pad_transform,
                self.pad_transform,  # pad LR
                # Random transforms
                self.random_crop_pair  # Random crop pair

            ]
        )

        return transforms


    def get_transforms_FACTS_Real(self, baseline=False):  # unused

        if baseline:
            self.random_crop_pair = mt.Identityd(keys=["H", "L"])

        if self.up_factor != 1:  # FACTS_Real requires resizing of LR
            resize_transform = mt.Zoomd(keys=["L"],
                                        zoom=1 / self.up_factor,
                                        mode=self.downsampling_method,
                                        align_corners=True,
                                        keep_size=False),
        else:
            resize_transform = mt.Identityd(keys=["L"])

        transforms = mt.Compose(
            [
                # Deterministic Transforms
                mt.LoadImaged(keys=["H", "L", "seg_coords"], dtype=None),
                mt.EnsureChannelFirstd(keys=["H", "L"], channel_dim=self.channel_dim),
                mt.SignalFillEmptyd(keys=["H", "L"], replacement=0),  # Remove any NaNs
                self.sample_crop_pad_transform,
                resize_transform, # Resize LR
                self.pad_transform,  # pad LR
                # Random transforms
                self.random_crop_pair  # Random crop pair

            ]
        )

        return transforms

    def get_transforms_binning_brain(self, baseline=False):

        if baseline:
            self.random_crop_pair = mt.Identityd(keys=["H", "L"])

        transforms = mt.Compose(
            [
                # Deterministic Transforms
                mt.LoadImaged(keys=["H", "L"], dtype=None),
                mt.EnsureChannelFirstd(keys=["H", "L"], channel_dim=self.channel_dim),
                mt.SignalFillEmptyd(keys=["H", "L"], replacement=0),  # Remove any NaNs
                #self.sample_crop_pad_transform,
                self.pad_transform,  # pad LR
                # Random transforms
                self.random_crop_pair  # Random crop pair

            ]
        )

        return transforms


    def get_transforms_FEMur(self, baseline=False):

        if baseline:
            self.random_crop_pair = mt.Identityd(keys=["H", "L"])

        if self.opt['dataset_opt']['norm_type'] == "scale_intensity":
            self.norm_transform = mt.ScaleIntensityd(keys=["H", "L"], minv=0.0, maxv=1.0)
        elif self.opt['dataset_opt']['norm_type'] == "znormalization":
            self.norm_transform = mt.NormalizeIntensityd(keys=["H", "L"])

        transforms = mt.Compose(
            [
                # Deterministic Transforms
                mt.LoadImaged(keys=["H", "L", "seg_coords"], dtype=None),
                mt.EnsureChannelFirstd(keys=["H", "L"], channel_dim=self.channel_dim),
                mt.SignalFillEmptyd(keys=["H", "L"], replacement=0),  # Remove any NaNs
                self.norm_transform,
                self.sample_crop_pad_transform,            #Commented out since FEMur dataset might have odd sizes (HR/LR pairs not necessarily 4:1 in size)
                self.pad_transform,  # pad LR
                # Random transforms
                self.random_crop_pair  # Random crop pair

            ]
        )

        return transforms

    def get_transforms_VoDaSuRe(self, baseline=False):

        if baseline:
            self.random_crop_pair = mt.Identityd(keys=["H", "L"])

        if self.opt['dataset_opt']['norm_type'] == "scale_intensity":
            self.norm_transform = mt.ScaleIntensityd(keys=["H", "L"], minv=0.0, maxv=1.0)
        elif self.opt['dataset_opt']['norm_type'] == "znormalization":
            self.norm_transform = mt.NormalizeIntensityd(keys=["H", "L"])

        transforms = mt.Compose(
            [
                # Deterministic Transforms
                mt.LoadImaged(keys=["H", "L"], dtype=None),
                mt.EnsureChannelFirstd(keys=["H", "L"], channel_dim=self.channel_dim),
                mt.SignalFillEmptyd(keys=["H", "L"], replacement=0),  # Remove any NaNs
                self.norm_transform,
                #self.sample_crop_pad_transform,
                self.pad_transform,  # pad LR
                # Random transforms
                self.random_crop_pair  # Random crop pair

            ]
        )

        return transforms


class ImplicitModelTransformd():
    def __init__(self, up_factor, mode,  **kwargs):

        self.up_factor = up_factor
        self.mode = mode
        if mode == "train":
            self.sample_size = 8000
        else:
            self.sample_size = -1


    def __call__(self, img_dict):

        patch_hr = img_dict['H']

        if self.mode == "train":
            # compute the size of HR patch according to the scale
            hr_h, hr_w, hr_d = (torch.tensor([10, 10, 10], dtype=torch.int) * self.up_factor)
            # generate HR patch by cropping
            patch_hr = patch_hr[0, :hr_h, :hr_w, :hr_d]
            # simulated LR patch by down-sampling HR patch
            patch_lr = img_dict['L'][:, :hr_h//self.up_factor, :hr_w//self.up_factor, :hr_d//self.up_factor]
        else:
            # Take whole HR/LR patch
            patch_hr = patch_hr[0, :, :, :]
            patch_lr = img_dict['L'][:, :, :, :]

        # generate coordinate set
        xyz_hr = make_coord(patch_hr.shape, flatten=True)
        # randomly sample voxel coordinates
        if self.mode == "train":
            sample_indices = torch.randperm(len(xyz_hr))[:self.sample_size]  # sample without replacement
            xyz_hr = xyz_hr[sample_indices]
            patch_hr = patch_hr.reshape(-1, 1)[sample_indices]
        else:
            patch_hr = patch_hr.reshape(-1, 1)

        return {'L': patch_lr, 'H_xyz': xyz_hr, 'H': patch_hr}
