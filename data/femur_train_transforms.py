import numpy as np
import torch
import monai.transforms as mt
from monai.transforms import Randomizable
from data.train_transforms import GaussianblurImaged, ImplicitModelTransformd


class CropCachedData():

    def __init__(self, patch_size_hr, crop_type, load_LR=True):
        self.size_hr = patch_size_hr
        self.crop_type = crop_type
        self.load_LR = load_LR

    def get_label_coords(self, seg_coords):

        # label sampling
        n_idx = seg_coords.shape[-1]
        # Sample randomly within list of indexes where mask == 1 until a valid position is returned
        while True:
            rand_index = torch.randint(low=0, high=n_idx + 1, size=(1,)).item()
            indexes = i, j, k = seg_coords[:, rand_index]
            if np.alltrue(indexes <= self.valid_range):
                break

        return i, j, k

    def __call__(self, img_dict: dict):

        self.valid_range = np.subtract(img_dict['H'].shape[1:], self.size_hr)
        if self.crop_type == "random_spatial":
            # sample uniformly
            i, j, k = tuple(int(torch.randint(low=0, high=x + 1, size=(1,)).item()) for x in self.valid_range)
        elif self.crop_type == "random_label":
            # sample uniformly within mask image
            i, j, k = self.get_label_coords(img_dict['seg_coords'])

        H_patch = img_dict['H'][:, i:i + self.size_hr, j:j + self.size_hr, k:k + self.size_hr]
        mask_patch = img_dict['mask'][:, i:i + self.size_hr, j:j + self.size_hr, k:k + self.size_hr]

        if self.load_LR:
            L_patch = img_dict['L'][:, i:i + self.size_hr, j:j + self.size_hr, k:k + self.size_hr]
            out_dict = {'H': H_patch, 'L': L_patch, 'mask': mask_patch}
        else:
            out_dict = {'H': H_patch, 'mask': mask_patch}

        return out_dict

class CropCachedDataPaird(Randomizable):

    def __init__(self, patch_size_lr, up_factor, crop_type, load_LR=True, pad_size=0, input_type="3D"):
        self.size_lr = patch_size_lr
        self.crop_type = crop_type
        self.load_LR = load_LR
        self.up_factor = up_factor
        self.size_hr = patch_size_lr * self.up_factor
        self.pad_size = pad_size
        if pad_size > 0:
            self.center_size = self.size_lr - 2*self.pad_size
            self.size_lr = self.size_lr
            self.size_hr = self.size_hr - 2*self.up_factor*self.pad_size
        self.spatial_dims = 3 if input_type == "3D" else 2  # 2D or 3D

    def get_label_coords(self, seg_coords):

        # label sampling
        n_idx = seg_coords.shape[-1]
        # Sample randomly within list of indexes where mask == 1 until a valid position is returned
        while True:
            rand_index = np.random.randint(low=0, high=n_idx + 1, size=(1,)).item()
            #rand_index = self.R.randint(low=0, high=n_idx + 1, size=(1,)).item()
            indexes = i, j, k = seg_coords[:, rand_index]
            if np.alltrue(indexes < self.valid_range_HR):
                break

        return int(i), int(j), int(k)

    def __call__(self, img_dict: dict):

        self.valid_range_LR = np.subtract(img_dict['L'].shape[1:], self.size_lr)
        self.valid_range_HR = np.subtract(img_dict['H'].shape[1:], self.size_hr)
        #print("img_L shape", img_dict['L'].shape[1:], "valid_range_LR", self.valid_range_LR)
        #print("img_H shape", img_dict['H'].shape[1:], "valid_range_HR", self.valid_range_HR)

        if self.crop_type == "random_spatial":
            # sample uniformly
            i_lr, j_lr, k_lr = tuple(int(self.R.randint(low=0, high=x + 1, size=(1,)).item()) for x in self.valid_range_LR)

            # Get corresponding position in HR
            i_hr, j_hr, k_hr = i_lr * self.up_factor, j_lr * self.up_factor, k_lr * self.up_factor

            H_patch = img_dict['H'][:, i_hr:i_hr + self.size_hr, j_hr:j_hr + self.size_hr, k_hr:k_hr + self.size_hr]

        elif self.crop_type == "random_label":
            # sample uniformly within mask image
            i_hr, j_hr, k_hr = self.get_label_coords(img_dict['seg_coords'])

            i_lr, j_lr, k_lr = i_hr // self.up_factor, j_hr // self.up_factor, k_hr // self.up_factor
            i_hr, j_hr, k_hr = i_lr * self.up_factor, j_lr * self.up_factor, k_lr * self.up_factor

            # Correct for padding of LR image, if any
            if self.pad_size > 0:
                i_lr = i_lr + self.pad_size + self.center_size//2
                j_lr = j_lr + self.pad_size + self.center_size//2
                k_lr = k_lr + self.pad_size + self.center_size//2

            H_patch = img_dict['H'][:, i_hr:i_hr + self.size_hr, j_hr:j_hr + self.size_hr, k_hr:k_hr + self.size_hr]

        if self.pad_size > 0:  # Center-based cropping instead
            L_patch = img_dict['L'][:, i_lr - self.size_lr//2:i_lr + self.size_lr//2,
                                       j_lr - self.size_lr//2:j_lr + self.size_lr//2,
                                       k_lr - self.size_lr//2:k_lr + self.size_lr//2]
        else:
            L_patch = img_dict['L'][:, i_lr:i_lr + self.size_lr, j_lr:j_lr + self.size_lr, k_lr:k_lr + self.size_lr]

        if self.spatial_dims == 2:
            # TODO fix assumption of no padding for 2D models.
            slice_idx_lr = np.random.randint(0, L_patch.shape[-1])  # Random slice index
            slice_idx_hr = slice_idx_lr * self.up_factor
            H_patch = H_patch[..., slice_idx_hr]
            L_patch = L_patch[..., slice_idx_lr]

        out_dict = {'H': H_patch.float(), 'L': L_patch.float()}
        return out_dict

class CropCachedDataPairImplicitd(Randomizable):

    def __init__(self, patch_size_lr, up_factor, crop_type, load_LR=True, mode="train", input_type="3D"):
        self.size_lr = patch_size_lr
        self.crop_type = crop_type
        self.load_LR = load_LR
        self.up_factor = up_factor
        self.size_hr = patch_size_lr * self.up_factor

        self.spatial_dims = 3 if input_type == "3D" else 2  # 2D or 3D

        # Fields for implicit transform
        self.mode = mode
        self.sample_size = 8000 if mode == "train" else -1

        if self.mode == "train":
            # Precompute HR shape and HR coordinate grid for training
            self.hr_shape = (torch.tensor([10, 10, 10], dtype=torch.int) * up_factor).tolist()
            self.lr_shape = (torch.tensor([10, 10, 10], dtype=torch.int)).tolist()
            self.coord_grid = self._precompute_coord(self.hr_shape)  # Precompute coord grid only once

    @staticmethod
    def make_coord(shape):
        """
        Generate the coordinate grid for a given shape.
        """
        ranges = [-1, 1]
        coord_seqs = [torch.linspace(ranges[0] + (1 / (2 * n)), ranges[1] - (1 / (2 * n)), n, device='cuda') for n in shape]
        ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
        return ret.view(-1, ret.shape[-1])

    def _precompute_coord(self, hr_shape):
        """
        Precompute and cache coordinate grid for training mode.
        """
        return self.make_coord(hr_shape)

    def get_label_coords(self, seg_coords):

        # label sampling
        n_idx = seg_coords.shape[-1]
        # Sample randomly within list of indexes where mask == 1 until a valid position is returned
        while True:
            rand_index = np.random.randint(low=0, high=n_idx + 1, size=(1,)).item()
            indexes = i, j, k = seg_coords[:, rand_index]
            if np.alltrue(indexes < self.valid_range_HR):
                break

        return int(i), int(j), int(k)

    def __call__(self, img_dict: dict):

        self.valid_range_LR = np.subtract(img_dict['L'].shape[1:], self.size_lr)
        self.valid_range_HR = np.subtract(img_dict['H'].shape[1:], self.size_hr)

        if self.crop_type == "random_spatial":
            # sample uniformly
            i_lr, j_lr, k_lr = tuple(int(self.R.randint(low=0, high=x + 1, size=(1,)).item()) for x in self.valid_range_LR)

            # Get corresponding position in HR
            i_hr, j_hr, k_hr = i_lr * self.up_factor, j_lr * self.up_factor, k_lr * self.up_factor

        else:
            # sample uniformly within mask image
            i_hr, j_hr, k_hr = self.get_label_coords(img_dict['seg_coords'])

            i_lr, j_lr, k_lr = i_hr // self.up_factor, j_hr // self.up_factor, k_hr // self.up_factor
            i_hr, j_hr, k_hr = i_lr * self.up_factor, j_lr * self.up_factor, k_lr * self.up_factor

        if self.mode == "train":
            hr_h, hr_w, hr_d = self.hr_shape
            lr_h, lr_w, lr_d = self.lr_shape

            # Crop the high-resolution patch
            patch_hr = img_dict['H'][:, i_hr:i_hr + hr_h, j_hr:j_hr + hr_w, k_hr:k_hr + hr_d]

            # Simulate low-resolution patch
            patch_lr = img_dict['L'][:, i_lr:i_lr + lr_h, j_lr:j_lr + lr_w, k_lr:k_lr + lr_d]

            # Use the precomputed coordinate grid
            xyz_hr = self.coord_grid

            # Sample random indices once
            sample_indices = torch.randperm(xyz_hr.shape[0], device=patch_hr.device)[:self.sample_size]
            xyz_hr = xyz_hr[sample_indices]
            patch_hr = patch_hr.reshape(-1, 1)[sample_indices]
        else:
            # For testing, just reshape the entire patch
            patch_hr = img_dict['H'][:, i_hr:i_hr + self.size_hr, j_hr:j_hr + self.size_hr, k_hr:k_hr + self.size_hr]
            patch_hr = patch_hr[0].reshape(-1, 1)
            patch_lr = img_dict['L'][:, i_lr:i_lr + self.size_lr, j_lr:j_lr + self.size_lr, k_lr:k_lr + self.size_lr]
            xyz_hr = self.make_coord(patch_hr.shape)

        return {'L': patch_lr.float(), 'H_xyz': xyz_hr, 'H': patch_hr.float()}

class MemmapCrop(Randomizable):

    def __init__(self, patch_size_hr, crop_type, load_LR=True):
        self.size_hr = patch_size_hr
        self.crop_type = crop_type
        self.load_LR = load_LR

    def get_label_coords(self, path_dict):

        # label sampling
        coor_map = np.load(path_dict['seg_coords'], mmap_mode='r')
        n_idx = coor_map.shape[-1]
        # Sample randomly within list of indexes where mask == 1 until a valid position is return
        while True:
            rand_index = self.R.randint(low=0, high=n_idx + 1, size=(1,)).item()
            indexes = i, j, k = coor_map[:, rand_index]
            if np.alltrue(indexes <= self.valid_range):
                break

        return i, j, k


    def __call__(self, path_dict: dict):
        H_map = np.load(path_dict['H'], mmap_mode='c')
        if self.load_LR:
            L_map = np.load(path_dict['L'], mmap_mode='c')
        mask_map = np.load(path_dict['mask'], mmap_mode='c')

        self.valid_range = np.subtract(H_map.shape, self.size_hr)
        if self.crop_type == "random_spatial":
            # sample uniformly
            i, j, k = tuple(int(self.R.randint(low=0, high=x + 1, size=(1,)).item()) for x in self.valid_range)
        elif self.crop_type == "random_label":
            # sample uniformly within mask image
            i, j, k = self.get_label_coords(path_dict)

        H_patch = H_map[i:i + self.size_hr, j:j + self.size_hr, k:k + self.size_hr]
        mask_patch = mask_map[i:i + self.size_hr, j:j + self.size_hr, k:k + self.size_hr]

        if self.load_LR:
            L_patch = L_map[i:i + self.size_hr, j:j + self.size_hr, k:k + self.size_hr]
            out_dict = {'H': H_patch, 'L': L_patch, 'mask': mask_patch}
        else:
            out_dict = {'H': H_patch, 'mask': mask_patch}

        return out_dict


class Femur_transforms():

    def __init__(self, opt, mode="train", synthetic=False):

        self.synthetic = synthetic  # Whether to create LR data from synthetic degradation of HR data

        self.mode = mode  # train or test
        self.implicit = True if opt['model'] == "implicit" else False

        self.size_hr = opt['dataset_opt']['patch_size_hr']
        self.size_lr = opt['dataset_opt']['patch_size']
        self.blur_sigma = opt['dataset_opt']['blur_sigma']
        self.downsampling_method = opt['dataset_opt']['downsampling_method']
        self.patches_per_batch = opt['dataset_opt']['train']['dataset_params']['patches_per_batch']
        self.channel_dim = opt['dataset_opt']['channel_dim']

        self.border_hr = opt['up_factor'] * 2  # 2 for 1x, 4 for 2x and 8 for 4x
        self.border_lr = self.border_hr // opt['up_factor']

        self.load_and_crop_transform = MemmapCrop(patch_size_hr=self.size_hr + self.border_hr,
                                                  crop_type=opt['dataset_opt']['patch_crop_type'],
                                                  load_LR=not synthetic)

        if opt['dataset_opt']['blur_method'] == '3d_gaussian_blur':
            self.blur_transform = GaussianblurImaged(blur_sigma=self.blur_sigma, in_key="L", out_key="L")
        elif opt['dataset_opt']['blur_method'] == 'monai_gaussian_blur':
            self.blur_transform = mt.GaussianSmoothd(keys=["L"], sigma=opt['dataset_opt']['blur_sigma'])

        self.implicit_model_transform = ImplicitModelTransformd(opt['up_factor'], mode=mode)

    def get_transforms(self):

        if self.synthetic:
            transforms = mt.Compose(
                [
                    self.load_and_crop_transform,
                    mt.EnsureChannelFirstd(keys=["H", "mask"], channel_dim=self.channel_dim),
                    mt.SignalFillEmptyd(keys=["H", "mask"], replacement=0),  # Remove any NaNs
                    #mt.DivisiblePadd(keys=["H"], k=2, mode="constant"),  # Ensure HR and LR scans have even dimensions
                    #self.norm_transform,
                    mt.CenterSpatialCropd(keys=["H", "mask"], roi_size=[self.size_hr, self.size_hr, self.size_hr]),
                    mt.CopyItemsd(keys=["H"], times=1, names=["L"]),
                    mt.CenterSpatialCropd(keys=["H"], roi_size=[self.size_hr, self.size_hr, self.size_hr]),
                    self.blur_transform,
                    mt.Resized(keys="L",
                               spatial_size=[self.size_lr + self.border_lr, self.size_lr + self.border_lr,
                                             self.size_lr + self.border_lr],
                               mode=self.downsampling_method,
                               align_corners=True,
                               anti_aliasing=False,
                               anti_aliasing_sigma=self.blur_sigma
                               ),

                    mt.CenterSpatialCropd(keys=["L"], roi_size=[self.size_lr, self.size_lr, self.size_lr]),
                ]
            )

        else:
            transforms = mt.Compose(
                [
                    self.load_and_crop_transform,
                    mt.EnsureChannelFirstd(keys=["H", "L", "mask"], channel_dim=self.channel_dim),
                    mt.SignalFillEmptyd(keys=["H", "L"], replacement=0),  # Remove any NaNs
                    #mt.DivisiblePadd(keys=["H", "L"], k=2, mode="constant"),  # Ensure HR and LR scans have even dimensions
                    mt.CenterSpatialCropd(keys=["H", "mask"], roi_size=[self.size_hr, self.size_hr, self.size_hr]),
                    #self.blur_transform,  # disable blur, but maybe we need this to not destroy features in the upscaled LR image when we downsample
                    mt.Resized(keys="L",
                               spatial_size=[self.size_lr + self.border_lr, self.size_lr + self.border_lr, self.size_lr + self.border_lr],
                               mode=self.downsampling_method,
                               align_corners=True,
                               anti_aliasing=False,
                               anti_aliasing_sigma=self.blur_sigma
                               ),

                    mt.CenterSpatialCropd(keys=["L"], roi_size=[self.size_lr, self.size_lr, self.size_lr])

                ]
            )

        if self.implicit:
            transforms = mt.Compose([transforms, self.implicit_model_transform])

        return transforms


class Femur_transforms_cachableV2():

    def __init__(self, opt, mode="train", synthetic=False):

        self.synthetic = synthetic  # Whether to create LR data from synthetic degradation of HR data

        self.mode = mode  # train or test
        self.implicit = True if opt['model'] == "implicit" else False

        self.size_hr = opt['dataset_opt']['patch_size_hr']
        self.size_lr = opt['dataset_opt']['patch_size']
        self.blur_sigma = opt['dataset_opt']['blur_sigma']
        self.downsampling_method = opt['dataset_opt']['downsampling_method']
        self.patches_per_batch = opt['dataset_opt']['train']['dataset_params']['patches_per_batch']
        self.channel_dim = opt['dataset_opt']['channel_dim']

        self.crop_type = opt['dataset_opt']['patch_crop_type']
        self.up_factor = opt['up_factor']

        self.pad_transform = mt.Identityd(keys=['H'])
        self.pad_size = 0

        self.input_type = opt['input_type']

        self.enable_femur_padding = opt['dataset_opt']['enable_femur_padding']
        print("Enable femur padding:", self.enable_femur_padding)
        self.pad_transform = mt.Identityd(keys=['L'])
        self.divisible_pad = 4 if opt['up_factor'] % 2 == 0 else 3  # Added adaptive divisible padding

        if opt['model_architecture'] == "MTVNet" and self.enable_femur_padding:
            center_size = opt['netG']['context_sizes'][-1]  # fixed assumption of level_ratio = 2

            self.pad_size = (self.size_lr - center_size) // 2  # pad half of the context on all sides
            #pad_size = pad_size * opt['up_factor']  # Since we pad the HR image, multiply padding by the upscaling factor
            if self.pad_size > 0:
                self.pad_transform = mt.BorderPadd(keys=['L'], spatial_border=[self.pad_size, self.pad_size, self.pad_size], mode='constant')  # Pad here if net is MTVNet

        if opt['dataset_opt']['blur_method'] == '3d_gaussian_blur':
            self.blur_transform = GaussianblurImaged(keys=["L"], blur_sigma=self.blur_sigma)
        elif opt['dataset_opt']['blur_method'] == 'monai_gaussian_blur':
            self.blur_transform = mt.GaussianSmoothd(keys=["L"], sigma=opt['dataset_opt']['blur_sigma'])

        if opt['dataset_opt']['norm_type'] == "scale_intensity":
            self.norm_transform = mt.ScaleIntensityd(keys=["H", "L"], minv=0.0, maxv=1.0)
        elif opt['dataset_opt']['norm_type'] == "znormalization":
            self.norm_transform = mt.NormalizeIntensityd(keys=["H", "L"])

        self.border_hr = opt['up_factor'] * 2  # 2 for 1x, 4 for 2x and 8 for 4x
        self.border_lr = self.border_hr // opt['up_factor']

        #self.crop_cached_transform = CropCachedData(patch_size_hr=self.size_hr + self.border_hr,
        #                                            crop_type=opt['dataset_opt']['crop_type'],
        #                                            load_LR=not synthetic)

        if self.implicit:
            self.crop_cached_data_pair = CropCachedDataPairImplicitd(self.size_lr,
                                                                     self.up_factor,
                                                                     crop_type=self.crop_type,
                                                                     load_LR=not self.synthetic,
                                                                     mode=self.mode,
                                                                     input_type=self.input_type)

        else:
            self.crop_cached_data_pair = CropCachedDataPaird(self.size_lr,
                                                             self.up_factor,
                                                             crop_type=self.crop_type,
                                                             load_LR=not self.synthetic,
                                                             pad_size=self.pad_size,
                                                             input_type=self.input_type)

    def get_transforms(self):

        if self.synthetic:
            transforms = mt.Compose(
                [
                    # Deterministic Transforms
                    mt.LoadImaged(keys=["H", "L", "seg_coords"], dtype=torch.float16),
                    mt.EnsureChannelFirstd(keys=["H"], channel_dim=self.channel_dim),
                    mt.SignalFillEmptyd(keys=["H"], replacement=0),  # Remove any NaNs
                    mt.DivisiblePadd(keys=["H"], k=self.divisible_pad, mode="constant"),  # Ensure HR and LR scans have even dimensions
                    self.pad_transform,

                    # Random transforms
                    self.crop_cached_data_pair,

                ]
            )

        else:
            transforms = mt.Compose(
                [
                    # Deterministic Transforms
                    mt.LoadImaged(keys=["H", "L", "seg_coords"], dtype=torch.float16),
                    mt.EnsureChannelFirstd(keys=["H", "L"], channel_dim=self.channel_dim),
                    mt.SignalFillEmptyd(keys=["H", "L"], replacement=0),  # Remove any NaNs
                    mt.DivisiblePadd(keys=["H", "L"], k=self.divisible_pad, mode="constant"),  # Ensure HR and LR scans have even dimensions
                    mt.Zoomd(keys=["L"],
                             zoom=1 / self.up_factor,
                             mode=self.downsampling_method,
                             align_corners=True,
                             keep_size=False),
                    self.pad_transform,

                    # Random transforms
                    self.crop_cached_data_pair,

                ]
            )

        return transforms
