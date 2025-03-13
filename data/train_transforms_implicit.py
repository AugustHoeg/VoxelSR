import math

import numpy as np
import torch
from monai.transforms import Randomizable, Transform, MapTransform


class RandomCropPairImplicitd(Randomizable):

    def __init__(self, patch_size_lr, up_factor, lr_foreground_threshold, mode):
        super().__init__()
        self.size_lr = patch_size_lr
        self.up_factor = up_factor
        self.size_hr = patch_size_lr * self.up_factor
        self.mode = mode
        self.sample_size = 8000 if mode == "train" else -1
        self.lr_foreground_threshold = lr_foreground_threshold

        if self.mode == "train":
            # Precompute HR shape and HR coordinate grid for training
            self.hr_shape = (torch.tensor([24, 24, 24], dtype=torch.int) * up_factor).tolist()
            self.lr_shape = (torch.tensor([24, 24, 24], dtype=torch.int)).tolist()
            self.coord_grid = self._precompute_coord(self.hr_shape)  # Precompute coord grid only once

    @staticmethod
    def make_coord(shape):
        """
        Generate the coordinate grid for a given shape.
        """
        ranges = [-1, 1]
        coord_seqs = [torch.linspace(ranges[0] + (1 / (2 * n)), ranges[1] - (1 / (2 * n)), n, device='cuda') for n in
                      shape]
        ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
        return ret.view(-1, ret.shape[-1])

    def _precompute_coord(self, hr_shape):
        """
        Precompute and cache coordinate grid for training mode.
        """
        return self.make_coord(hr_shape)

    def randomize(self):
        self.sub_seed = self.R.randint(999999, dtype="uint32")

    def __call__(self, img_dict: dict):

        # Get valid index range of LR image
        self.valid_range_lr = torch.tensor(img_dict['L'].shape[1:]) - self.size_lr

        # Sample uniform random indexes in valid range in LR
        i_lr, j_lr, k_lr = tuple(int(self.R.randint(low=0, high=x + 1, size=(1,)).item()) for x in self.valid_range_lr)

        # Get corresponding position in HR
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

            # Filter background points
            pos_indices = patch_hr > self.lr_foreground_threshold
            sorted_patch_hr = patch_hr[pos_indices]
            sorted_xyz_hr = xyz_hr[pos_indices.reshape(-1), :]

            # Check if we have fewer than 8000 points and add more if necessary
            if sorted_xyz_hr.shape[0] < self.sample_size:
                extra_indices = torch.randperm(xyz_hr.shape[0], device=patch_hr.device)[
                                :self.sample_size - sorted_xyz_hr.shape[0]]
                sorted_xyz_hr = torch.cat([sorted_xyz_hr, xyz_hr[extra_indices]], dim=0)
                sorted_patch_hr = torch.cat([sorted_patch_hr, patch_hr.reshape(-1)[extra_indices]], dim=0)

            # Sample random indices once
            sample_indices = torch.randperm(sorted_xyz_hr.shape[0], device=patch_hr.device)[:self.sample_size]
            xyz_hr = sorted_xyz_hr[sample_indices]
            patch_hr = sorted_patch_hr.reshape(-1, 1)[sample_indices]
        else:
            # For testing, just reshape the entire patch
            patch_hr = img_dict['H'][:, i_hr:i_hr + self.size_hr, j_hr:j_hr + self.size_hr, k_hr:k_hr + self.size_hr]
            patch_hr = patch_hr[0].reshape(-1, 1)
            patch_lr = img_dict['L'][:, i_lr:i_lr + self.size_lr, j_lr:j_lr + self.size_lr, k_lr:k_lr + self.size_lr]
            xyz_hr = self.make_coord(patch_hr.shape)

        return {'L': patch_lr, 'H_xyz': xyz_hr, 'H': patch_hr}


class ImplicitModelTransformFastd:
    def __init__(self, up_factor, mode, **kwargs):
        self.up_factor = up_factor
        self.mode = mode
        self.sample_size = 8000 if mode == "train" else -1

        if self.mode == "train":
            # Precompute HR shape and HR coordinate grid for training
            self.hr_shape = (torch.tensor([10, 10, 10], dtype=torch.int) * up_factor).tolist()
            self.coord_grid = self._precompute_coord(self.hr_shape)  # Precompute coord grid only once

    def __call__(self, img_dict):
        patch_hr = img_dict['H']

        if self.mode == "train":
            hr_h, hr_w, hr_d = self.hr_shape

            # Crop the high-resolution patch
            patch_hr = patch_hr[0, :hr_h, :hr_w, :hr_d]

            # Simulate low-resolution patch
            patch_lr = img_dict['L'][:, :hr_h // self.up_factor, :hr_w // self.up_factor, :hr_d // self.up_factor]

            # Use the precomputed coordinate grid
            xyz_hr = self.coord_grid

            # Sample random indices once
            sample_indices = torch.randperm(xyz_hr.shape[0], device=patch_hr.device)[:self.sample_size]
            xyz_hr = xyz_hr[sample_indices]
            patch_hr = patch_hr.reshape(-1, 1)[sample_indices]
        else:
            # For testing, just reshape the entire patch
            patch_hr = patch_hr[0]
            patch_lr = img_dict['L']
            xyz_hr = self.make_coord(patch_hr.shape)
            patch_hr = patch_hr.reshape(-1, 1)

        return {'L': patch_lr, 'H_xyz': xyz_hr, 'H': patch_hr}

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
