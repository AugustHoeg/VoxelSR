import os
import glob

import monai.transforms as mt
import torch
import numpy as np
from data.femur_train_transforms import Femur_transforms, Femur_transforms_cachableV2
from data.femur_baseline_transforms import Femur_baseline_transformsV2
from data.kspace import KspaceTrunc
from os.path import dirname as pd

def save_low_resolution_femur_images(images, opt):
    divisible_pad = 4 if opt['up_factor'] % 2 == 0 else 3   # Added adaptive divisible padding
    # Create LR images for synthetic femur data
    LR_transforms = mt.Compose(
        [
            mt.LoadImage(reader="NumpyReader", dtype=torch.float32, image_only=True),
            mt.EnsureChannelFirst(channel_dim=opt['dataset_opt']['channel_dim']),
            mt.SignalFillEmpty(replacement=0),  # Remove any NaNs
            mt.DivisiblePad(k=divisible_pad, mode="constant"),  # Ensure HR and LR scans have even dimensions
            mt.GaussianSmooth(sigma=opt['dataset_opt']['blur_sigma']),
            mt.Zoom(zoom=1 / opt['up_factor'],
                    mode=opt['dataset_opt']['downsampling_method'],
                    align_corners=True,
                    keep_size=False,
                    dtype=torch.float32),
        ]
    )

    for i in range(len(images)):
        img_path = images[i]
        filename = os.path.basename(img_path)
        scan_dir = pd(pd(pd(img_path)))
        synthetic_lr_dir = "synthetic_lr_" + str(opt['up_factor']) + "x" + "/volume/"
        absolute_synthetic_lr_dir = os.path.join(scan_dir, synthetic_lr_dir)
        if not os.path.exists(absolute_synthetic_lr_dir):
            os.makedirs(absolute_synthetic_lr_dir)
        else:
            continue

        img = LR_transforms(img_path)
        print("Creating synthetic LR image for %s" % os.path.basename(img_path))
        np.save(os.path.join(absolute_synthetic_lr_dir, filename), img)

def save_kspace_femur_images(images, opt):
    divisible_pad = 4 if opt['up_factor'] % 2 == 0 else 3   # Added adaptive divisible padding
    kspace_trunc_dim = 3
    trunc_factor = opt['dataset_opt']['trunc_factor']
    # Create LR images for synthetic femur data
    LR_transforms = mt.Compose(
        [
            mt.LoadImage(reader="NumpyReader", dtype=torch.float32, image_only=True),
            mt.EnsureChannelFirst(channel_dim=opt['dataset_opt']['channel_dim']),
            mt.SignalFillEmpty(replacement=0),  # Remove any NaNs
            mt.DivisiblePad(k=divisible_pad, mode="constant"),  # Ensure HR and LR scans have even dimensions
            KspaceTrunc(trunc_factor=trunc_factor, norm_val=1.0, slice_dim=kspace_trunc_dim)
        ]
    )

    for i in range(len(images)):
        img_path = images[i]
        filename = os.path.basename(img_path)
        scan_dir = pd(pd(pd(img_path)))
        synthetic_lr_dir = "kspace_" + str(opt['dataset_opt']['trunc_factor']) + "/volume/"
        absolute_synthetic_lr_dir = os.path.join(scan_dir, synthetic_lr_dir)
        if not os.path.exists(absolute_synthetic_lr_dir):
            os.makedirs(absolute_synthetic_lr_dir)
        else:
            continue

        img = LR_transforms(img_path)
        print("Creating synthetic LR image for %s" % os.path.basename(img_path))
        np.save(os.path.join(absolute_synthetic_lr_dir, filename), img)


def save_femur_seg_coords(masks, opt, base_path):
    divisible_pad = 4 if opt['up_factor'] % 2 == 0 else 3   # Added adaptive divisible padding
    mask_transforms = mt.Compose(
        [
            mt.LoadImage(reader="NumpyReader", dtype=torch.int16, image_only=True),
            mt.EnsureChannelFirst(channel_dim=opt['dataset_opt']['channel_dim']),
            mt.SignalFillEmpty(replacement=0),  # Remove any NaNs
            mt.DivisiblePad(k=divisible_pad, mode="constant"),  # Ensure HR and LR scans have even dimensions
        ]
    )

    # Create lists of coordinates where segmentation mask is 1
    for i in range(len(masks)):
        mask = masks[i]
        # Check if seg_coords exists and make it
        # seg_coords_dir = os.path.join(os.path.dirname(os.path.dirname(mask)), "seg_coords")
        femur_name = os.path.basename(pd(pd(pd(mask))))
        seg_coords_dir = os.path.join(base_path, "2022_QIM_52_Bone", femur_name, "mask", "seg_coords")
        if not os.path.exists(seg_coords_dir):
            os.makedirs(seg_coords_dir)
        else:
            continue

        #mask_map = np.load(mask)
        mask_map = mask_transforms(mask)
        # Find x,y,z indexes where segmentation mask is 1
        print("Creating segmentation coordinate map for mask %s" % os.path.basename(mask))
        # z_idx, y_idx, x_idx = np.where(mask_map == 1)
        zyx_seg_idx = np.int16(np.reshape(np.where(mask_map[0] == 1), (3, -1)))
        filename_coor = os.path.basename(mask)

        # Save index lists as .npy file
        np.save(os.path.join(seg_coords_dir, filename_coor), zyx_seg_idx)


def split_paths(paths, test_paths):
    train = [string for string in paths if not any(test in string for test in test_paths)]
    test = [string for string in paths if any(test in string for test in test_paths)]
    return train, test

class Dataset_2022_QIM_52_Bone():

    def __init__(self, opt, apply_split=True):
        self.apply_split = apply_split
        self.opt = opt

        self.synthetic = "Synthetic" in opt['dataset_opt']['name']

        if opt['run_type'] == "HOME PC":
            self.data_path = "../Vedrana_master_project/3D_datasets/datasets/2022_QIM_52_Bone/femur*"
            base_path = "../Vedrana_master_project/3D_datasets/datasets/"
            test_paths = ["f_002.npy"]
        elif opt['cluster'] == "TITANS":
            self.data_path = "/scratch/aulho/Python/3D_datasets/datasets/2022_QIM_52_Bone/femur_*"
            base_path = "/scratch/aulho/Python/3D_datasets/datasets/"
            test_paths = ["f_002.npy", "f_086.npy", "f_138.npy"]
        else:  # Default is opt['cluster'] = DTU_HPC
            self.data_path = "../3D_datasets/datasets/2022_QIM_52_Bone/femur_*" # Use this path to read from personal scratch space
            base_path = "../3D_datasets/datasets/"
            test_paths = ["f_002.npy", "f_086.npy", "f_138.npy"]

        images_HR = sorted(glob.glob(os.path.join(self.data_path, "micro/volume/", "f_*.npy")))
        masks = sorted(glob.glob(os.path.join(self.data_path, "mask/volume/", "f_*.npy")))

        save_femur_seg_coords(masks, opt, base_path)
        seg_coords = sorted(glob.glob(os.path.join(self.data_path, "mask/seg_coords/", "f_*.npy")))
        # seg_coords = sorted(glob.glob(os.path.join(base_path, "2022_QIM_52_Bone", "femur_*", "mask", "seg_coords", "zyx_seg_idx.npy")))

        if opt['dataset_opt']['name'] == "Synthetic_2022_QIM_52_Bone":
            if opt['dataset_opt']['degradation_type'] == "kspace_trunc":
                save_kspace_femur_images(images_HR, opt)
                synthetic_dir = "kspace_" + str(opt['dataset_opt']['trunc_factor']) + "/volume/"
            else:
                save_low_resolution_femur_images(images_HR, opt)
                synthetic_dir = "synthetic_lr_" + str(opt['up_factor']) + "x" + "/volume/"
            images_LR = sorted(glob.glob(os.path.join(self.data_path, synthetic_dir, "f_*.npy")))
        else:
            images_LR = sorted(glob.glob(os.path.join(self.data_path, "clinical/volume/", "f_*.npy")))

        # Remove f_031, according to sophia this scan is faulty
        images_HR = [string for string in images_HR if "f_031.npy" not in string]
        images_LR = [string for string in images_LR if "f_031.npy" not in string]
        masks = [string for string in masks if "f_031.npy" not in string]
        seg_coords = [string for string in seg_coords if "f_031.npy" not in string]
        
        if opt['dataset_opt']['name'] == "2022_QIM_52_Bone":
            # Remove f_086 from test set
            images_HR = [string for string in images_HR if "f_086.npy" not in string]
            images_LR = [string for string in images_LR if "f_086.npy" not in string]
            masks = [string for string in masks if "f_086.npy" not in string]
            seg_coords = [string for string in seg_coords if "f_086.npy" not in string]

        if self.apply_split:
            # Apply the split to each list
            self.HR_train, self.HR_test = split_paths(images_HR, test_paths)
            self.LR_train, self.LR_test = split_paths(images_LR, test_paths)
            self.masks_train, self.masks_test = split_paths(masks, test_paths)
            self.seg_coords_train, self.seg_coords_test = split_paths(seg_coords, test_paths)

        for i in range(len(self.HR_train)):
            print("HR femur train " + os.path.basename(self.HR_train[i]))
            print("LR femur train " + os.path.basename(self.LR_train[i]))
            print("Mask train " + os.path.basename(self.masks_train[i]))
            print("Segmentation coords train " + os.path.basename(self.seg_coords_train[i]))

        for i in range(len(self.HR_test)):
            print("HR femur test " + os.path.basename(self.HR_test[i]))
            print("LR femur test " + os.path.basename(self.LR_test[i]))
            print("Mask test " + os.path.basename(self.masks_test[i]))
            print("Segmentation coords test " + os.path.basename(self.seg_coords_test[i]))

    def get_file_paths(self):

        if self.synthetic:
            train_files = [{"H": img_HR, "L": img_LR, "mask": mask, "seg_coords": seg_coords} for img_HR, img_LR, mask, seg_coords in zip(self.HR_train, self.LR_train, self.masks_train, self.seg_coords_train)]
            test_files = [{"H": img_HR, "L": img_LR, "mask": mask, "seg_coords": seg_coords} for img_HR, img_LR, mask, seg_coords in zip(self.HR_test, self.LR_test, self.masks_test, self.seg_coords_test)]
        else:
            train_files = [{"H": img_HR, "L": img_LR, "mask": mask, "seg_coords": seg_coords} for img_HR, img_LR, mask, seg_coords in zip(self.HR_train, self.LR_train, self.masks_train, self.seg_coords_train)]
            test_files = [{"H": img_HR, "L": img_LR, "mask": mask, "seg_coords": seg_coords} for img_HR, img_LR, mask, seg_coords in zip(self.HR_test, self.LR_test, self.masks_test, self.seg_coords_test)]

        return train_files, test_files

    def get_transforms(self, mode):

        # Define transforms for Femur dataset
        if self.opt['dataset_opt']["dataset_type"] == "MonaiSmartCacheDataset" or self.opt['dataset_opt']["dataset_type"] == "CacheDataset":
            data_trans = Femur_transforms_cachableV2(self.opt, mode, self.synthetic)
        else:
            data_trans = Femur_transforms(self.opt, mode, self.synthetic)
        transforms = data_trans.get_transforms()
        return transforms

    def get_baseline_transforms(self, mode):

        data_trans = Femur_baseline_transformsV2(self.opt, mode, self.synthetic)
        transforms = data_trans.get_transforms()
        return transforms

    # TODO: incorporate functions such as these in data.dataset.Dataset class. This way it will be dependent on the dataset.
    def save_point_feats(self, point_feats, data_path, opt, directory_name):
        sample_dir = glob.glob(data_path)
        sample_names = [os.path.basename(dir) for dir in sample_dir]
        if opt['dataset_opt']['name'].find("2022_QIM_52_Bone") >= 0:
            dir = glob.glob(data_path)
            np.save(
                f"{sample_dir[sample_idx]}/{directory_name}/point_feats/{os.path.basename(sample_filenames[sample_idx]).rsplit('.', 1)[0]}",
                point_feats)
        else:
            point_feats_dir = f"{data_path}train/{directory_name}/point_feats/{os.path.basename(sample_filenames[sample_idx]).rsplit('.', 1)[0]}_{sample_idx:04d}"
            np.save(point_feats_dir, point_feats)