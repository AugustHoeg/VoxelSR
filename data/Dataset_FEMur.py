import os
from os.path import dirname as pd
import numpy as np
import glob
import torch
import monai.transforms as mt
from data.train_transforms import BasicSRTransforms

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
        femur_name = os.path.basename(pd(pd(mask)))
        seg_coords_dir = os.path.join(base_path, femur_name, "MS", "seg_coords")
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

class Dataset_FEMur():
    def __init__(self, opt, apply_split=True):
        self.apply_split = apply_split
        self.opt = opt

        self.patch_size_hr = opt['dataset_opt']['patch_size_hr']
        self.patch_size_lr = opt['dataset_opt']['patch_size']
        self.degradation_type = opt['dataset_opt']['degradation_type']

        if opt['run_type'] == "HOME PC":
            self.data_path = "../Vedrana_master_project/3D_datasets/datasets/FEMur/CAD*"
            base_path = "../Vedrana_master_project/3D_datasets/datasets/FEMur"
            test_paths = ["CAD045.npy", "CAD050.npy", "CAD054.npy", "CAD045.nii", "CAD050.nii", "CAD054.nii"]
        elif opt['cluster'] == "TITANS":
            self.data_path = "/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/LUND_data/CAD*"
            base_path = "/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/LUND_data/"
            test_paths = ["CAD045.npy", "CAD050.npy", "CAD054.npy", "CAD045.nii", "CAD050.nii", "CAD054.nii"]
        else:  # Default is opt['cluster'] = "DTU_HPC"
            self.data_path = "/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/LUND_data/CAD*"
            base_path = "/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/LUND_data/"
            test_paths = ["CAD045.npy", "CAD050.npy", "CAD054.npy", "CAD045.nii", "CAD050.nii", "CAD054.nii"]

        images_HR = sorted(glob.glob(os.path.join(self.data_path, "HR/", "CAD*.npy")))
        images_LR = sorted(glob.glob(os.path.join(self.data_path, "LR/", "CAD*.nii")))
        masks = sorted(glob.glob(os.path.join(self.data_path, "MS/", "CAD[0-9][0-9][0-9].npy")))

        #print("masks", masks)

        save_femur_seg_coords(masks, opt, base_path)
        seg_coords = sorted(glob.glob(os.path.join(self.data_path, "MS/seg_coords/", "CAD*.npy")))

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

        train_files = [{"H": img_HR, "L": img_LR, "mask": mask, "seg_coords": seg_coords} for img_HR, img_LR, mask, seg_coords in zip(self.HR_train, self.LR_train, self.masks_train, self.seg_coords_train)]
        test_files = [{"H": img_HR, "L": img_LR, "mask": mask, "seg_coords": seg_coords} for img_HR, img_LR, mask, seg_coords in zip(self.HR_test, self.LR_test, self.masks_test, self.seg_coords_test)]

        return train_files, test_files

    def get_transforms(self, mode):

        self.mode = mode

        # Define transforms for FEMur
        data_trans = BasicSRTransforms(self.opt, mode) #Resize_transformsV2(self.opt, mode)

        transforms = data_trans.get_transforms_FEMur()

        return transforms

    def get_baseline_transforms(self, mode="train"):

        self.mode = mode

        # Define transforms for HCP_1200
        data_trans = BasicSRTransforms(self.opt, mode)  # Resize_transformsV2(self.opt, mode)

        transforms = data_trans.get_transforms_FEMur(baseline=True)

        return transforms

        '''  # This is the original code for baseline transformations used in MTVNet paper
        self.mode = mode

        # Define transforms for HCP_1200
        if self.degradation_type == "resize":
            data_trans = Resize_baseline_transformsV2(self.opt, mode)
        elif self.degradation_type == "kspace_trunc":
            data_trans = Kspace_baseline_transforms(self.opt)

        transforms = data_trans.get_transforms()

        return transforms
        '''
