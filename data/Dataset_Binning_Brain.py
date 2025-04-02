import os
import glob
import numpy as np
from data.train_transforms import BasicSRTransforms

def split_paths(paths, test_paths):
    train = [string for string in paths if not any(test in string for test in test_paths)]
    test = [string for string in paths if any(test in string for test in test_paths)]
    return train, test

class Dataset_Binning_Brain():

    def __init__(self, opt, apply_split=True):
        self.apply_split = apply_split
        self.opt = opt

        if opt['run_type'] == "HOME PC":
            self.data_path = "../Vedrana_master_project/3D_datasets/datasets/danmax/binning_brain"
            test_paths = ["brain_1_test.npy"]

        elif opt['cluster'] == "TITANS":
            raise Exception(f"Dataset Binning Bone not supported for run type: {opt['run_type']}")

        else:  # Default is opt['cluster'] = DTU_HPC
            self.data_path = "../3D_datasets/datasets/danmax/binning_bone" # Use this path to read from personal scratch space
            test_paths = ["brain_1_test.npy"]

        images_HR = sorted(glob.glob(os.path.join(self.data_path, "bin2x2/", "brain_*", "brain_*.npy")))
        images_LR = sorted(glob.glob(os.path.join(self.data_path, "bin4x4/", "brain_*", "brain_*.npy")))

        if self.apply_split:
            # Apply the split to each list
            self.HR_train, self.HR_test = split_paths(images_HR, test_paths)
            self.LR_train, self.LR_test = split_paths(images_LR, test_paths)

        for i in range(len(self.HR_train)):
            print("HR train " + os.path.basename(self.HR_train[i]))
            print("LR train " + os.path.basename(self.LR_train[i]))

        for i in range(len(self.HR_test)):
            print("HR test " + os.path.basename(self.HR_test[i]))
            print("LR test " + os.path.basename(self.LR_test[i]))

    def get_file_paths(self):

        train_files = [{"H": img_HR, "L": img_LR} for img_HR, img_LR in zip(self.HR_train, self.LR_train)]
        test_files = [{"H": img_HR, "L": img_LR} for img_HR, img_LR in zip(self.HR_test, self.LR_test)]

        return train_files, test_files

    def get_transforms(self, mode="train"):

        self.mode = mode

        # Define transforms
        data_trans = BasicSRTransforms(self.opt, mode)

        transforms = data_trans.get_transforms_binning_brain()

        return transforms

    def get_baseline_transforms(self, mode="train"):

        self.mode = mode

        # Define transforms for HCP_1200
        data_trans = BasicSRTransforms(self.opt, mode)  # Resize_transformsV2(self.opt, mode)

        transforms = data_trans.get_transforms_binning_brain(baseline=True)

        return transforms

