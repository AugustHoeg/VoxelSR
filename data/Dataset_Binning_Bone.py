import os
import glob
import numpy as np
from data.binning_train_transforms import binning_transforms_cachable
from data.binning_baseline_transforms import binning_baseline_transforms

def split_paths(paths, test_paths):
    train = [string for string in paths if not any(test in string for test in test_paths)]
    test = [string for string in paths if any(test in string for test in test_paths)]
    return train, test

class Dataset_Binning_Bone():

    def __init__(self, opt, apply_split=True):
        self.apply_split = apply_split
        self.opt = opt

        self.synthetic = "Synthetic" in opt['datasets']['name']

        if opt['run_type'] == "HOME PC":
            raise Exception(f"Dataset Binning Bone not supported for run type: {opt['run_type']}")

        elif opt['cluster'] == "TITANS":
            raise Exception(f"Dataset Binning Bone not supported for run type: {opt['run_type']}")

        else:  # Default is opt['cluster'] = DTU_HPC
            self.data_path = "../3D_datasets/datasets/danmax/binning_bone" # Use this path to read from personal scratch space
            base_path = "../3D_datasets/datasets/"
            test_paths = ["bone_2_crop_norm.npy"]

        images_HR = sorted(glob.glob(os.path.join(self.data_path, "bin2x2/", "bone_*", "bone_*_crop_norm.npy")))
        images_LR = sorted(glob.glob(os.path.join(self.data_path, "bin4x4/", "bone_*", "bone_*_crop_norm.npy")))

        if self.apply_split:
            # Apply the split to each list
            self.HR_train, self.HR_test = split_paths(images_HR, test_paths)
            self.LR_train, self.LR_test = split_paths(images_LR, test_paths)

        for i in range(len(self.HR_train)):
            print("HR train " + os.path.basename(self.HR_train[i]))
            print("LR train " + os.path.basename(self.LR_train[i]))

        for i in range(len(self.HR_test)):
            print("HR test test " + os.path.basename(self.HR_test[i]))
            print("LR test test " + os.path.basename(self.LR_test[i]))

    def get_file_paths(self):

        train_files = [{"H": img_HR, "L": img_LR} for img_HR, img_LR in zip(self.HR_train, self.LR_train)]
        test_files = [{"H": img_HR, "L": img_LR} for img_HR, img_LR in zip(self.HR_test, self.LR_test)]

        return train_files, test_files

    def get_transforms(self, mode):

        # Define transforms for Femur dataset
        if self.opt['datasets']["dataset_type"] == "MonaiSmartCacheDataset" or self.opt['datasets']["dataset_type"] == "CacheDataset":
            data_trans = binning_transforms_cachable(self.opt, mode)
            #data_trans = Femur_transforms_cachableV2(self.opt, mode, self.synthetic)
        else:
            raise Exception(f"Dataset type {self.opt['datasets']['dataset_type']} not implemeted for dataset {self.opt['datasets']['name']}")
            #data_trans = binning_transforms(self.opt, mode)

        transforms = data_trans.get_transforms()
        return transforms

    def get_baseline_transforms(self, mode):

        data_trans = binning_baseline_transforms(self.opt, mode, self.synthetic)
        transforms = data_trans.get_transforms()
        return transforms

    # TODO: incorporate functions such as these in data.dataset.Dataset class. This way it will be dependent on the dataset.
    def save_point_feats(self, point_feats, data_path, opt, directory_name):
        sample_dir = glob.glob(data_path)
        sample_names = [os.path.basename(dir) for dir in sample_dir]
        if opt['datasets']['name'].find("2022_QIM_52_Bone") >= 0:
            dir = glob.glob(data_path)
            np.save(
                f"{sample_dir[sample_idx]}/{directory_name}/point_feats/{os.path.basename(sample_filenames[sample_idx]).rsplit('.', 1)[0]}",
                point_feats)
        else:
            point_feats_dir = f"{data_path}train/{directory_name}/point_feats/{os.path.basename(sample_filenames[sample_idx]).rsplit('.', 1)[0]}_{sample_idx:04d}"
            np.save(point_feats_dir, point_feats)