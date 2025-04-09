import os
import glob
from data.train_transforms import BasicSRTransforms

class Dataset_BRATS2023():
    def __init__(self, opt):
        self.opt = opt
        self.patch_size_hr = opt['dataset_opt']['patch_size_hr']
        self.patch_size_lr = opt['dataset_opt']['patch_size']
        self.degradation_type = opt['dataset_opt']['degradation_type']

        if opt['run_type'] == "HOME PC":
            self.data_path = "../Vedrana_master_project/3D_datasets/datasets/BraTS2023/"
        elif opt['cluster'] == "TITANS":
            self.data_path = "/scratch/aulho/Python/3D_datasets/datasets/BraTS2023/"
        else:  # Default is opt['cluster'] = "DTU_HPC"
            self.data_path = "../3D_datasets/datasets/BraTS2023/"
        self.HR_train = sorted(glob.glob(os.path.join(self.data_path, "train/BraTS-GLI*", "*t1c.nii.gz")))
        self.HR_test = sorted(glob.glob(os.path.join(self.data_path, "test/BraTS-GLI*", "*t1c.nii.gz")))

    def get_file_paths(self):

        train_files = [{"H": img_HR} for img_HR in self.HR_train]
        test_files = [{"H": img_HR} for img_HR in self.HR_test]

        return train_files, test_files

    def get_transforms(self, mode):

        self.mode = mode

        # Define transforms for BRATS2023
        data_trans = BasicSRTransforms(self.opt, mode) #Resize_transformsV2(self.opt, mode)

        transforms = data_trans.get_transforms()

        return transforms

    def get_baseline_transforms(self, mode="train"):

        self.mode = mode

        # Define transforms for HCP_1200
        data_trans = BasicSRTransforms(self.opt, mode)  # Resize_transformsV2(self.opt, mode)

        transforms = data_trans.get_transforms(baseline=True)

        return transforms
