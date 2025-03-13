import os
import glob
from data.train_transforms import BasicSRTransforms
from data.baseline_transforms import Resize_baseline_transformsV2

class Dataset_IXI():
    def __init__(self, opt):
        self.opt = opt
        self.patch_size_hr = opt['dataset_opt']['patch_size_hr']
        self.patch_size_lr = opt['dataset_opt']['patch_size']
        self.degradation_type = opt['dataset_opt']['degradation_type']

        if opt['run_type'] == "HOME PC":
            self.data_path = "../Vedrana_master_project/3D_datasets/datasets/IXI/"
        elif opt['cluster'] == "TITANS":
            self.data_path = "/scratch/aulho/Python/3D_datasets/datasets/IXI/"
        else:  # Default is opt['cluster'] = "DTU_HPC"
            self.data_path = "../3D_datasets/datasets/IXI/"

        self.HR_train = sorted(glob.glob(os.path.join(self.data_path, "train", "*.nii")))
        self.HR_test = sorted(glob.glob(os.path.join(self.data_path, "test", "*.nii")))

    def get_file_paths(self):

        train_files = [{"H": img_HR} for img_HR in self.HR_train]
        test_files = [{"H": img_HR} for img_HR in self.HR_test]

        return train_files, test_files

    def get_transforms(self, mode):

        self.mode = mode

        # Define transforms for IXI
        data_trans = BasicSRTransforms(self.opt, mode) #Resize_transformsV2(self.opt, mode)

        transforms = data_trans.get_transforms()

        return transforms

    def get_baseline_transforms(self, mode="train"):

        self.mode = mode

        # Define transforms for HCP_1200
        data_trans = BasicSRTransforms(self.opt, mode)  # Resize_transformsV2(self.opt, mode)

        transforms = data_trans.get_transforms(baseline=True)

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
