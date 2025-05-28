import os
import glob
import numpy as np
from data.train_transforms import BasicSRTransforms
import monai.transforms as mt
from data.train_transforms import GaussianblurImaged, KspaceTruncd, \
    RandomCropPairImplicitd, RandomCropUniform, RandomCropForeground, \
    RandomCropLabel, get_context_pad_size # CustomRand3DElasticd


class Dataset_HCP_1200():
    def __init__(self, opt):
        self.opt = opt
        self.patch_size_hr = opt['dataset_opt']['patch_size_hr']
        self.patch_size_lr = opt['dataset_opt']['patch_size']
        self.degradation_type = opt['dataset_opt']['degradation_type']

        if opt['run_type'] == "HOME PC":
            self.data_path = "../Vedrana_master_project/3D_datasets/datasets/HCP_1200/" # maybe need to edit in pycharm due to inconsisted indent
            self.HR_train = sorted(glob.glob(os.path.join(self.data_path, "train/*/T1w", "T1w_acpc_dc.nii")))
            self.HR_test = sorted(glob.glob(os.path.join(self.data_path, "test/*/T1w", "T1w_acpc_dc.nii")))
        elif opt['cluster'] == "TITANS":
            self.data_path = "/scratch/aulho/Python/3D_datasets/datasets/HCP_1200_unprocessed/"
            self.HR_train = sorted(glob.glob(os.path.join(self.data_path, "train", "*_3T_T1w_MPR1.nii.gz")))
            self.HR_test = sorted(glob.glob(os.path.join(self.data_path, "test", "*_3T_T1w_MPR1.nii.gz")))
        else:  # Default is opt['cluster'] = DTU_HPC
            self.data_path = "../3D_datasets/datasets/HCP_1200_unprocessed/"
            self.HR_train = sorted(glob.glob(os.path.join(self.data_path, "train", "*_3T_T1w_MPR1.nii.gz")))
            self.HR_test = sorted(glob.glob(os.path.join(self.data_path, "test", "*_3T_T1w_MPR1.nii.gz")))

    def get_file_paths(self):

        train_files = [{"H": img_HR} for img_HR in self.HR_train]
        test_files = [{"H": img_HR} for img_HR in self.HR_test]

        return train_files, test_files

    def get_transforms(self, mode="train", baseline=False):

        # parse the options
        p = self.opt
        pdata = p.dataset_opt
        pdata.pad_size = get_context_pad_size(p)

        trans_list = []
        trans_list.append(mt.LoadImaged(keys=["H"], dtype=None))  # Load the image
        trans_list.append(mt.EnsureChannelFirstd(keys=["H"], channel_dim=pdata.channel_dim))  # Load the image
        trans_list.append(mt.SignalFillEmptyd(keys=["H"], replacement=0))

        # Normalization and scaling
        if pdata.norm_type == "scale_intensity":
            trans_list.append(mt.ScaleIntensityd(keys=["H"], minv=0.0, maxv=1.0))
        elif pdata.norm_type == "znormalization":
            trans_list.append(mt.NormalizeIntensityd(keys=["H"]))

        # Foreground cropping
        if pdata.sample_crop_pad_type == "sample_crop_foreground":
            trans_list.append(mt.CropForegroundd(keys=["H"], source_key="H", margin=0, select_fn=lambda a: a > 0.05, k_divisible=4))
        elif pdata.sample_crop_pad_type == "sample_divisible_padding":
            trans_list.append(mt.DivisiblePadd(keys=["H"], k=4, mode="constant"))

        # Minimum padding
        trans_list.append(mt.SpatialPadd(keys=["H"], spatial_size=[pdata.patch_size_hr, pdata.patch_size_hr, pdata.patch_size_hr], mode="constant", value=0))

        # Create LR image
        trans_list.append(mt.CopyItemsd(keys=["H"], times=1, names=["L"]))
        if pdata.degradation_type == "resize":
            # Apply Gaussian blur
            if pdata.blur_method == '3d_gaussian_blur':
                trans_list.append(GaussianblurImaged(keys=["L"], blur_sigma=pdata.blur_sigma))
            elif pdata.blur_method == 'monai_gaussian_blur':
                trans_list.append(mt.GaussianSmoothd(keys=["L"], sigma=pdata.blur_sigma))
            # Resize the image
            trans_list.append(mt.Zoomd(keys=["L"], zoom=1 / p.up_factor, mode=pdata.downsampling_method, align_corners=True, keep_size=False))

        elif pdata.degradation_type == "kspace_trunc":
            trans_list.append(KspaceTruncd(keys=["L"], trunc_factor=pdata.trunc_factor, norm_val=1.0, slice_dim=pdata.kspace_trunc_dim))

        # Pad for MTVNet
        trans_list.append(mt.BorderPadd(keys=["L"], spatial_border=[pdata.pad_size, pdata.pad_size, pdata.pad_size], mode='constant'))

        # Random crop
        if not baseline:
            if p.model_opt.model == "implicit":
                trans_list.append(RandomCropPairImplicitd(pdata.patch_size, p.up_factor, pdata.foreground_thresh, mode))
            else:
                if pdata.patch_crop_type == "random_spatial":
                    trans_list.append(RandomCropUniform(pdata.patch_size, p.up_factor, pdata.pad_size, p.input_type))
                elif pdata.patch_crop_type == "random_foreground":
                    trans_list.append(RandomCropForeground(pdata.patch_size, p.up_factor, pdata.foreground_thresh, pdata.pad_size, p.input_type))
                elif pdata.patch_crop_type == "random_label":
                    trans_list.append(RandomCropLabel(pdata.patch_size, p.up_factor, pdata.pad_size, p.input_type, p.mask_mode))

        # Augmentations after crop
        if mode == "train":
            # Random augmentations
            # trans_list.append(mt.RandFlipd(keys=["H", "L"], prob=0.20, spatial_axis=[0, 1, 2]))  # Random flip

            trans_list.append(mt.RandFlipd(keys=["H", "L"], spatial_axis=0, prob=0.5))
            trans_list.append(mt.RandFlipd(keys=["H", "L"], spatial_axis=1, prob=0.5))
            trans_list.append(mt.RandFlipd(keys=["H", "L"], spatial_axis=2, prob=0.5))
            trans_list.append(mt.RandRotated(keys=["H", "L"], prob=0.50, range_x=(-np.pi / 6, np.pi / 6), range_y=(-np.pi / 6, np.pi / 6), range_z=(-np.pi / 6, np.pi / 6), mode="bilinear"))
            # trans_list.append(mt.Rand3DElasticd(keys=["H", "L"], prob=0.80, sigma_range=(4, 8), magnitude_range=(-0.1, 0.1), mode="bilinear"))
            # trans_list.append(CustomRand3DElasticd(keys=["H", "L"], prob=0.80, sigma_range=(4, 8), magnitude_range=(-0.1, 0.1), mode="bilinear"))

            trans_list.append(mt.RandZoomd(keys=["H", "L"], prob=0.50, min_zoom=0.9, max_zoom=1.1, mode="bilinear", align_corners=True, keep_size=True))
            # trans_list.append(mt.RandGaussianNoised(keys=["L"], prob=0.2, mean=0.0, std=0.005))

        return mt.Compose(trans_list)