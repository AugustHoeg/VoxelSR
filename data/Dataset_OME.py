import os
import torch
import glob
import numpy as np
from data.train_transforms import BasicSRTransforms, GlobalScaleIntensityd
import monai.transforms as mt
from data.train_transforms import GaussianblurImaged, KspaceTruncd, \
    RandomCropPairImplicitd, RandomCropUniform, RandomCropForeground, \
    RandomCropLabel, get_context_pad_size # CustomRand3DElasticd

class Dataset_OME():
    def __init__(self, opt):

        self.synthetic = opt['dataset_opt']['synthetic']

        self.opt = opt
        self.patch_size_hr = opt['dataset_opt']['patch_size_hr']
        self.patch_size_lr = opt['dataset_opt']['patch_size']
        self.degradation_type = opt['dataset_opt']['degradation_type']

        if opt['run_type'] == "HOME PC":
            self.data_path = "../Vedrana_master_project/3D_datasets/datasets/danmax/binning_bone"
            self.group_name = "volume"
        elif opt['cluster'] == "TITANS":
            self.data_path = "/scratch/aulho/Python/3D_datasets/datasets/danmax/binning_bone"
            self.group_name = "HR"
        else:  # Default is opt['cluster'] = "DTU_HPC"
            self.data_path = "../3D_datasets/datasets/danmax"
            self.group_name = "HR"

        self.OME_train = sorted(glob.glob(os.path.join(self.data_path, "bone_1_ome.zarr")))
        #self.LR_train = sorted(glob.glob(os.path.join(self.data_path, "train/", "*.zarr")))

        # if self.synthetic:
        #     self.LR_train = sorted(glob.glob(os.path.join(self.data_path, "train/HR_chunks_down4/", "*.npy")))

        self.OME_test = sorted(glob.glob(os.path.join(self.data_path, "bone_2_ome.zarr")))
        # self.LR_test = sorted(glob.glob(os.path.join(self.data_path, "test/", "*.zarr")))

        # if self.synthetic:
        #     self.LR_test = sorted(glob.glob(os.path.join(self.data_path, "test/HR_chunks_down4/", "*.npy")))

    def get_file_paths(self):

        #train_files = [{"H": img_HR, "L": img_LR} for img_HR, img_LR in zip(self.HR_train, self.LR_train)]
        #test_files = [{"H": img_HR, "L": img_LR} for img_HR, img_LR in zip(self.HR_test, self.LR_test)]

        return self.OME_train, self.OME_test

    def get_transforms(self, mode="train", baseline=False):

        # parse the options
        p = self.opt
        pdata = p.dataset_opt
        pdata.pad_size = get_context_pad_size(p)

        trans_list = []
        #trans_list.append(mt.LoadImaged(keys=["H", "L"], dtype=None))  # Load the image
        trans_list.append(mt.EnsureChannelFirstd(keys=["H", "L"], channel_dim=pdata.channel_dim))  # Load the image
        trans_list.append(mt.CastToTyped(keys=["H", "L"], dtype=torch.float32))  # Cast to float32
        trans_list.append(mt.SignalFillEmptyd(keys=["H", "L"], replacement=0))

        # Normalization and scaling
        if pdata.norm_type == "scale_intensity":
            trans_list.append(GlobalScaleIntensityd(keys=["H"], global_min=-0.000936, global_max=0.000998))
            trans_list.append(GlobalScaleIntensityd(keys=["L"], global_min=-0.002063, global_max=0.002476))
        elif pdata.norm_type == "znormalization":
            trans_list.append(mt.NormalizeIntensityd(keys=["H", "L"]))

        # Minimum padding
        # trans_list.append(mt.SpatialPadd(keys=["H"], spatial_size=[pdata.patch_size_hr, pdata.patch_size_hr, pdata.patch_size_hr], mode="constant", value=0))

        # Pad for MTVNet
        trans_list.append(mt.BorderPadd(keys=["L"], spatial_border=[pdata.pad_size, pdata.pad_size, pdata.pad_size], mode='constant'))

        # # Random crop
        # if not baseline:
        #     if p.model_opt.model == "implicit":
        #         trans_list.append(RandomCropPairImplicitd(pdata.patch_size, p.up_factor, pdata.foreground_thresh, mode))
        #     else:
        #         if pdata.patch_crop_type == "random_spatial":
        #             trans_list.append(RandomCropUniform(pdata.patch_size, p.up_factor, pdata.pad_size, p.input_type))
        #         elif pdata.patch_crop_type == "random_foreground":
        #             trans_list.append(RandomCropForeground(pdata.patch_size, p.up_factor, pdata.foreground_thresh, pdata.pad_size, p.input_type))
        #         elif pdata.patch_crop_type == "random_label":
        #             trans_list.append(RandomCropLabel(pdata.patch_size, p.up_factor, pdata.pad_size, p.input_type, p.mask_mode))

        # Augmentations after crop
        if mode == "train":
            # Random augmentations
            # trans_list.append(mt.RandFlipd(keys=["H", "L"], prob=0.20, spatial_axis=[0, 1, 2]))  # Random flip

            trans_list.append(mt.RandFlipd(keys=["H", "L"], spatial_axis=0, prob=0.5))
            trans_list.append(mt.RandFlipd(keys=["H", "L"], spatial_axis=1, prob=0.5))
            trans_list.append(mt.RandFlipd(keys=["H", "L"], spatial_axis=2, prob=0.5))
            # trans_list.append(mt.RandRotated(keys=["H", "L"], prob=0.50, range_x=(-np.pi / 6, np.pi / 6), range_y=(-np.pi / 6, np.pi / 6), range_z=(-np.pi / 6, np.pi / 6), mode="bilinear", align_corners=True, keep_size=True))
            # trans_list.append(mt.Rand3DElasticd(keys=["H", "L"], prob=0.80, sigma_range=(4, 8), magnitude_range=(-0.1, 0.1), mode="bilinear"))
            # trans_list.append(CustomRand3DElasticd(keys=["H", "L"], prob=0.80, sigma_range=(4, 8), magnitude_range=(-0.1, 0.1), mode="bilinear"))

            # trans_list.append(mt.RandZoomd(keys=["H", "L"], prob=0.50, min_zoom=0.9, max_zoom=1.1, mode="bilinear", align_corners=True, keep_size=True))
            # trans_list.append(mt.RandGaussianNoised(keys=["L"], prob=0.2, mean=0.0, std=0.005))

        return mt.Compose(trans_list)

    # Set length of dataset to batch_size (HACK)
    # TODO: Remove this hack when the issue when sampling more patch per image is resolved
    # def __len__(self):
    #    return len(self.opt.dataset_opt.batch_size)
