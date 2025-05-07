import os
import glob
from data.train_transforms import BasicSRTransforms
import monai.transforms as mt

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
        param = parse_transform_params(self.opt)
        trans_list = []
        trans_list.append(mt.LoadImaged(keys=["H"], dtype=None))  # Load the image
        trans_list.append(mt.EnsureChannelFirstd(keys=["H"], channel_dim=param.channel_dim))  # Load the image
        trans_list.append(mt.SignalFillEmptyd(keys=["H"], replacement=0))

        # Normalization and scaling
        if param.norm_type == "scale_intensity":
            trans_list.append(mt.ScaleIntensityd(keys=["H"], minv=0.0, maxv=1.0))
        elif param.norm_type == "znormalization":
            trans_list.append(mt.NormalizeIntensityd(keys=["H"]))

        # Foreground cropping
        if param.sample_crop_pad_type == "sample_crop_foreground":
            trans_list.append(mt.CropForegroundd(keys=["H"], source_key="H", margin=0, select_fn=lambda a: a > 0.05, k_divisible=4))
        elif param.sample_crop_pad_type == "sample_divisible_padding":
            trans_list.append(mt.DivisiblePadd(keys=["H"], k=4, mode="constant"))

        # Minimum padding
        trans_list.append(mt.SpatialPadd(keys=["H"], spatial_size=[param.size_hr, param.size_hr, param.size_hr], mode="constant", value=0))

        # Create LR image
        trans_list.append(mt.CopyItemsd(keys=["H"], times=1, names=["L"]))
        if param.degradation_type == "resize":
            # Apply Gaussian blur
            if param.blur_method == '3d_gaussian_blur':
                trans_list.append(GaussianblurImaged(keys=["L"], blur_sigma=param.blur_sigma))
            elif param.blur_method == 'monai_gaussian_blur':
                trans_list.append(mt.GaussianSmoothd(keys=["L"], sigma=param.blur_sigma))
            # Resize the image
            trans_list.append(mt.Zoomd(keys=["L"], zoom=1 / param.up_factor, mode=param.downsampling_method, align_corners=True, keep_size=False))

        elif param.degradation_type == "kspace_trunc":
            trans_list.append(KspaceTruncd(keys=["L"], trunc_factor=param.trunc_factor, norm_val=1.0, slice_dim=param.kspace_trunc_dim))

        # Pad for MTVNet
        trans_list.append(mt.BorderPadd(keys=["L"], spatial_border=[param.pad_size, param.pad_size, param.pad_size], mode='constant'))

        # Random crop
        if not baseline:
            if param.implicit:
                trans_list.append(RandomCropPairImplicitd(param.size_lr, param.up_factor, param.foreground_thresh, mode))
            else:
                if param.patch_crop_type == "random_spatial":
                    trans_list.append(RandomCropUniform(param.size_lr, param.up_factor, param.pad_size, param.input_type))
                elif self.patch_crop_type == "random_foreground":
                    trans_list.append(RandomCropForeground(param.size_lr, param.up_factor, param.foreground_thresh, param.pad_size, param.input_type))
                elif self.patch_crop_type == "random_label":
                    trans_list.append(RandomCropLabel(param.size_lr, param.up_factor, param.pad_size, param.input_type, param.mask_mode))

        return mt.Compose(trans_list)