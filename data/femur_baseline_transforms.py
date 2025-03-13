import torch
import monai.transforms as mt
import torch

from data.baseline_transforms import Resize_functional
from data.train_transforms import GaussianblurImaged, ImplicitModelTransformd

class PadToMultiple():
    def __init__(self, opt):
        self.patch_size_hr = opt['dataset_opt']['patch_size_hr']
        self.patch_size_lr = opt['dataset_opt']['patch_size']
        self.up_factor = opt['up_factor']

    def get_padding(self, img, overlap_lr=4):
        c, h, w, d = img.shape
        stride = (self.patch_size_hr - self.up_factor * overlap_lr)
        n_patches = torch.ceil(((torch.tensor([h, w, d]) - self.up_factor * overlap_lr) / stride))
        pad_h = int((((n_patches[0] * stride) + self.up_factor * overlap_lr) - h) / 2)
        pad_w = int((((n_patches[1] * stride) + self.up_factor * overlap_lr) - w) / 2)
        pad_d = int((((n_patches[2] * stride) + self.up_factor * overlap_lr) - d) / 2)

        return pad_h, pad_w, pad_d


    def __call__(self, img_dict: dict):

        c, h, w, d = img_dict['H'].shape
        size_even = torch.tensor([h - (h % 2), w - (w % 2), d - (d % 2)])
        img_dict['H'] = img_dict['H'][:, 0:size_even[0], 0:size_even[1], 0:size_even[2]]
        img_dict['L'] = img_dict['L'][:, 0:size_even[0], 0:size_even[1], 0:size_even[2]]
        img_dict['mask'] = img_dict['mask'][:, 0:size_even[0], 0:size_even[1], 0:size_even[2]]

        pad_h, pad_w, pad_d = self.get_padding(img_dict['H'], overlap_lr=4)
        padding = torch.tensor([pad_h, pad_w, pad_d])
        # add padding here equal to padding on each side of the images
        #mt.BorderPad(spatial_border=padding, mode='constant', lazy=False)
        img_dict['H'] = torch.nn.functional.pad(img_dict['H'], pad=(pad_d, pad_d, pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
        img_dict['L'] = torch.nn.functional.pad(img_dict['L'], pad=(pad_d, pad_d, pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
        img_dict['mask'] = torch.nn.functional.pad(img_dict['mask'], pad=(pad_d, pad_d, pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)

        # Adding padding to img_dict to use for cropping later
        img_dict['padding'] = padding

        return img_dict


class CropToMultiple():

    def __init__(self, opt):
        self.patch_size_hr = opt['dataset_opt']['patch_size_hr']
        self.patch_size_lr = opt['dataset_opt']['patch_size']
        self.up_factor = opt['up_factor']

    def get_start_and_dim(self, img):
        c, h, w, d = img.shape
        dim_h = h - h % self.patch_size_hr
        dim_w = w - w % self.patch_size_hr
        dim_d = d - d % self.patch_size_hr
        start_h = h // 2 - dim_h // 2
        start_w = w // 2 - dim_w // 2
        start_d = d // 2 - dim_d // 2
        return dim_h, dim_w, dim_d, start_h, start_w, start_d

    def get_padding(self, img, overlap_lr=4):
        c, h, w, d = img.shape
        stride = (self.patch_size_hr - self.up_factor * overlap_lr)
        n_patches = torch.ceil(((torch.tensor([h, w, d]) - self.up_factor * overlap_lr) / stride))
        pad_h = int((((n_patches[0] * stride) + self.up_factor * overlap_lr) - h) / 2)
        pad_w = int((((n_patches[1] * stride) + self.up_factor * overlap_lr) - w) / 2)
        pad_d = int((((n_patches[2] * stride) + self.up_factor * overlap_lr) - d) / 2)

        return pad_h, pad_w, pad_d


    def __call__(self, img_dict: dict):

        c, h, w, d = img_dict['H'].shape
        size_even = torch.tensor([h - (h % 2), w - (w % 2), d - (d % 2)])
        img_dict['H'] = img_dict['H'][:, 0:size_even[0], 0:size_even[1], 0:size_even[2]]
        img_dict['L'] = img_dict['L'][:, 0:size_even[0], 0:size_even[1], 0:size_even[2]]
        img_dict['mask'] = img_dict['mask'][:, 0:size_even[0], 0:size_even[1], 0:size_even[2]]

        pad_h, pad_w, pad_d = self.get_padding(img_dict['H'], overlap_lr=4)
        # add padding here equal to padding on each side of the images
        #mt.BorderPad(spatial_border=padding, mode='constant', lazy=False)
        img_dict['H'] = torch.nn.functional.pad(img_dict['H'], pad=(pad_d, pad_d, pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
        img_dict['L'] = torch.nn.functional.pad(img_dict['L'], pad=(pad_d, pad_d, pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
        img_dict['mask'] = torch.nn.functional.pad(img_dict['mask'], pad=(pad_d, pad_d, pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)

        return img_dict



class Femur_baseline_transforms():

    def __init__(self, opt):

        self.size_hr = opt['dataset_opt']['patch_size_hr']
        self.size_lr = opt['dataset_opt']['patch_size']
        self.up_factor = opt['up_factor']

        self.blur_sigma = opt['dataset_opt']['blur_sigma']
        self.downsampling_method = opt['dataset_opt']['downsampling_method']
        self.patches_per_batch = opt['dataset_opt']['train']['dataset_params']['patches_per_batch']
        self.channel_dim = opt['dataset_opt']['channel_dim']

        if opt['dataset_opt']['blur_method'] == '3d_gaussian_blur':
            self.blur_transform = GaussianblurImaged(blur_sigma=self.blur_sigma, in_key="L", out_key="L")
        elif opt['dataset_opt']['blur_method'] == 'monai_gaussian_blur':
            self.blur_transform = mt.GaussianSmoothd(keys=["L"], sigma=opt['dataset_opt']['blur_sigma'])

        self.resize_functional = Resize_functional(self.up_factor, self.downsampling_method, in_key='L', out_key='L')
        self.crop_to_mulitple = CropToMultiple(opt)
        self.pad_to_multiple = PadToMultiple(opt)

    def get_transforms(self):

        transforms = mt.Compose(
            [
                mt.LoadImaged(keys=["H", "L", "mask"], dtype=None),
                mt.EnsureChannelFirstd(keys=["H", "L", "mask"], channel_dim=self.channel_dim),
                mt.SignalFillEmptyd(keys=["H", "L"], replacement=0),  # Remove any NaNs
                self.pad_to_multiple,
                #self.crop_to_mulitple,
                self.resize_functional,
            ]
        )

        return transforms


class Femur_baseline_transformsV2():

    def __init__(self, opt, mode="train", synthetic=False):

        self.synthetic = synthetic  # Whether to create LR data from synthetic degradation of HR data

        self.implicit = True if opt['model'] == "implicit" else False
        self.mode = mode

        self.size_hr = opt['dataset_opt']['patch_size_hr']
        self.size_lr = opt['dataset_opt']['patch_size']
        self.up_factor = opt['up_factor']

        self.blur_sigma = opt['dataset_opt']['blur_sigma']
        self.downsampling_method = opt['dataset_opt']['downsampling_method']
        self.patches_per_batch = opt['dataset_opt']['train']['dataset_params']['patches_per_batch']
        self.channel_dim = opt['dataset_opt']['channel_dim']

        if opt['dataset_opt']['blur_method'] == '3d_gaussian_blur':
            self.blur_transform = GaussianblurImaged(blur_sigma=self.blur_sigma, in_key="L", out_key="L")
        elif opt['dataset_opt']['blur_method'] == 'monai_gaussian_blur':
            self.blur_transform = mt.GaussianSmoothd(keys=["L"], sigma=opt['dataset_opt']['blur_sigma'])

        self.resize_functional = Resize_functional(self.up_factor, self.downsampling_method, in_key='L', out_key='L')
        #self.crop_to_mulitple = CropToMultiple(opt)
        #self.pad_to_multiple = PadToMultiple(opt)

        self.implicit_model_transform = ImplicitModelTransformd(opt['up_factor'], mode=mode)

        self.size_lr = opt['dataset_opt']['patch_size']
        self.pad_transform_lr = mt.Identityd(keys=['L'])
        self.divisible_pad = 4 if opt['up_factor'] % 2 == 0 else 3  # Added adaptive divisible padding
        if opt['model_architecture'] == "MTVNet":
            center_size = opt['netG']['context_sizes'][-1]  # fixed assumption of level_ratio = 2

            pad_size = (self.size_lr - center_size) // 2  # pad half of the context on all sides
            if pad_size > 0:
                self.pad_transform_lr = mt.BorderPadd(keys=['L'], spatial_border=[pad_size, pad_size, pad_size], mode='constant')  # Pad here if net is MTVNet
            else:
                self.pad_transform_lr = mt.Identityd(keys=['L'])


    def get_transforms(self):
        if self.synthetic:
            transforms = mt.Compose(
                [
                    # Deterministic Transforms
                    mt.LoadImaged(keys=["H", "L"], dtype=torch.float32),
                    mt.EnsureChannelFirstd(keys=["H"], channel_dim=self.channel_dim),
                    mt.SignalFillEmptyd(keys=["H"], replacement=0),  # Remove any NaNs
                    mt.DivisiblePadd(keys=["H"], k=self.divisible_pad, mode="constant"),  # Ensure HR and LR scans have even dimensions
                    self.pad_transform_lr,
                    # Skip cropping for baseline

                ]
            )

        else:
            transforms = mt.Compose(
                [
                    mt.LoadImaged(keys=["H", "L"], dtype=torch.float32),
                    mt.EnsureChannelFirstd(keys=["H", "L"], channel_dim=self.channel_dim),
                    mt.SignalFillEmptyd(keys=["H", "L"], replacement=0),  # Remove any NaNs
                    mt.DivisiblePadd(keys=["H", "L"], k=self.divisible_pad, mode="constant"),  # Ensure HR and LR scans have even dimensions
                    mt.Zoomd(keys=["L"],
                             zoom=1 / self.up_factor,
                             mode=self.downsampling_method,
                             align_corners=True,
                             keep_size=False,
                             dtype=torch.float32),
                    self.pad_transform_lr,
                ]
            )

        return transforms
