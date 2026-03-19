import os
import torch
import glob
import numpy as np
from data.train_transforms import RandSRFlipd, RandSRRotated, RandSRZoomd, RandSRContrastd, RandSRCLAHEd
import monai.transforms as mt
from data.train_transforms import get_context_pad_size

class Dataset_VoDaSuRe_OME():
    def __init__(self, opt, dataset_path="../3D_datasets/datasets/"):

        self.synthetic = opt['dataset_opt']['synthetic']

        self.opt = opt
        self.patch_size_hr = opt['dataset_opt']['patch_size_hr']
        self.patch_size_lr = opt['dataset_opt']['patch_size']
        self.degradation_type = opt['dataset_opt']['degradation_type']

        self.data_path = dataset_path

        sampling_weights = {"HCP_1200": 3.0,
                            "IXI": 1.0,
                            "LITS": 2.0,
                            "CTSpine1K": 5.0,
                            "LIDC-IDRI": 5.0,
                            "VoDaSuRe": 15.0}

        group_pairs = {}
        group_pairs["HCP_1200"] = {"4": [{"H": "HR/0", "L": "HR/2"}], "2": [{"H": "HR/0", "L": "HR/1"}]}
        group_pairs["IXI"] = {"4": [{"H": "HR/0", "L": "HR/2"}], "2": [{"H": "HR/0", "L": "HR/1"}]}
        group_pairs["LITS"] = {"4": [{"H": "HR/0", "L": "HR/2"}], "2": [{"H": "HR/0", "L": "HR/1"}]}
        group_pairs["CTSpine1K"] = {"4": [{"H": "HR/0", "L": "HR/2"}], "2": [{"H": "HR/0", "L": "HR/1"}]}
        group_pairs["LIDC-IDRI"] = {"4": [{"H": "HR/0", "L": "HR/2"}], "2": [{"H": "HR/0", "L": "HR/1"}]}

        store_type = {"HCP_1200": "LocalStore",
                      "IXI": "LocalStore",
                      "LITS": "LocalStore",
                      "CTSpine1K": "LocalStore",
                      "LIDC-IDRI": "LocalStore",
                      "VoDaSuRe": "LocalStore"}

        train_paths = {}
        test_paths = {}
        if "HCP_1200" in opt['dataset_opt']['datasets']:
            train_paths["HCP_1200"] = glob.glob(os.path.join(self.data_path, "HCP_1200/ome/train/*.zarr"))
            test_paths["HCP_1200"] = glob.glob(os.path.join(self.data_path, "HCP_1200/ome/test/*.zarr"))

        if "IXI" in opt['dataset_opt']['datasets']:
            train_paths["IXI"] = glob.glob(os.path.join(self.data_path, "IXI/ome/train/*.zarr"))
            test_paths["IXI"] = glob.glob(os.path.join(self.data_path, "IXI/ome/test/*.zarr"))

        if "LITS" in opt['dataset_opt']['datasets']:
            train_paths["LITS"] = glob.glob(os.path.join(self.data_path, "LITS/ome/train/*.zarr"))
            test_paths["LITS"] = glob.glob(os.path.join(self.data_path, "LITS/ome/test/*.zarr"))

        if "CTSpine1K" in opt['dataset_opt']['datasets']:
            train_paths["CTSpine1K"] = glob.glob(os.path.join(self.data_path, "CTSpine1K/ome/train/*.zarr"))
            test_paths["CTSpine1K"] = glob.glob(os.path.join(self.data_path, "CTSpine1K/ome/test/*.zarr"))

        if "LIDC-IDRI" in opt['dataset_opt']['datasets']:
            train_paths["LIDC-IDRI"] = glob.glob(os.path.join(self.data_path, "LIDC_IDRI/ome/train/*.zarr"))
            test_paths["LIDC-IDRI"] = glob.glob(os.path.join(self.data_path, "LIDC_IDRI/ome/test/*.zarr"))

        if "VoDaSuRe" in opt['dataset_opt']['datasets']:
            train_paths["VoDaSuRe"] = glob.glob(os.path.join(self.data_path, "VoDaSuRe/ome/train/*.zarr"))
            test_paths["VoDaSuRe"] = glob.glob(os.path.join(self.data_path, "VoDaSuRe/ome/test/*.zarr"))

            if "domain_test_wood" in opt:
                train_paths["VoDaSuRe"] = [os.path.join(self.data_path, "VoDaSuRe/ome/train/Larch_B_bin1x1_ome_1.zarr"),
                                           os.path.join(self.data_path, "VoDaSuRe/ome/train/Oak_A_bin1x1_ome_1.zarr"),
                                           os.path.join(self.data_path, "VoDaSuRe/ome/train/Bamboo_A_bin1x1_ome_1.zarr")]

                test_paths["VoDaSuRe"] = [os.path.join(self.data_path, "VoDaSuRe/ome/test/Cypress_A_bin1x1_ome_0.zarr"),
                                          os.path.join(self.data_path, "VoDaSuRe/ome/test/Elm_A_bin1x1_ome_0.zarr")]

                print("Running domain generalization test on wood samples!")
                print("Train paths: ", train_paths["VoDaSuRe"])
                print("Test paths: ", test_paths["VoDaSuRe"])

            if "domain_test_bone" in opt:
                train_paths["VoDaSuRe"] = [os.path.join(self.data_path, "VoDaSuRe/ome/train/Femur_15_80kV_ome.zarr"),
                                           os.path.join(self.data_path, "VoDaSuRe/ome/train/Femur_21_80kV_ome.zarr"),
                                           os.path.join(self.data_path, "VoDaSuRe/ome/train/Femur_74_80kV_ome.zarr"),
                                           os.path.join(self.data_path, "VoDaSuRe/ome/test/Femur_01_80kV_ome.zarr"),
                                           os.path.join(self.data_path, "VoDaSuRe/ome/train/bone_1_ome.zarr"),
                                           os.path.join(self.data_path, "VoDaSuRe/ome/test/bone_2_ome.zarr")]

                test_paths["VoDaSuRe"] = [os.path.join(self.data_path, "VoDaSuRe/ome/train/Vertebrae_A_80kV_ome.zarr"),
                                          os.path.join(self.data_path, "VoDaSuRe/ome/train/Vertebrae_B_80kV_ome.zarr"),
                                          os.path.join(self.data_path, "VoDaSuRe/ome/train/Vertebrae_C_80kV_ome.zarr"),
                                          os.path.join(self.data_path, "VoDaSuRe/ome/test/Vertebrae_D_80kV_ome.zarr"),
                                          os.path.join(self.data_path, "VoDaSuRe/ome/test/Ox_bone_A_bin1x1_ome_0.zarr")]

                print("Running domain generalization test on bone samples!")
                print("Train paths: ", train_paths["VoDaSuRe"])
                print("Test paths: ", test_paths["VoDaSuRe"])

            if "single_sample_test" in opt:
                train_paths["VoDaSuRe"] = [os.path.join(self.data_path, "VoDaSuRe/ome/train/Elm_A_bin1x1_ome_1.zarr")]

                test_paths["VoDaSuRe"] = [os.path.join(self.data_path, "VoDaSuRe/ome/test/Elm_A_bin1x1_ome_0.zarr")]

                print("Running single sample test on: Elm_A_bin1x1_ome_0.zarr")
                print("Train paths: ", train_paths["VoDaSuRe"])
                print("Test paths: ", test_paths["VoDaSuRe"])

            if "test_registration" in opt:
                train_paths["VoDaSuRe"] = [os.path.join(self.data_path, "VoDaSuRe/ome/train/Elm_A_bin1x1_shifted_ome_1.zarr")]

                test_paths["VoDaSuRe"] = [os.path.join(self.data_path, "VoDaSuRe/ome/test/Elm_A_bin1x1_shifted_ome_0.zarr")]

                print("Running single sample test on: Elm_A_bin1x1_shifted_ome_0.zarr")
                print("Train paths: ", train_paths["VoDaSuRe"])
                print("Test paths: ", test_paths["VoDaSuRe"])

            if "ablation_downsampling_test" in opt:
                print("Running ablation downsample test")
                if self.synthetic:
                    group_pairs["VoDaSuRe"] = {"4": [{"H": "HR/1", "L": "HR/3"}], "2": [{"H": "HR/1", "L": "HR/2"}]}
                else:
                    group_pairs["VoDaSuRe"] = {"4": [{"H": "HR/1", "L": "REG/1"}], "2": [{"H": "HR/2", "L": "REG/1"}]}
                print("Group pairs for VoDaSuRe: ", group_pairs["VoDaSuRe"])
            else:
                if self.synthetic:
                    group_pairs["VoDaSuRe"] = {"4": [{"H": "HR/0", "L": "HR/2"}], "2": [{"H": "HR/0", "L": "HR/1"}]}
                else:
                    group_pairs["VoDaSuRe"] = {"4": [{"H": "HR/0", "L": "REG/0"}], "2": [{"H": "HR/1", "L": "REG/0"}]}

        self.dataset_dict_train = {}
        self.dataset_dict_test = {}

        for dataset in train_paths:
            self.dataset_dict_train[dataset] = self.create_dataset_dict(train_paths[dataset],
                                                                        group_pairs[dataset],
                                                                        sampling_weight=sampling_weights[dataset],
                                                                        store_type=store_type[dataset])

            self.dataset_dict_test[dataset] = self.create_dataset_dict(test_paths[dataset],
                                                                       group_pairs[dataset],
                                                                       sampling_weight=sampling_weights[dataset],
                                                                       store_type=store_type[dataset])

            if opt['rank'] == 0:
                print(f"Number of training paths in {dataset}: {len(train_paths[dataset])}")
                print(f"Number of test paths in {dataset}: {len(test_paths[dataset])}")
                print(f"Sampling weight for {dataset} is {sampling_weights[dataset]}")
                print(f"Store type for {dataset} is {store_type[dataset]}")

    def create_dataset_dict(self, paths, group_pairs, sampling_weight=1.0, store_type="LocalStore"):

        d = {
            "paths": paths,
            "group_pairs": group_pairs,
            "sampling_weight": sampling_weight,
            "store_type": store_type
        }
        return d

    def get_dataset_dicts(self):

        return self.dataset_dict_train, self.dataset_dict_test

    def get_transforms(self, mode="train", baseline=False):

        # parse the options
        p = self.opt
        pdata = p.dataset_opt
        pdata.pad_size = get_context_pad_size(p)

        trans_list = []
        trans_list.append(mt.EnsureChannelFirstd(keys=["H", "L"], channel_dim=pdata.channel_dim))  # Load the image
        trans_list.append(mt.CastToTyped(keys=["H", "L"], dtype=torch.float32))  # Cast to float32
        trans_list.append(mt.ScaleIntensityRanged(keys=["H", "L"], a_min=0, a_max=65535, b_min=0.0, b_max=1.0, clip=True))  # Scale to [0, 1]
        trans_list.append(mt.SignalFillEmptyd(keys=["H", "L"], replacement=0))  # Remove any NaNs

        # Normalization and scaling
        if pdata.norm_type == "scale_intensity":
            pass
        elif pdata.norm_type == "znormalization":
            pass

        # Augmentations after crop
        if mode == "train":
            # Random augmentations
            #trans_list.append(RandSRCLAHEd(keys=["H", "L"], prob=0.5, clip_limit_range=(0.005, 0.02)))
            trans_list.append(RandSRContrastd(keys=["H", "L"], prob=0.5, gamma_range=(0.8, 1.2)))
            trans_list.append(RandSRFlipd(keys=["H", "L"], spatial_axis=0, prob=0.5))
            trans_list.append(RandSRFlipd(keys=["H", "L"], spatial_axis=1, prob=0.5))
            trans_list.append(RandSRFlipd(keys=["H", "L"], spatial_axis=2, prob=0.5))

            trans_list.append(RandSRRotated(keys=["H", "L"], prob=0.25, range_x=(-np.pi / 6, np.pi / 6), range_y=(-np.pi / 6, np.pi / 6), range_z=(-np.pi / 6, np.pi / 6), mode="bilinear", align_corners=True, keep_size=True))
            trans_list.append(RandSRZoomd(keys=["H", "L"], prob=0.25, min_zoom=0.9, max_zoom=1.1, mode="bilinear", align_corners=True, keep_size=True))

            # trans_list.append(mt.Rand3DElasticd(keys=["H", "L"], prob=0.80, sigma_range=(4, 8), magnitude_range=(-0.1, 0.1), mode="bilinear"))
            # trans_list.append(mt.RandGaussianNoised(keys=["L"], prob=0.2, mean=0.0, std=0.005))

        return mt.Compose(trans_list)
