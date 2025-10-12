import os
import torch
import glob
import numpy as np
from data.train_transforms import BasicSRTransforms, GlobalScaleIntensityd, RandSRFlipd, RandSRRotated, RandSRZoomd, RandSRContrastd, RandSRCLAHEd
import monai.transforms as mt
from data.train_transforms import GaussianblurImaged, KspaceTruncd, \
    RandomCropPairImplicitd, RandomCropUniform, RandomCropForeground, \
    RandomCropLabel, get_context_pad_size # CustomRand3DElasticd

class Dataset_VoDaSuRe_OME():
    def __init__(self, opt):

        self.synthetic = opt['dataset_opt']['synthetic']
        print(f"Using synthetically downsampled LR images: {self.synthetic}")

        self.opt = opt
        self.patch_size_hr = opt['dataset_opt']['patch_size_hr']
        self.patch_size_lr = opt['dataset_opt']['patch_size']
        self.degradation_type = opt['dataset_opt']['degradation_type']

        print(f"Using datasets: {opt['dataset_opt']['datasets']} on {opt['run_type']}")

        if opt['run_type'] == "HOME PC":
            self.data_path = "../Vedrana_master_project/3D_datasets/datasets/"

            train_paths = {}
            test_paths = {}
            if "HCP_1200" in opt['dataset_opt']['datasets']:
                train_paths["HCP_1200"] = glob.glob(os.path.join(self.data_path, "HCP_1200/ome/train/*.zarr"))
                test_paths["HCP_1200"] = glob.glob(os.path.join(self.data_path, "HCP_1200/ome/test/*.zarr"))

            if "IXI" in opt['dataset_opt']['datasets']:
                train_paths["IXI"] = glob.glob(os.path.join(self.data_path, "IXI/ome/train/*.zarr"))
                test_paths["IXI"] = glob.glob(os.path.join(self.data_path, "IXI/ome/test/*.zarr"))

            if "VoDaSuRe" in opt['dataset_opt']['datasets']:
                train_paths["VoDaSuRe"] = glob.glob(os.path.join(self.data_path, "VoDaSuRe/ome/train/*.zarr"))
                test_paths["VoDaSuRe"] = glob.glob(os.path.join(self.data_path, "VoDaSuRe/ome/test/*.zarr"))

            sampling_weights = {"HCP_1200": 1.0,
                                "IXI":      1.0,
                                "VoDaSuRe": 1.0}

            group_pairs = {}
            group_pairs["HCP_1200"] = {"4": [{"H": "HR/0", "L": "HR/2"}], "2": [{"H": "HR/0", "L": "HR/1"}]}
            group_pairs["IXI"] = {"4": [{"H": "HR/0", "L": "HR/2"}], "2": [{"H": "HR/0", "L": "HR/1"}]}

            if self.synthetic:
                group_pairs["VoDaSuRe"] = {"4": [{"H": "HR/0", "L": "HR/2"}], "2": [{"H": "HR/0", "L": "HR/1"}]}
            else:
                group_pairs["VoDaSuRe"] = {"4": [{"H": "HR/0", "L": "REG/0"}], "2": [{"H": "HR/1", "L": "REG/0"}]}

            store_type = {"HCP_1200": "LocalStore",
                          "IXI":      "LocalStore",
                          "VoDaSuRe": "LocalStore"}

        elif opt['cluster'] == "TITANS":
            self.data_path = ""
            raise NotImplementedError(f"Dataset VoDaSuRe not supported for run type: {opt['run_type']}")

        else:  # Default is opt['cluster'] = "DTU_HPC"
            self.data_path = "../3D_datasets/datasets/"

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

            sampling_weights = {"HCP_1200":  3.0,
                                "IXI":       1.0,
                                "LITS":      2.0,
                                "CTSpine1K": 5.0,
                                "LIDC-IDRI": 5.0,
                                "VoDaSuRe":  15.0}


            # group_pairs = {
            #     "HCP_1200": {
            #         "4": [{"H": "HR/0", "L": "HR/2"}],
            #         "2": [{"H": "HR/0", "L": "HR/1"}, {"H": "HR/1", "L": "HR/2"}]
            #     },
            #     "IXI": {
            #         "4": [{"H": "HR/0", "L": "HR/2"}],
            #         "2": [{"H": "HR/0", "L": "HR/1"}, {"H": "HR/1", "L": "HR/2"}]
            #     },
            #     "LITS": {
            #         "4": [{"H": "HR/0", "L": "HR/2"}, {"H": "HR/1", "L": "HR/3"}],
            #         "2": [{"H": "HR/0", "L": "HR/1"}, {"H": "HR/1", "L": "HR/2"}]
            #     },
            #     "CTSpine1K": {
            #         "4": [{"H": "HR/0", "L": "HR/2"}, {"H": "HR/1", "L": "HR/3"}],
            #         "2": [{"H": "HR/0", "L": "HR/1"}, {"H": "HR/1", "L": "HR/2"}]
            #     },
            #     "LIDC-IDRI": {
            #         "4": [{"H": "HR/0", "L": "HR/2"}, {"H": "HR/1", "L": "HR/3"}],
            #         "2": [{"H": "HR/0", "L": "HR/1"}, {"H": "HR/1", "L": "HR/2"}]
            #     },
            #     "VoDaSuRe": {
            #         "4": [{"H": "HR/0", "L": "HR/2"}, {"H": "HR/1", "L": "HR/3"}, {"H": "HR/0", "L": "REG/0"}],
            #         "2": [{"H": "HR/0", "L": "HR/1"}, {"H": "HR/1", "L": "HR/2"}, {"H": "HR/1", "L": "REG/0"}]
            #     },
            # }

            # group_pairs = {  # only synthetic
            #     "HCP_1200": {
            #         "4": [{"H": "HR/0", "L": "HR/2"}],
            #         "2": [{"H": "HR/0", "L": "HR/1"}]
            #     },
            #     "IXI": {
            #         "4": [{"H": "HR/0", "L": "HR/2"}],
            #         "2": [{"H": "HR/0", "L": "HR/1"}]
            #     },
            #     "LITS": {
            #         "4": [{"H": "HR/0", "L": "HR/2"}],
            #         "2": [{"H": "HR/0", "L": "HR/1"}]
            #     },
            #     "CTSpine1K": {
            #         "4": [{"H": "HR/0", "L": "HR/2"}],
            #         "2": [{"H": "HR/0", "L": "HR/1"}]
            #     },
            #     "LIDC-IDRI": {
            #         "4": [{"H": "HR/0", "L": "HR/2"}],
            #         "2": [{"H": "HR/0", "L": "HR/1"}]
            #     },
            #     "VoDaSuRe": {
            #         "4": [{"H": "HR/0", "L": "HR/2"}],
            #         "2": [{"H": "HR/0", "L": "HR/1"}]
            #     },
            # }

            group_pairs = {}
            group_pairs["HCP_1200"] = {"4": [{"H": "HR/0", "L": "HR/2"}], "2": [{"H": "HR/0", "L": "HR/1"}]}
            group_pairs["IXI"] = {"4": [{"H": "HR/0", "L": "HR/2"}], "2": [{"H": "HR/0", "L": "HR/1"}]}
            group_pairs["LITS"] = {"4": [{"H": "HR/0", "L": "HR/2"}], "2": [{"H": "HR/0", "L": "HR/1"}]}
            group_pairs["CTSpine1K"] = {"4": [{"H": "HR/0", "L": "HR/2"}], "2": [{"H": "HR/0", "L": "HR/1"}]}
            group_pairs["LIDC-IDRI"] = {"4": [{"H": "HR/0", "L": "HR/2"}], "2": [{"H": "HR/0", "L": "HR/1"}]}

            if self.synthetic:
                group_pairs["VoDaSuRe"] = {"4": [{"H": "HR/0", "L": "HR/2"}], "2": [{"H": "HR/0", "L": "HR/1"}]}
            else:
                group_pairs["VoDaSuRe"] = {"4": [{"H": "HR/0", "L": "REG/0"}], "2": [{"H": "HR/1", "L": "REG/0"}]}

            store_type = {"HCP_1200":  "LocalStore",
                          "IXI":       "LocalStore",
                          "LITS":      "LocalStore",
                          "CTSpine1K": "LocalStore",
                          "LIDC-IDRI": "LocalStore",
                          "VoDaSuRe":  "LocalStore"}

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
