import os
import torch
import glob
import numpy as np
from data.train_transforms import BasicSRTransforms, GlobalScaleIntensityd
import monai.transforms as mt
from data.train_transforms import GaussianblurImaged, KspaceTruncd, \
    RandomCropPairImplicitd, RandomCropUniform, RandomCropForeground, \
    RandomCropLabel, get_context_pad_size # CustomRand3DElasticd

class Dataset_VoDaSuRe_OME():
    def __init__(self, opt):

        self.synthetic = opt['dataset_opt']['synthetic']

        self.opt = opt
        self.patch_size_hr = opt['dataset_opt']['patch_size_hr']
        self.patch_size_lr = opt['dataset_opt']['patch_size']
        self.degradation_type = opt['dataset_opt']['degradation_type']

        if opt['run_type'] == "HOME PC":
            self.data_path = ""

            train_paths = {"HCP_1200": glob.glob("../Vedrana_master_project/3D_datasets/datasets/HCP_1200/ome/train/*.zarr"),
                           "IXI": glob.glob("../Vedrana_master_project/3D_datasets/datasets/IXI/ome/train/*.zarr")}

            test_paths = {"HCP_1200": glob.glob("../Vedrana_master_project/3D_datasets/datasets/HCP_1200/ome/test/*.zarr"),
                          "IXI": glob.glob("../Vedrana_master_project/3D_datasets/datasets/IXI/ome/test/*.zarr")}

            sampling_weights = {"HCP_1200": 1.0,
                                "IXI":      1.0}

            group_pairs = {"HCP_1200": [{"H": "HR/0", "L": "HR/2"}],
                           "IXI":      [{"H": "HR/0", "L": "HR/2"}]}

            store_type = {"HCP_1200": "DirectoryStore",
                          "IXI":      "DirectoryStore"}

        elif opt['cluster'] == "TITANS":
            self.data_path = ""
            raise NotImplementedError(f"Dataset VoDaSuRe not supported for run type: {opt['run_type']}")

        else:  # Default is opt['cluster'] = "DTU_HPC"
            self.data_path = ""

            basedir_work2 = "/work2/aulho/"
            basedir_3dic = "/dtu/3d-imaging-center/projects/2025_DANFIX_163_VoDaSuRe/raw_data_extern/stitched/processed/"

            VoDaSuRe_train_paths = [os.path.join(basedir_work2, "Vertebrae_A_80kV/processed/*.zarr"),
                                    os.path.join(basedir_work2, "Vertebrae_B_80kV/processed/*.zarr"),
                                    os.path.join(basedir_work2, "Vertebrae_C_80kV/processed/*.zarr"),
                                    os.path.join(basedir_work2, "Femur_01_80kV/processed/*.zarr"),
                                    os.path.join(basedir_work2, "Femur_15_80kV/processed/*.zarr"),
                                    os.path.join(basedir_work2, "Femur_21_80kV/processed/*.zarr"),
                                    os.path.join(basedir_work2, "processed/Oak_A_bin1x1/*ome_1.zarr"),
                                    os.path.join(basedir_work2, "processed/Larch_B_bin1x1/*ome_1.zarr"),
                                    os.path.join(basedir_3dic,  "Bamboo_A_bin1x1/*ome_1.zarr"),
                                    os.path.join(basedir_3dic,  "Cardboard_A_bin1x1/*ome_1.zarr"),
                                    os.path.join(basedir_3dic,  "Cypress_A_bin1x1/*ome_1.zarr"),
                                    os.path.join(basedir_3dic,  "Elm_A_bin1x1/*ome_1.zarr"),
                                    os.path.join(basedir_3dic,  "MDF_A_bin1x1/*ome_1.zarr"),
                                    os.path.join(basedir_3dic,  "Ox_bone_A_bin1x1/*ome_1.zarr")]

            VoDaSuRe_test_paths = [os.path.join(basedir_work2, "Vertebrae_D_80kV/processed/*.zarr"),
                                   os.path.join(basedir_work2, "Femur_74_80kV/processed/*.zarr"),
                                   os.path.join(basedir_work2, "processed/Oak_A_bin1x1/*ome_0.zarr"),
                                   os.path.join(basedir_work2, "processed/Larch_B_bin1x1/*ome_0.zarr"),
                                   os.path.join(basedir_3dic,  "Bamboo_A_bin1x1/*ome_0.zarr"),
                                   os.path.join(basedir_3dic,  "Cardboard_A_bin1x1/*ome_0.zarr"),
                                   os.path.join(basedir_3dic,  "Cypress_A_bin1x1/*ome_0.zarr"),
                                   os.path.join(basedir_3dic,  "Elm_A_bin1x1/*ome_0.zarr"),
                                   os.path.join(basedir_3dic,  "MDF_A_bin1x1/*ome_0.zarr"),
                                   os.path.join(basedir_3dic,  "Ox_bone_A_bin1x1/*ome_0.zarr")]

            train_paths = {"HCP_1200":  glob.glob("/work2/aulho/HCP_1200/ome/train/*.zarr"),
                           "IXI":       glob.glob("/work2/aulho/IXI/ome/train/*.zarr"),
                           "LITS":      glob.glob("/work2/aulho/LITS/ome/train/*.zarr"),
                           "CTSpine1K": glob.glob("/work2/aulho/CTSpine1K/ome/train/*.zarr"),
                           "LIDC-IDRI": glob.glob("/work2/aulho/LIDC_IDRI/ome/train/*.zarr"),
                           "VoDaSuRe":  VoDaSuRe_train_paths}

            test_paths = {"HCP_1200":   glob.glob("/work2/aulho/HCP_1200/ome/test/*.zarr"),
                          "IXI":        glob.glob("/work2/aulho/IXI/ome/test/*.zarr"),
                          "LITS":       glob.glob("/work2/aulho/LITS/ome/test/*.zarr"),
                          "CTSpine1K":  glob.glob("/work2/aulho/CTSpine1K/ome/test/*.zarr"),
                          "LIDC-IDRI":  glob.glob("/work2/aulho/LIDC_IDRI/ome/test/*.zarr"),
                          "VoDaSuRe":   VoDaSuRe_test_paths}

            sampling_weights = {"HCP_1200":  1.0,
                                "IXI":       1.0,
                                "LITS":      1.0,
                                "CTSpine1K": 1.0,
                                "LIDC-IDRI": 1.0,
                                "VoDaSuRe":  5.0}

            group_pairs = {"HCP_1200":  [{"H": "HR/0", "L": "HR/2"}, {"H": "HR/1", "L": "HR/3"}],
                           "IXI":       [{"H": "HR/0", "L": "HR/2"}, {"H": "HR/1", "L": "HR/3"}],
                           "LITS":      [{"H": "HR/0", "L": "HR/2"}, {"H": "HR/1", "L": "HR/3"}],
                           "CTSpine1K": [{"H": "HR/0", "L": "HR/2"}, {"H": "HR/1", "L": "HR/3"}],
                           "LIDC-IDRI": [{"H": "HR/0", "L": "HR/2"}, {"H": "HR/1", "L": "HR/3"}],
                           "VoDaSuRe":  [{"H": "HR/0", "L": "HR/2"}, {"H": "HR/1", "L": "HR/3"}, {"H": "HR/0", "L": "REG/0"}, {"H": "HR/1", "L": "REG/1"}]}

            store_type = {"HCP_1200":  "DirectoryStore",
                          "IXI":       "DirectoryStore",
                          "LITS":      "DirectoryStore",
                          "CTSpine1K": "DirectoryStore",
                          "LIDC-IDRI": "DirectoryStore",
                          "VoDaSuRe":  "DirectoryStore"}


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

            print("Dataset dict train:")
            print(self.dataset_dict_train[dataset])
            print("Dataset dict test:")
            print(self.dataset_dict_test[dataset])
            exit(0)

    def create_dataset_dict(self, paths, group_pairs, sampling_weight=1.0, store_type="DirectoryStore"):

        d = {
            "paths": paths,
            "group_pairs": {
                "4": group_pairs,  # {"H": "HR/1", "L": "HR/3"}
            },
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
        # trans_list.append(mt.CastToTyped(keys=["H", "L"], dtype=torch.float32))  # Cast to float32
        trans_list.append(mt.SignalFillEmptyd(keys=["H", "L"], replacement=0))

        # Normalization and scaling
        if pdata.norm_type == "scale_intensity":
            pass
            #trans_list.append(GlobalScaleIntensityd(keys=["H"], global_min=-0.000936, global_max=0.000998))
            #trans_list.append(GlobalScaleIntensityd(keys=["L"], global_min=-0.002063, global_max=0.002476))
        elif pdata.norm_type == "znormalization":
            pass
            #trans_list.append(mt.NormalizeIntensityd(keys=["H", "L"]))

        # Pad for MTVNet
        trans_list.append(mt.BorderPadd(keys=["L"], spatial_border=[pdata.pad_size, pdata.pad_size, pdata.pad_size], mode='constant'))

        # Augmentations after crop
        if mode == "train":
            # Random augmentations
            # trans_list.append(mt.RandFlipd(keys=["H", "L"], prob=0.20, spatial_axis=[0, 1, 2]))  # Random flip
            trans_list.append(mt.RandFlipd(keys=["H", "L"], spatial_axis=0, prob=0.5))
            trans_list.append(mt.RandFlipd(keys=["H", "L"], spatial_axis=1, prob=0.5))
            trans_list.append(mt.RandFlipd(keys=["H", "L"], spatial_axis=2, prob=0.5))
            trans_list.append(mt.RandRotated(keys=["H", "L"], prob=0.50, range_x=(-np.pi / 6, np.pi / 6), range_y=(-np.pi / 6, np.pi / 6), range_z=(-np.pi / 6, np.pi / 6), mode="bilinear", align_corners=True, keep_size=True))
            # trans_list.append(mt.Rand3DElasticd(keys=["H", "L"], prob=0.80, sigma_range=(4, 8), magnitude_range=(-0.1, 0.1), mode="bilinear"))
            # trans_list.append(CustomRand3DElasticd(keys=["H", "L"], prob=0.80, sigma_range=(4, 8), magnitude_range=(-0.1, 0.1), mode="bilinear"))
            trans_list.append(mt.RandZoomd(keys=["H", "L"], prob=0.50, min_zoom=0.9, max_zoom=1.1, mode="bilinear", align_corners=True, keep_size=True))
            # trans_list.append(mt.RandGaussianNoised(keys=["L"], prob=0.2, mean=0.0, std=0.005))

        return mt.Compose(trans_list)
