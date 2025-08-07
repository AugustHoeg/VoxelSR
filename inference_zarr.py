import datetime
import os
import time
import zarr

import hydra
import matplotlib.pyplot as plt
import monai
import numpy as np
import scipy.stats as stats
import torch
import torchio as tio
from PIL import Image
from omegaconf import DictConfig
from omegaconf import OmegaConf
from tqdm import tqdm

import config
from utils import utils_3D_image
from utils.utils_image import calculate_psnr_2D, calculate_ssim_2D, calculate_nrmse_2D
from utils.utils_3D_image import run_strided_inference_zarr
from utils.load_options import load_options_from_experiment_id

def get_full_sample_metrics(img_H, img_E, slice_dim=3, slice_step=1):

    num_slices = img_H.shape[slice_dim]

    # Compute PSNR, SSIM and NRMSE slice-wise. Slice-wise approach is chosen as some dataset samples are very large.
    psnr_slice_list = []
    ssim_slice_list = []
    nrmse_slice_list = []

    for i in range(0, num_slices, slice_step):
        if slice_dim == 1:
            H_slice = img_H[i, :, :]
            E_slice = img_E[i, :, :]
        elif slice_dim == 2:
            H_slice = img_H[:, i, :]
            E_slice = img_E[:, i, :]
        else:
            H_slice = img_H[:, :, i]
            E_slice = img_E[:, :, i]

        slice_psnr = calculate_psnr_2D(E_slice, H_slice, border=0)
        psnr_slice_list.append(slice_psnr)

        slice_ssim = calculate_ssim_2D(E_slice, H_slice, border=0)
        ssim_slice_list.append(slice_ssim)

        slice_nrmse = calculate_nrmse_2D(E_slice, H_slice, border=0)
        nrmse_slice_list.append(slice_nrmse)

    return psnr_slice_list, ssim_slice_list, nrmse_slice_list


@hydra.main(version_base=None, config_path="options", config_name=config.MODEL_ARCHITECTURE)
def main(opt: DictConfig):

    # Load options file from experiment ID
    experiment_id = opt['experiment_id']
    print("Experiment ID:", experiment_id)
    opt_path = load_options_from_experiment_id(experiment_id, root_dir=config.ROOT_DIR, file_type="yaml")
    opt = OmegaConf.load(opt_path)
    wandb_path = opt_path.rsplit("files", 1)[0]

    # Set input type to 3D if not specified
    if 'input_type' not in opt:
        opt['input_type'] = '3D'

    print("Cuda is available", torch.cuda.is_available())
    print("Cuda device count", torch.cuda.device_count())
    print("Cuda current device", torch.cuda.current_device())
    print("Cuda device name", torch.cuda.get_device_name(0))

    from models.select_model import define_Model
    model = define_Model(opt)
    model.init_test(experiment_id)

    zarr_path = "../Vedrana_master_project/3D_datasets/datasets/HCP_1200/ome/test/volume_826353.zarr"
    out_path = f"{wandb_path}/files/model_outputs/volume_826353.zarr"

    # zarr_path = "/work3/s173944/Python/venv_srgan/3D_datasets/datasets/danmax/bone_2_ome.zarr"
    # out_path = "/work2/aulho/bone_2_ome_super.zarr"

    # Define dataset
    from data.Dataset_VoDaSuRe_OME import Dataset_VoDaSuRe_OME as D
    dataset = D(opt)
    data_dict = dataset.dataset_dict_test

    for name, dataset in data_dict.items():
        print(f"Dataset name: {name}")
        paths = dataset['paths']
        group_pairs = dataset['group_pairs']

        group_pair = group_pairs[f"{opt['up_factor']}"][0]  # Assuming we want to use the first group pair for testing

    group_pair = {"H": "HR/0", "L": "HR/2"}

    run_strided_inference_zarr(
        model=model,
        zarr_path=zarr_path,
        out_path=out_path,
        group_pair=group_pair,
        f=opt['up_factor'],
        size_lr=opt.dataset_opt.patch_size,
        border=4,
        batch_size=opt.dataset_opt.train_dataloader_params.dataloader_batch_size,
        overlap_mode="hann"
    )

    level_L = int(group_pair["L"].split("/")[-1])
    level_H = int(group_pair["H"].split("/")[-1])

    zarr_E = zarr.open(out_path, mode='r')
    img_E = zarr_E[f"SR/{level_H}"]

    zarr_H = zarr.open(zarr_path, mode='r')
    img_H = zarr_H[f"HR/{level_H}"]

    psnr_slice_list, ssim_slice_list, nrmse_slice_list = get_full_sample_metrics(img_H, img_E, slice_dim=0, slice_step=1)

    sample_psnr = np.mean(psnr_slice_list)
    sample_ssim = np.mean(ssim_slice_list)
    sample_nrmse = np.mean(nrmse_slice_list)

    print(f"Sample PSNR: {sample_psnr:.2f}, SSIM: {sample_ssim:.4f}, NRMSE: {sample_nrmse:.4f}")

if __name__ == "__main__":
    main()