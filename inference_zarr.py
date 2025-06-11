import datetime
import os
import time

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
from utils.utils_3D_image import run_strided_inference_zarr
from utils.load_options import load_options_from_experiment_id


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

    # zarr_path = "../PyHPC/ome_array_pyramid.zarr"
    # out_path = "../PyHPC/ome_array_pyramid_inference.zarr"

    zarr_path = "/work3/s173944/Python/venv_srgan/3D_datasets/datasets/danmax/bone_2_ome.zarr"
    out_path = "/dtu/3d-imaging-center/projects/2024_DANFIX_130_ExtremeCT/analysis/binning/bone_2_ome_super.zarr"

    run_strided_inference_zarr(
        model=model,
        zarr_path=zarr_path,
        out_path=out_path,
        group_name="HR",
        level_L='2',
        level_H='0',
        f=4,
        size_lr=32,
        border=4,
        batch_size=18,
        overlap_mode="hann"
    )

if __name__ == "__main__":
    main()