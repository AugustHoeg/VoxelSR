import datetime
import os
import shutil
import time
import zarr

import hydra
import numpy as np
import scipy.stats as stats
import torch
from PIL import Image
from omegaconf import DictConfig
from omegaconf import OmegaConf
import lpips
import matplotlib.pyplot as plt
from zarr.core.buffer import NDArrayLike

import config
from utils import utils_3D_image
from utils.utils_image import calculate_psnr_2D, calculate_ssim_2D, calculate_nrmse_2D
from utils.utils_3D_image import run_strided_inference_zarr, run_strided_inference, run_strided_inference_pad
from utils.load_options import load_options_from_experiment_id


from inference_zarr_V2 import create_metric_file, write_metric_statistics


def save_image_retry(img, path, retries=5, delay=0.3):
    """Save a PIL image, retrying on transient OSError.

    The output tree lives inside a OneDrive-synced folder, whose Files-On-Demand
    filter driver intermittently fails the open() with
    `OSError: [Errno 22] Invalid argument` while it is syncing/hydrating. The write
    succeeds on retry, so back off briefly and try again instead of crashing the job.
    """
    for attempt in range(retries):
        try:
            img.save(path)
            return
        except OSError as e:
            if attempt == retries - 1:
                raise
            print(f"  save failed ({e}); retrying {attempt + 1}/{retries - 1} -> {path}")
            time.sleep(delay * (attempt + 1))


@hydra.main(version_base=None, config_path="options", config_name=config.MODEL_ARCHITECTURE)
def main(opt: DictConfig):

    datasets_flag = False
    synthetic_flag = False

    # Set dataset override options from command line arguments
    if 'dataset_override' in opt['dataset_opt']:
        datasets_flag = opt['dataset_opt']['dataset_override']
    if 'synthetic_override' in opt['dataset_opt']:
        synthetic_flag = opt['dataset_opt']['synthetic_override']
    override_datasets = opt['dataset_opt']['datasets']
    override_synthetic = opt['dataset_opt']['synthetic']

    # Load options file from experiment ID
    experiment_id = opt['experiment_id']
    print("Experiment ID:", experiment_id)

    # REMOVE THIS LINE
    experiment_id = "RMSVQVAE3D_VoDaSuRe2_64_VoDaSuRe_OME_ID000055"

    opt_path = load_options_from_experiment_id(experiment_id, root_dir=config.ROOT_DIR, file_type="yaml")
    opt = OmegaConf.load(opt_path)
    wandb_path = opt_path.rsplit("files", 1)[0]

    # REMOVE THIS LINE
    # opt['dataset_opt']['datasets'] = ["VoDaSuRe"]

    # Override datasets if specified
    if datasets_flag:
        opt['dataset_opt']['datasets'] = override_datasets
        print(f"Using datasets {override_datasets} from command line argument.")
    else:
        print(f"Using datasets {opt['dataset_opt']['datasets']} from config file.")

    # Override synthetic flag if specified
    if synthetic_flag:
        opt['dataset_opt']['synthetic'] = override_synthetic
        print(f"Using synthetic flag: {override_synthetic} from command line argument.")

    # Set input type to 3D if not specified
    if 'input_type' not in opt:
        opt['input_type'] = '3D'

    print("Cuda is available", torch.cuda.is_available())
    print("Cuda device count", torch.cuda.device_count())
    print("Cuda current device", torch.cuda.current_device())
    print("Cuda device name", torch.cuda.get_device_name(0))

    # Set inference mode: 'zarr' or 'in_memory'
    inference_mode = 'in_memory'  # 'zarr' or 'in_memory'
    print(f"Running inference with mode: {inference_mode}")

    # Disable torch.compile
    opt['compile'] = False

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    from models.select_model import define_Model
    model = define_Model(opt, mode='test', data_parallel=False)  # currently supports only 1 GPU
    model.init_test(experiment_id)

    # Metrics to calculate
    # metric_names = ["psnr", "ssim", "lpips", "fid", "frc"]
    metric_names = ["psnr", "ssim", "lpips", "fid", "maniqa", "clipiqa", "musiq", "dists", "niqe", "frc"]
    print("Evaluating metrics:", metric_names)

    mask_zero_slices = False
    print(f"Mask zero slices: {mask_zero_slices}")

    # For 2D models, upscales batch_size_2D slices at a time, or None = predict to predict all slices at once.
    batch_size_2D = 16 if opt['input_type'] == '2D' else None
    print(f"2D slice batch size: {batch_size_2D}")
    slice_step = 1 if opt['input_type'] == '3D' else opt['up_factor']

    # Define dataset
    from data.Dataset_VoDaSuRe_OME import Dataset_VoDaSuRe_OME as D
    dataset = D(opt)
    data_dict = dataset.dataset_dict_test

    if opt['model_opt']['netG']['net_type'] == "MTVNet":
        patch_size = opt['dataset_opt']['patch_size']
        center_size = opt['model_opt']['netG']['context_sizes'][-1]  # New
        context_width = (patch_size - center_size) // 2
        patch_size_hr = center_size * opt['up_factor']
    elif opt['model_opt']['netG']['net_type'] == "AESOP3D":
        opt['up_factor'] = 1  # Force up_factor of 1 for AutoEncoder
        context_width = 0
        patch_size_hr = opt['dataset_opt']['patch_size']
    elif opt['model_opt']['model'] == "vqvae" or opt['model_opt']['model'] == "vqgan":
        opt['up_factor'] = 1
        context_width = 0
        opt['dataset_opt']['patch_size'] = opt['dataset_opt']['patch_size_hr']
        patch_size_hr = opt['dataset_opt']['patch_size']
        slice_step = 1 if opt['input_type'] == '3D' else opt['up_factor']
    else:
        context_width = 0
        patch_size_hr = opt['dataset_opt']['patch_size_hr']

    from utils.utils_3D_image import SliceMetrics3D
    slice_metrics = SliceMetrics3D(slice_dim=0, slice_step=slice_step, slice_max_val=65535.0, metric_names=metric_names, device=model.device)

    if hasattr(model, "latent_shape_hr"):  #
        size_hr = opt['dataset_opt']['patch_size_hr']
        model.latent_shape_hr = (size_hr // 8, size_hr // 8, size_hr // 8)

    # Create directory for test patch comparisons
    image_path = os.path.join(wandb_path, "files/", "media/", "images/")
    print("Saving image comparisons to:", image_path)
    if not os.path.exists(image_path + "patch_comparisons/"):
        os.makedirs(image_path + "patch_comparisons/")
    if not os.path.exists(image_path + "full_slice_comparisons/"):
        os.makedirs(image_path + "full_slice_comparisons/")

    batch_size = opt.dataset_opt.test_dataloader_params.dataloader_batch_size
    if opt['input_type'] == '2D' and batch_size > 1:
        batch_size = 1  # Force batch size of 1 for 2D models

    for name, dataset in data_dict.items():
        print(f"Dataset name: {name}")

        paths = dataset['paths']
        group_pairs = dataset['group_pairs']

        # Create metrics file
        # metric_file_path = create_metric_file(wandb_path, opt, dataset_name=name)

        sample_vals = {}
        sample_means = {}
        for metric_name in metric_names:
            if metric_name == "fid":
                continue
            sample_vals[metric_name] = []
            sample_means[metric_name] = []
        sample_names = [os.path.basename(path) for path in paths]

        for group_idx, group_pair in enumerate(group_pairs[f"{opt['up_factor']}"]):
            group_text = group_pair['H'].replace("/", "") + "_" + group_pair['L'].replace("/", "")

            #if "HR0" not in group_text:
            #    continue  # skip group pairs that do not contain HR0
            print(f"Group pair: {group_pair}")

            for image_idx, zarr_path in enumerate(paths):
                print(f"Processing image {image_idx + 1}/{len(paths)}: {zarr_path}")

                # Open LR zarr and convert to numpy array
                z = zarr.open(zarr_path, mode='r')
                img_L = z[group_pair["L"]]
                img_H = z[group_pair["H"]]

                img_L = np.array(img_L)
                img_L = np.reshape(img_L, (1, *img_L.shape))
                img_L = torch.from_numpy(img_L)


                # Crop center
                D, H, W = img_H.shape
                size_crop = 256
                img_H = img_H[
                    (D - size_crop) // 2 : (D + size_crop) // 2,
                    (H - size_crop) // 2 : (H + size_crop) // 2,
                    (W - size_crop) // 2 : (W + size_crop) // 2
                ]

                img_L = img_L[:,
                    (D - size_crop) // 2 : (D + size_crop) // 2,
                    (H - size_crop) // 2 : (H + size_crop) // 2,
                    (W - size_crop) // 2 : (W + size_crop) // 2,
                ]

                comp_path = os.path.join(image_path, "scale_depth_comparisons")
                axis = 0

                scales = [3, 4, 5, 6] # opt['model_opt']['netG']['v_patch_nums'] = [1, 2, 3, 4, 5, 6, 8]
                depths = opt["model_opt"]["netG"]['n_rq_depth']
                for scale_idx in scales:
                    for code_idx in range(depths):
                        print(f"scale_idx: {scale_idx}, code_idx: {code_idx}")
                        img_E = run_strided_inference_pad(
                            model=model,
                            img_L=img_L,
                            f=opt["up_factor"],
                            size_lr=opt.dataset_opt.patch_size,
                            size_hr=patch_size_hr,
                            border=4 + context_width * 2,
                            context_width=context_width,
                            batch_size=batch_size,
                            overlap_mode="hann",
                            model_input_type=opt["input_type"],
                            unnorm=opt["dataset_opt"]["norm_type"] == "znormalization",
                            input_rgb=opt["dataset_opt"]["expand_channels_rgb"],
                            batch_size_2D=batch_size_2D,
                            scale_idx=scale_idx,
                            code_idx=code_idx,
                        )
                        slice_idx_list = np.linspace(img_H.shape[axis] // 4, img_H.shape[axis] - img_H.shape[axis] // 4, 3)
                        for slice_idx in slice_idx_list:
                            slice_idx = int(slice_idx)
                            img_H_slice = Image.fromarray(img_H[slice_idx, :, :])
                            img_E_slice = Image.fromarray(img_E[0, slice_idx, :, :])

                            os.makedirs(os.path.join(comp_path, name, group_text), exist_ok=True)
                            file_name_E = f"{name}/{group_text}/image_{image_idx}_{slice_idx}_{opt['model_opt']['model_architecture']}_scale_{scale_idx}_code_{code_idx}.png"
                            save_image_retry(img_E_slice, os.path.join(comp_path, file_name_E))

                            file_name_H = f"{name}/{group_text}/image_{image_idx}_{slice_idx}.png"
                            path_H = os.path.join(comp_path, file_name_H)
                            if not os.path.exists(path_H):
                                save_image_retry(img_H_slice, path_H)


if __name__ == "__main__":
    main()