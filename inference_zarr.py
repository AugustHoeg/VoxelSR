import datetime
import os
import shutil
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

def get_mean_and_ci(data_sequence, confidence=0.95):

    data = np.array(data_sequence)
    n = len(data)
    mean, se = np.mean(data), stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return mean, h

def create_metric_file(wandb_path, opt, dataset_name):

    # Create a file with performance metric statistics
    file_dir = os.path.join(wandb_path, "files/")
    if not os.path.exists(file_dir + "performance_statistics/"):
        os.makedirs(file_dir + "performance_statistics/")

    performance_statistics_dir = os.path.join(file_dir, "performance_statistics/")

    # Specify the file name
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"metrics_{dataset_name}_{opt['experiment_id']}_{current_time}.txt"
    file_path = performance_statistics_dir + file_name

    # Open the file in write mode and write the contents
    with open(file_path, 'w') as file:
        file.write("MODEL ARCHITECTURE: " + opt['model_opt']['model_architecture'] + "\n")
        file.write("DATASET: " + opt['dataset_opt']['name'] + " NAME: " + dataset_name + "\n")
        file.write("EXPERIMENT ID: " + opt['experiment_id'] + "\n")
        file.write("RUN NAME: " + opt['run_name'] + "\n")
        file.write("MAX ITERATIONS: " + str(opt['train_opt']['iterations']) + "\n")

        file.write("PATCH SIZE: " + str(opt['dataset_opt']['patch_size']) + "\n")
        file.write("UP FACTOR: " + str(opt['up_factor']) + "\n")
        file.write("LEARNING RATE: " + str(opt['train_opt']['G_optimizer_lr']) + "\n")

    print(f"File '{file_name}' has been created and saved.")

    return file_path


def write_metric_statistics(file_path, psnr_vals, ssim_vals, nrmse_vals, text=None):

    psnr_slice_mean, ci_psnr = get_mean_and_ci(psnr_vals['slice_vals'])
    ssim_slice_mean, ci_ssim = get_mean_and_ci(ssim_vals['slice_vals'])
    nrmse_slice_mean, ci_nrmse = get_mean_and_ci(nrmse_vals['slice_vals'])

    # Open the file in write mode and write the contents
    with open(file_path, 'a+') as file:
        if text is not None:
            file.write("\n" + "METRICS: " + text.upper() + "\n")

        file.write("SAMPLE PERFORMANCE METRICS \n")
        file.write("PSNR SAMPLE LIST: " + str(torch.tensor(psnr_vals['sample_means']).numpy()) + "\n")
        file.write("SSIM SAMPLE LIST: " + str(torch.tensor(ssim_vals['sample_means']).numpy()) + "\n")
        file.write("NRMSE SAMPLE LIST: " + str(torch.tensor(nrmse_vals['sample_means']).numpy()) + "\n")

        # Write the individual values to the file
        file.write("AVERAGE SLICE-WISE PERFORMANCE METRICS \n")
        file.write("AVERAGE SLICE-WISE PSNR: " + str(psnr_slice_mean) + "+-" + str(ci_psnr) + "\n")
        file.write("AVERAGE SLICE-WISE SSIM: " + str(ssim_slice_mean) + "+-" + str(ci_ssim) + "\n")
        file.write("AVERAGE SLICE-WISE NRSME: " + str(nrmse_slice_mean) + "+-" + str(ci_nrmse) + "\n")


def get_full_sample_metrics(img_H, img_E, slice_dim=0, slice_step=1):

    num_slices = img_H.shape[slice_dim]

    # Compute PSNR, SSIM and NRMSE slice-wise. Slice-wise approach is chosen as some dataset samples are very large.
    psnr_slice_list = []
    ssim_slice_list = []
    nrmse_slice_list = []

    for i in range(0, num_slices, slice_step):
        if slice_dim == 0:
            H_slice = img_H[i, :, :]
            E_slice = img_E[i, :, :]
        elif slice_dim == 1:
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
    model = define_Model(opt, mode='test')
    model.init_test(experiment_id)

    # Define dataset
    from data.Dataset_VoDaSuRe_OME import Dataset_VoDaSuRe_OME as D
    dataset = D(opt)
    data_dict = dataset.dataset_dict_test

    # Create directory for test patch comparisons
    image_path = os.path.join(wandb_path, "files/", "media/", "images/")
    print("Saving image comparisons to:", image_path)
    if not os.path.exists(image_path + "patch_comparisons/"):
        os.makedirs(image_path + "patch_comparisons/")
    if not os.path.exists(image_path + "full_slice_comparisons/"):
        os.makedirs(image_path + "full_slice_comparisons/")

    for name, dataset in data_dict.items():
        print(f"Dataset name: {name}")

        paths = dataset['paths']
        group_pairs = dataset['group_pairs']

        # Create metrics file
        metric_file_path = create_metric_file(wandb_path, opt, dataset_name=name)

        psnr_sample_means = []
        ssim_sample_means = []
        nrmse_sample_means = []

        for group_idx, group_pair in enumerate(group_pairs[f"{opt['up_factor']}"]):
            print(f"Group pair: {group_pair}")
            group_text = group_pair['H'].replace("/", "") + "_" + group_pair['L'].replace("/", "")

            if "HR0" not in group_text:
                continue  # skip group pairs that do not contain HR0

            # Create metric lists
            psnr_vals = {"sample_means": [], "slice_vals": []}
            ssim_vals = {"sample_means": [], "slice_vals": []}
            nrmse_vals = {"sample_means": [], "slice_vals": []}

            for image_idx, zarr_path in enumerate(paths):
                print(f"Processing image {image_idx + 1}/{len(paths)}: {zarr_path}")
                out_path = os.path.join(wandb_path, f"files/model_outputs/{os.path.basename(zarr_path)}")

                if "bone_2_cropped" in zarr_path:
                    print(f"Skipping very large bone sample: {zarr_path}. Please run inference on this sample separately if needed.")
                    continue  # skip the very large danmax bone sample

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

                zarr_H = zarr.open(zarr_path, mode='r')
                img_H = zarr_H[group_pair["H"]]
                img_L = zarr_H[group_pair["L"]]

                zarr_E = zarr.open(out_path, mode='r')
                img_E = zarr_E['SR/0']  # Always read the top level

                psnr_slice_list, ssim_slice_list, nrmse_slice_list = get_full_sample_metrics(img_H, img_E, slice_dim=0, slice_step=1)

                sample_psnr = np.mean(psnr_slice_list)
                sample_ssim = np.mean(ssim_slice_list)
                sample_nrmse = np.mean(nrmse_slice_list)
                print("Dimension %d, Sample PSNR: %0.4f, SSIM: %0.6f, NRMSE: %0.6f" % (0, sample_psnr, sample_ssim, sample_nrmse))

                psnr_vals['slice_vals'].extend(psnr_slice_list)
                ssim_vals['slice_vals'].extend(ssim_slice_list)
                nrmse_vals['slice_vals'].extend(nrmse_slice_list)

                psnr_vals['sample_means'].append(sample_psnr)
                ssim_vals['sample_means'].append(sample_ssim)
                nrmse_vals['sample_means'].append(sample_nrmse)

                for axis in range(3):
                    target_shape = list(img_H.shape)
                    del target_shape[axis]  # remove slice axis
                    target_shape.insert(0, 1)  # prepend 1
                    # Save full slice comparisons over whole sample
                    baseline_comparison_tool = utils_3D_image.ImageComparisonTool3D(
                        patch_size_hr=target_shape,
                        upscaling_methods=["tio_nearest"],  ## or tio_linear
                        unnorm=False,
                        div_max=False,
                        out_dtype=np.uint8,
                        upscale_slice=True)

                    img_dict = {'H': img_H, 'E': img_E, 'L': img_L}
                    comp_path = os.path.join(image_path, "full_slice_comparisons")

                    slice_idx_list = np.linspace(img_H.shape[axis] // 4, img_H.shape[axis] - img_H.shape[axis] // 4, 3)
                    for slice_idx in slice_idx_list:
                        grid_image = baseline_comparison_tool.get_comparison_image(img_dict, slice_idx=int(slice_idx), axis=axis)
                        grid_image = Image.fromarray(grid_image)

                        os.makedirs(os.path.join(comp_path, name, group_text), exist_ok=True)
                        file_name = f"{name}/{group_text}/image_{image_idx}_comp_axis_{axis}_{slice_idx}_{opt['model_opt']['model_architecture']}_{opt['up_factor']}x.png"
                        path = os.path.join(comp_path, file_name)
                        grid_image.save(path)

                # Delete SR zarr to save space
                if os.path.exists(out_path):
                    shutil.rmtree(out_path)

            # Save sample metrics
            psnr_sample_means.extend(psnr_vals['sample_means'])
            ssim_sample_means.extend(ssim_vals['sample_means'])
            nrmse_sample_means.extend(nrmse_vals['sample_means'])

            # Save group pair metrics
            write_metric_statistics(metric_file_path, psnr_vals, ssim_vals, nrmse_vals, text=group_text)

        # Average metrics across group pairs
        avg_psnr = np.mean(psnr_sample_means)
        avg_ssim = np.mean(ssim_sample_means)
        avg_nrmse = np.mean(nrmse_sample_means)
        print(f"Performance metrics for dataset {name}: Average PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.6f}, NRMSE: {avg_nrmse:.6f}")

        # Write final dataset metric averages
        with open(metric_file_path, 'a+') as file:
            file.write("\nDATASET SAMPLE AVERAGES\n")
            file.write("PSNR AVERAGE: " + str(avg_psnr) + "\n")
            file.write("SSIM AVERAGE: " + str(avg_ssim) + "\n")
            file.write("NRMSE AVERAGE: " + str(avg_nrmse) + "\n")


if __name__ == "__main__":
    main()