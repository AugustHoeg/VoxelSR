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


def _mask_zero_slices(img_src, img_ref):
    # Zero out slices in img_src where img_ref is zero.
    for i in range(0, img_ref.shape[0]):
        img_src[i, :, :] = np.where(img_ref[i, :, :], img_src[i, :, :], 0)


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


def write_metric_statistics(file_path, sample_vals, sample_means, sample_names, text=None):

    # Open the file in write mode and write the contents
    with open(file_path, 'a+') as file:
        if text is not None:
            file.write("\n" + "METRICS: " + text.upper() + "\n")
        file.write(f"SAMPLE PERFORMANCE METRICS \n")
        sample_names_str = ', '.join(str(x) for x in sample_names)
        file.write(f"SAMPLE NAMES: {sample_names_str}\n")

        for metric_name, metric_vals in sample_means.items():
            metric_vals_str = f', '.join(str(x.round(6)) for x in metric_vals)
            file.write(f"{metric_name.upper()} SAMPLE LIST: {metric_vals_str}\n")

        # Write the individual values to the file
        file.write("AVERAGE SLICE-WISE PERFORMANCE METRICS \n")

        for metric_name, metric_vals in sample_vals.items():
            mean, ci = get_mean_and_ci(sample_vals[metric_name])
            mean_str = str(mean.round(6))
            ci_str = f', '.join(str(x.round(6)) for x in ci)
            file.write(f"AVERAGE SLICE-WISE {metric_name.upper()}: {mean_str} +- {ci_str} \n")


def get_full_sample_metrics(img_H, img_E, slice_dim=0, slice_step=1, eps=1e-9, lpips_model=None, device='cuda'):

    num_slices = img_H.shape[slice_dim]

    # Compute PSNR, SSIM and NRMSE slice-wise. Slice-wise approach is chosen as some dataset samples are very large.
    psnr_slice_list = []
    ssim_slice_list = []
    nrmse_slice_list = []
    lpips_slice_list = []

    for i in range(0, num_slices, slice_step):
        if i % 100 == 0:
            print(f"Evaluating slice {i}/{num_slices}")

        if slice_dim == 0:
            H_slice = img_H[i, :, :]
            E_slice = img_E[i, :, :]
        elif slice_dim == 1:
            H_slice = img_H[:, i, :]
            E_slice = img_E[:, i, :]
        else:
            H_slice = img_H[:, :, i]
            E_slice = img_E[:, :, i]

        # Normalize to [0, 1]
        E_min, E_max = E_slice.min(), E_slice.max()
        H_min, H_max = H_slice.min(), H_slice.max()

        if E_max - E_min > eps and H_max - H_min > eps:
            E_slice = (E_slice - E_slice.min()) / (E_slice.max() - E_slice.min())
            H_slice = (H_slice - H_slice.min()) / (H_slice.max() - H_slice.min())
        else:
            continue

        slice_psnr = calculate_psnr_2D(E_slice, H_slice, border=0)
        psnr_slice_list.append(slice_psnr)

        slice_ssim = calculate_ssim_2D(E_slice, H_slice, border=0)
        ssim_slice_list.append(slice_ssim)

        slice_nrmse = calculate_nrmse_2D(E_slice, H_slice, border=0)
        nrmse_slice_list.append(slice_nrmse)

        slice_lpips = -1
        if lpips_model is not None:
            E_slice = torch.from_numpy(E_slice).to(device)
            H_slice = torch.from_numpy(H_slice).to(device)
            slice_lpips = lpips_model(E_slice, H_slice).item()
        lpips_slice_list.append(slice_lpips)

    return psnr_slice_list, ssim_slice_list, nrmse_slice_list, lpips_slice_list

def get_full_sample_metrics_V2(img_H, img_E, slice_dim=0, slice_step=1, eps=1e-10, max_val=65535.0, lpips_model=None, device='cuda'):

    num_slices = img_H.shape[slice_dim]

    # Compute PSNR, SSIM and NRMSE slice-wise. Slice-wise approach is chosen as some dataset samples are very large.
    psnr_slice_list = []
    ssim_slice_list = []
    nrmse_slice_list = []
    lpips_slice_list = []

    for i in range(0, num_slices, slice_step):
        if i % 100 == 0:
            print(f"Evaluating slice {i}/{num_slices}")

        if slice_dim == 0:
            H_slice = img_H[i, :, :]
            E_slice = img_E[i, :, :]
        elif slice_dim == 1:
            H_slice = img_H[:, i, :]
            E_slice = img_E[:, i, :]
        else:
            H_slice = img_H[:, :, i]
            E_slice = img_E[:, :, i]

        # Normalize to [0, 1]
        H_slice = H_slice.astype(np.float32) / max_val
        E_slice = E_slice.astype(np.float32) / max_val

        H_min, H_max = H_slice.min(), H_slice.max()
        if H_max - H_min < eps:
            continue

        E_slice = np.clip(E_slice, 0.0, 1.0)
        H_slice = np.clip(H_slice, 0.0, 1.0)

        slice_psnr = calculate_psnr_2D(E_slice, H_slice, border=0)
        psnr_slice_list.append(slice_psnr)

        slice_ssim = calculate_ssim_2D(E_slice, H_slice, border=0)
        ssim_slice_list.append(slice_ssim)

        slice_nrmse = calculate_nrmse_2D(E_slice, H_slice, border=0)
        nrmse_slice_list.append(slice_nrmse)

        slice_lpips = -1
        if lpips_model is not None:
            E_slice = torch.from_numpy(E_slice).to(device)
            H_slice = torch.from_numpy(H_slice).to(device)
            slice_lpips = lpips_model(E_slice, H_slice).item()
        lpips_slice_list.append(slice_lpips)

    return psnr_slice_list, ssim_slice_list, nrmse_slice_list, lpips_slice_list

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
    # experiment_id = "mDCSRN_MRI_4x_VoDaSuRe_OME_ID004200"

    opt_path = load_options_from_experiment_id(experiment_id, root_dir=config.ROOT_DIR, file_type="yaml")
    opt = OmegaConf.load(opt_path)
    wandb_path = opt_path.rsplit("files", 1)[0]

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

    from models.select_model import define_Model
    model = define_Model(opt, mode='test', data_parallel=False)  # currently supports only 1 GPU
    model.init_test(experiment_id)

    # Metrics to calculate
    # metric_names = ["psnr", "ssim", "fid"]
    metric_names = ["psnr", "ssim", "lpips", "fid", "maniqa", "clipiqa", "musiq", "dists", "niqe"]
    print("Evaluating metrics:", metric_names)

    slice_step = 1 if opt['input_type'] == '3D' else opt['up_factor']
    from utils.utils_3D_image import SliceMetrics3D
    slice_metrics = SliceMetrics3D(slice_dim=0, slice_step=slice_step, slice_max_val=65535.0, metric_names=metric_names, device=model.device)

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
    else:
        context_width = 0
        patch_size_hr = opt['dataset_opt']['patch_size_hr']

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

    batch_size = 2 # opt.dataset_opt.train_dataloader_params.dataloader_batch_size
    if opt['input_type'] == '2D' and batch_size > 1:
        batch_size = 1  # Force batch size of 1 for 2D models

    for name, dataset in data_dict.items():
        print(f"Dataset name: {name}")

        paths = dataset['paths']
        group_pairs = dataset['group_pairs']

        # Create metrics file
        metric_file_path = create_metric_file(wandb_path, opt, dataset_name=name)

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
                out_path = os.path.join(wandb_path, f"files/model_outputs/{os.path.basename(zarr_path)}")

                if inference_mode == 'zarr':
                    run_strided_inference_zarr(
                        model=model,
                        zarr_path=zarr_path,
                        out_path=out_path,
                        group_pair=group_pair,
                        f=opt["up_factor"],
                        size_lr=opt.dataset_opt.patch_size,
                        size_hr=patch_size_hr,
                        border=4 + context_width * 2,
                        batch_size=batch_size,
                        overlap_mode="hann",
                        model_input_type=opt["input_type"],
                        unnorm=opt["dataset_opt"]["norm_type"] == "znormalization",
                    )

                    zarr_H = zarr.open(zarr_path, mode='r')
                    img_H = zarr_H[group_pair["H"]]
                    img_L = zarr_H[group_pair["L"]]

                    zarr_E = zarr.open(out_path, mode='r')
                    img_E = zarr_E['SR/0']  # Always read the top level

                elif inference_mode == 'in_memory':
                    # Open LR zarr and convert to numpy array
                    z = zarr.open(zarr_path, mode='r')
                    img_L = z[group_pair["L"]]
                    img_H = z[group_pair["H"]]

                    img_L = np.array(img_L)
                    img_L = np.reshape(img_L, (1, *img_L.shape))
                    img_L = torch.from_numpy(img_L)

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
                    )
                    img_L = img_L[0]  # assumes single channel dimension
                    img_E = img_E[0]  # assumes single channel dimension

                    print("HR shape:", img_H.shape, "LR shape:", img_L.shape, "SR shape:", img_E.shape)
                    # img_H = np.array(img_H)

                else:
                    raise ValueError(f"Inference mode {inference_mode} not recognized.")

                # Set values in SR prediction to zero where HR is zero to avoid metric bias
                _mask_zero_slices(img_E, img_H)

                start = time.time()
                vals, means = slice_metrics.get_avg_metrics(img_E, img_H)
                for metric_name in sample_vals:
                    sample_vals[metric_name].extend(vals[metric_name])
                    sample_means[metric_name].append(means[metric_name])
                    print("Sample %s: %0.6f" % (metric_name, means[metric_name]))
                    
                stop = time.time()
                print("Time elapsed for full sample evaluation:", stop - start)

                for axis in range(3):
                    target_shape = list(img_H.shape)
                    del target_shape[axis]  # remove slice axis
                    target_shape.insert(0, 1)  # prepend 1
                    # Save full slice comparisons over whole sample
                    baseline_comparison_tool = utils_3D_image.ImageComparisonTool3D(
                        patch_size_hr=target_shape,
                        upscaling_methods=["tio_nearest"],  ## or tio_linear
                        unnorm=False,
                        div_max=True,
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

            # Save group pair metrics
            write_metric_statistics(metric_file_path, sample_vals, sample_means, sample_names, text=group_text)

            # Write FID score once per dataset
            if "fid" in metric_names:
                with open(metric_file_path, "a+") as file:
                    fid = slice_metrics.compute_fid()
                    file.write(f"\nDATASET {name} FID SCORE: {fid}\n")
                    print(f"DATASET {name} FID SCORE: {fid}")

        # Write final dataset metric averages
        with open(metric_file_path, 'a+') as file:
            file.write("\nDATASET SAMPLE AVERAGES\n")
            
            for metric_name in sample_means:
                total_avg = np.mean(sample_means[metric_name])
                print("Sample %s: %0.6f" % (metric_name, total_avg))
                file.write("METRIC AVERAGE: " + str(total_avg.round(6)) + "\n")


if __name__ == "__main__":
    main()