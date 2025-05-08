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
from data.train_transforms_implicit import ImplicitModelTransformFastd
from models.model_implicit import coords_to_image
from utils import utils_3D_image
from utils.load_options import load_options_from_experiment_id
from utils.utils_2D_image import upscale_slices, upscale_slices_upfactor
from utils.utils_image import calculate_ssim_2D, calculate_nrmse_2D, calculate_psnr_2D


def plot_slice_comparison(img_L, img_E, img_H, opt):

    idx = img_L.shape[-1] // 2
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(img_L[0, :, :, idx], cmap='gray')
    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([])  # Remove y-axis ticks
    plt.subplot(1, 3, 2)
    plt.imshow(img_E[0, :, :, idx * opt['up_factor']].float().numpy(), cmap='gray')
    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([])  # Remove y-axis ticks
    plt.subplot(1, 3, 3)
    plt.imshow(img_H[0, :, :, idx * opt['up_factor']], cmap='gray')
    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([])  # Remove y-axis ticks
    # Adjust subplot parameters to remove borders
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()

def create_metric_files(metric_vals, wandb_path, opt):

    for dim in range(3):

        avg_psnr_list = metric_vals[dim]['avg_psnr_list']
        avg_ssim_list = metric_vals[dim]['avg_ssim_list']
        avg_nrmse_list = metric_vals[dim]['avg_nrmse_list']

        psnr_slice_list = metric_vals[dim]['psnr_slice_list']
        ssim_slice_list = metric_vals[dim]['ssim_slice_list']
        nrmse_slice_list = metric_vals[dim]['nrmse_slice_list']

        psnr_slice_mean, ci_psnr = get_mean_and_ci(psnr_slice_list)
        ssim_slice_mean, ci_ssim = get_mean_and_ci(ssim_slice_list)
        nrmse_slice_mean, ci_nrmse = get_mean_and_ci(nrmse_slice_list)

        # Create a file with performance metric statistics
        file_dir = os.path.join(wandb_path, "files/")
        if not os.path.exists(file_dir + "performance_statistics/"):
            os.makedirs(file_dir + "performance_statistics/")

        performance_statistics_dir = os.path.join(file_dir, "performance_statistics/")

        # Specify the file name
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"performance_metrics_dim_{dim+1}_{opt['experiment_id']}_{current_time}.txt"

        # Open the file in write mode and write the contents
        with open(performance_statistics_dir + file_name, 'w') as file:
            file.write("MODEL ARCHITECTURE: " + opt['model_opt']['model_architecture'] + "\n")
            file.write("DATASET: " + opt['dataset_opt']['name'] + "\n")
            file.write("EXPERIMENT ID: " + opt['experiment_id'] + "\n")
            file.write("RUN NAME: " + opt['run_name'] + "\n")
            file.write("MAX ITERATIONS: " + str(opt['train_opt']['iterations']) + "\n")

            file.write("PATCH SIZE: " + str(opt['dataset_opt']['patch_size']) + "\n")
            file.write("UP FACTOR: " + str(opt['up_factor']) + "\n")
            file.write("LEARNING RATE: " + str(opt['train_opt']['G_optimizer_lr']) + "\n")

            file.write("SLICE DIMENSION: " + str(dim+1) + "\n")

            # Write the lists to the file
            file.write("PERFORMANCE METRIC LISTS \n")
            file.write("PSNR SAMPLE LIST: " + str(torch.tensor(avg_psnr_list).numpy()) + "\n")
            file.write("SSIM SAMPLE LIST: " + str(torch.tensor(avg_ssim_list).numpy()) + "\n")
            file.write("NRMSE SAMPLE LIST: " + str(torch.tensor(avg_nrmse_list).numpy()) + "\n")

            # Write the individual values to the file
            file.write("FINAL AVERAGE SAMPLE-WISE PERFORMANCE METRICS \n")
            file.write("AVERAGE SAMPLE-WISE PSNR: " + str(np.mean(avg_psnr_list)) + "\n")
            file.write("AVERAGE SAMPLE-WISE SSIM: " + str(np.mean(avg_ssim_list)) + "\n")
            file.write("AVERAGE SAMPLE-WISE NRSME: " + str(np.mean(avg_nrmse_list)) + "\n")

            # Write the individual values to the file
            file.write("FINAL AVERAGE SLICE-WISE PERFORMANCE METRICS \n")
            file.write("AVERAGE SLICE-WISE PSNR: " + str(psnr_slice_mean) + "+-" + str(ci_psnr) + "\n")
            file.write("AVERAGE SLICE-WISE SSIM: " + str(ssim_slice_mean) + "+-" + str(ci_ssim) + "\n")
            file.write("AVERAGE SLICE-WISE NRSME: " + str(nrmse_slice_mean) + "+-" + str(ci_nrmse) + "\n")

        print(f"File '{file_name}' has been created and saved.")



def get_full_sample_metrics(img_H, img_E, slice_dim=3, slice_step=1):

    num_slices = img_H.shape[slice_dim]

    # Compute PSNR, SSIM and NRMSE slice-wise. Slice-wise approach is chosen as some dataset samples are very large.
    psnr_slice_list = []
    ssim_slice_list = []
    nrmse_slice_list = []

    for i in range(0, num_slices, slice_step):
        if slice_dim == 1:
            H_slice = img_H[:, i, :, :].float().squeeze().clamp(min=0.0, max=1.0).numpy()
            E_slice = img_E[:, i, :, :].float().squeeze().clamp(min=0.0, max=1.0).numpy()
        elif slice_dim == 2:
            H_slice = img_H[:, :, i, :].float().squeeze().clamp(min=0.0, max=1.0).numpy()
            E_slice = img_E[:, :, i, :].float().squeeze().clamp(min=0.0, max=1.0).numpy()
        else:
            H_slice = img_H[:, :, :, i].float().squeeze().clamp(min=0.0, max=1.0).numpy()
            E_slice = img_E[:, :, :, i].float().squeeze().clamp(min=0.0, max=1.0).numpy()

        slice_psnr = calculate_psnr_2D(E_slice, H_slice, border=0)
        psnr_slice_list.append(slice_psnr)

        slice_ssim = calculate_ssim_2D(E_slice, H_slice, border=0)
        ssim_slice_list.append(slice_ssim)

        slice_nrmse = calculate_nrmse_2D(E_slice, H_slice, border=0)
        nrmse_slice_list.append(slice_nrmse)

    return psnr_slice_list, ssim_slice_list, nrmse_slice_list


def get_mean_and_ci(data_sequence, confidence=0.95):

    data = np.array(data_sequence)
    n = len(data)
    mean, se = np.mean(data), stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return mean, h

def trim_background_slices(img_L, img_E, img_H, dataset_name, up_factor, threshhold_percentage=0.20):
    # Set a simple threshold for the background. Any voxel larger will be considered brain/bone
    # A slice is discarded if it contains less can X% foreground voxels.
    if dataset_name == "KIRBY21":
        lr_foreground_threshold = 0.02  # Kirby21 is T2w and therefore we need lower threshold
    elif dataset_name == "2022_QIM_52_Bone" or dataset_name == "Synthetic_2022_QIM_52_Bone":
        lr_foreground_threshold = 0.02
    elif dataset_name == "IXI":
        # Rotate such that scans are z-sliced, then flip to match orientation of other datasets
        img_L = torch.flip(img_L.permute(0, 3, 1, 2), dims=[1, 2, 3])
        img_E = torch.flip(img_E.permute(0, 3, 1, 2), dims=[1, 2, 3])
        img_H = torch.flip(img_H.permute(0, 3, 1, 2), dims=[1, 2, 3])
        lr_foreground_threshold = 0.05
    elif dataset_name == "BRATS2023":
        lr_foreground_threshold = 0.05
    else:
        lr_foreground_threshold = 0.05  # The rest of the datasets use this:

    C_lr, H_lr, W_lr, D_lr = img_L.shape
    num_voxels_foreground = torch.sum(img_L.reshape(-1, D_lr) > lr_foreground_threshold, dim=0)
    lr_foreground_slices = num_voxels_foreground > (H_lr * W_lr * threshhold_percentage)
    img_L = img_L[:, :, :, lr_foreground_slices]
    print("Number of LR foreground slices %d / %d" % (torch.sum(lr_foreground_slices).numpy(), D_lr))
    img_E = img_E[:, :, :, lr_foreground_slices.repeat_interleave(up_factor)]
    img_H = img_H[:, :, :, lr_foreground_slices.repeat_interleave(up_factor)]
    return img_L, img_E, img_H

def trim_background_slices_dim(img_L, img_E, img_H, dataset_name, up_factor, threshhold_percentage=0.20, slice_dim=3):
    # Set a simple threshold for the background. Any voxel larger will be considered brain/bone
    # A slice is discarded if it contains less can X% foreground voxels.
    if dataset_name == "KIRBY21":
        lr_foreground_threshold = 0.02  # Kirby21 is T2w and therefore we need lower threshold
    elif dataset_name == "2022_QIM_52_Bone" or dataset_name == "Synthetic_2022_QIM_52_Bone":
        lr_foreground_threshold = 0.02
    elif dataset_name == "IXI":
        # Rotate such that scans are z-sliced, then flip to match orientation of other datasets
        #img_L = torch.flip(img_L.permute(0, 3, 1, 2), dims=[1, 2, 3])
        #img_E = torch.flip(img_E.permute(0, 3, 1, 2), dims=[1, 2, 3])
        #img_H = torch.flip(img_H.permute(0, 3, 1, 2), dims=[1, 2, 3])
        lr_foreground_threshold = 0.05
    elif dataset_name == "BRATS2023":
        lr_foreground_threshold = 0.05
    else:
        lr_foreground_threshold = 0.05  # The rest of the datasets use this:

    num_slices = img_L.shape[slice_dim]
    slice_area = np.prod(img_L.shape[1:]) // num_slices

    mask = img_L > lr_foreground_threshold
    if slice_dim == 1:
        num_voxels_foreground = torch.sum(mask, axis=(0, 2, 3))
        lr_foreground_slices = num_voxels_foreground > (slice_area * threshhold_percentage)
        img_L = img_L[:, lr_foreground_slices, :, :]
        img_E = img_E[:, lr_foreground_slices.repeat_interleave(up_factor), :, :]
        img_H = img_H[:, lr_foreground_slices.repeat_interleave(up_factor), :, :]
    elif slice_dim == 2:
        num_voxels_foreground = torch.sum(mask, axis=(0, 1, 3))
        lr_foreground_slices = num_voxels_foreground > (slice_area * threshhold_percentage)
        img_L = img_L[:, :, lr_foreground_slices, :]
        img_E = img_E[:, :, lr_foreground_slices.repeat_interleave(up_factor), :]
        img_H = img_H[:, :, lr_foreground_slices.repeat_interleave(up_factor), :]
    else:
        num_voxels_foreground = torch.sum(mask, axis=(0, 1, 2))
        lr_foreground_slices = num_voxels_foreground > (slice_area * threshhold_percentage)
        img_L = img_L[:, :, :, lr_foreground_slices]
        img_E = img_E[:, :, :, lr_foreground_slices.repeat_interleave(up_factor)]
        img_H = img_H[:, :, :, lr_foreground_slices.repeat_interleave(up_factor)]

    print("Number of LR foreground slices %d / %d" % (torch.sum(lr_foreground_slices).numpy(), num_slices))

    return img_L, img_E, img_H

def apply_implicit_transform(batch, transform, out_dtype=torch.float32):
    """
    Applies implicit transform to each patch in the batch and returns the transformed batch.
    """
    L = []
    H = []
    H_xyz = []
    for patch_lr, patch_hr in zip(batch[0]['L']['data'], batch[1]['H']['data']):
        patch_dict = transform({'H': patch_hr, 'L': patch_lr})

        L.append(patch_dict['L'])
        H.append(patch_dict['H'])
        H_xyz.append(patch_dict['H_xyz'])

    L = torch.stack(L)
    H = torch.stack(H)
    H_xyz = torch.stack(H_xyz)

    if out_dtype == torch.float32:
        return {'L': L.float(), 'H': H.float(), 'H_xyz': H_xyz}
    else:
        return {'L': L, 'H': H, 'H_xyz': H_xyz}


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

    # Overwriting dataset type to MonaiDataset for testing
    opt['dataset_opt']["dataset_type"] = "MonaiDataset"

    from data.select_dataset import define_Dataset
    train_dataset, test_dataset, baseline_dataset = define_Dataset(opt)

    dataloader_params_train = opt['dataset_opt']['train_dataloader_params']
    dataloader_params_test = opt['dataset_opt']['test_dataloader_params']

    baseline_loader = monai.data.DataLoader(baseline_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=dataloader_params_test['num_load_workers'],
                                            persistent_workers=dataloader_params_test['persist_workers'],
                                            pin_memory=dataloader_params_test['pin_memory'])

    psnr_list, ssim_list, nrmse_list = [], [], []

    if opt['model_opt']['model_architecture'] == "MTVNet":
        patch_size = opt['dataset_opt']['patch_size']
        center_size = opt['model_opt']['netG']['context_sizes'][-1]  # New
        context_width = (patch_size - center_size) // 2
        patch_size_hr = center_size * opt['up_factor']
    else:
        patch_size = opt['dataset_opt']['patch_size']
        context_width = 0
        patch_size_hr = opt['dataset_opt']['patch_size_hr']

    border = 4  # border voxels
    border_hr = border * opt['up_factor']

    # Create directory for test patch comparisons
    image_dir = os.path.join(wandb_path, "files/", "media/", "images/")
    print("Saving image comparisons to:", image_dir)
    if not os.path.exists(image_dir + "patch_comparisons/"):
        os.makedirs(image_dir + "patch_comparisons/")
    if not os.path.exists(image_dir + "full_slice_comparisons/"):
        os.makedirs(image_dir + "full_slice_comparisons/")

    implicit_model_transform = ImplicitModelTransformFastd(opt['up_factor'], mode="test")

    slice_step = 1
    if opt['input_type'] and opt['up_factor'] > 1:
        slice_step = opt['up_factor']

    print("RUNNING TEST")

    # For computing mean and std across each sample
    psnr_slice_list = []
    ssim_slice_list = []
    nrmse_slice_list = []

    metric_vals = [{"avg_psnr_list": [], "avg_ssim_list": [], "avg_nrmse_list": [], "psnr_slice_list": [], "ssim_slice_list": [], "nrmse_slice_list": []},
                   {"avg_psnr_list": [], "avg_ssim_list": [], "avg_nrmse_list": [], "psnr_slice_list": [], "ssim_slice_list": [], "nrmse_slice_list": []},
                   {"avg_psnr_list": [], "avg_ssim_list": [], "avg_nrmse_list": [], "psnr_slice_list": [], "ssim_slice_list": [], "nrmse_slice_list": []}]

    for sample_idx, baseline_batch in enumerate(baseline_loader):

        # Assume batch_size of baseline_loader is always one (only reconstruct one sample in the dataset at a time)
        img_H = baseline_batch['H'][0]
        img_L = baseline_batch['L'][0]
        img_E = torch.zeros_like(img_H)
        del baseline_batch

        overlap_lr = border
        overlap_hr = border * opt['up_factor']
        subject_hr = tio.Subject(H=tio.ScalarImage(tensor=img_H))
        subject_lr = tio.Subject(L=tio.ScalarImage(tensor=img_L))
        grid_sampler_lr = tio.GridSampler(subject=subject_lr, patch_size=patch_size, patch_overlap=2*overlap_lr+2*context_width, padding_mode=None)
        grid_sampler_hr = tio.GridSampler(subject=subject_hr, patch_size=patch_size_hr, patch_overlap=2*overlap_hr, padding_mode=None)

        test_batch_size = opt['dataset_opt']['test_dataloader_params']['dataloader_batch_size']
        #patch_loader_lr = torch.utils.data.DataLoader(grid_sampler_lr, batch_size=test_batch_size)
        #patch_loader_hr = torch.utils.data.DataLoader(grid_sampler_hr, batch_size=test_batch_size)
        patch_loader_lr = tio.SubjectsLoader(grid_sampler_lr, batch_size=test_batch_size)
        patch_loader_hr = tio.SubjectsLoader(grid_sampler_hr, batch_size=test_batch_size)
        aggregator_hr = tio.inference.GridAggregator(grid_sampler_hr, overlap_mode='hann')

        model.netG.eval()
        i = 0
        time_in = time.time()

        with torch.inference_mode():
            c = 1
            for patches_batch_lr, patches_batch_hr in tqdm(zip(patch_loader_lr, patch_loader_hr), desc='Reconstructing patches', mininterval=2):
                if opt['model_opt']['model'] == 'implicit':
                    data = apply_implicit_transform((patches_batch_lr, patches_batch_hr), implicit_model_transform)
                    model.feed_data(data)
                    model.netG_forward()
                    sr_patch = coords_to_image(model.E, patch_size=opt['dataset_opt']['patch_size_hr'])
                elif opt['input_type'] == '2D':
                    # Upsample 2D slices individually
                    if opt['up_factor'] > 1:
                        sr_patch = upscale_slices_upfactor(model, patches_batch_lr['L']['data'], patches_batch_hr['H']['data'], 16, opt['up_factor'])
                    else:
                        sr_patch = upscale_slices(model, patches_batch_lr['L']['data'], patches_batch_hr['H']['data'], batch_size_2D=16)
                else:
                    model.feed_data({'H': patches_batch_hr['H'], 'L': patches_batch_lr['L']}, add_key='data')
                    model.netG_forward()
                    sr_patch = model.E
                locations_hr = patches_batch_hr['location']
                aggregator_hr.add_batch(sr_patch, locations_hr)

        img_E = aggregator_hr.get_output_tensor().float()  # convert from FP16 to FP32
        print("Full reconstruction size:", img_E.size())
        img_H = img_H.unsqueeze(0)
        img_L = img_L.unsqueeze(0)
        img_E = img_E.unsqueeze(0)

        print(i)
        time_end = time.time()

        print(f'Time taken for sample {sample_idx}: {time_end - time_in} seconds')

        if opt['model_opt']['model_architecture'] == "MTVNet":
            # Crop context if needed
            if context_width > 0:
                img_L = img_L[:, :, context_width:-context_width, context_width:-context_width, context_width:-context_width]

        # Crop the borders
        if border > 0:
            img_L = img_L[:, :, border:-border, border:-border, border:-border]
            img_E = img_E[:, :, border_hr:-border_hr, border_hr:-border_hr, border_hr:-border_hr]
            img_H = img_H[:, :, border_hr:-border_hr, border_hr:-border_hr, border_hr:-border_hr]

        for dim in range(3):
            # Trim background slices, assuming batch size is 1
            img_L_trim, img_E_trim, img_H_trim = trim_background_slices_dim(img_L[0], img_E[0], img_H[0],
                                                                            opt['dataset_opt']['name'],
                                                                            up_factor=opt['up_factor'],
                                                                            threshhold_percentage=.20,
                                                                            slice_dim=dim + 1)

            # Compute PSNR, SSIM and NRMSE along the slice dimensions
            psnr_slice_list, ssim_slice_list, nrmse_slice_list = get_full_sample_metrics(img_H_trim, img_E_trim,
                                                                                         slice_dim=dim + 1,
                                                                                         slice_step=slice_step)
            metric_vals[dim]['psnr_slice_list'].extend(psnr_slice_list)
            metric_vals[dim]['ssim_slice_list'].extend(ssim_slice_list)
            metric_vals[dim]['nrmse_slice_list'].extend(nrmse_slice_list)
            sample_psnr = np.mean(psnr_slice_list)
            sample_ssim = np.mean(ssim_slice_list)
            sample_nrmse = np.mean(nrmse_slice_list)
            metric_vals[dim]['avg_psnr_list'].append(sample_psnr)
            metric_vals[dim]['avg_ssim_list'].append(sample_ssim)
            metric_vals[dim]['avg_nrmse_list'].append(sample_nrmse)
            print("Dimension %d, Sample PSNR: %0.4f, SSIM: %0.6f, NRMSE: %0.6f" % (dim+1, sample_psnr, sample_ssim, sample_nrmse))

        H_trim, W_trim, D_trim = img_H_trim.shape[1:]

        # Plot slice for visual inspection
        if opt['run_type'] == "HOME PC":
            plot_slice_comparison(img_L_trim, img_E_trim, img_H_trim, opt)

        # Save full slice comparisons over whole sample
        baseline_comparison_tool = utils_3D_image.ImageComparisonTool3D(
            patch_size_hr=(H_trim, W_trim, D_trim),
            upscaling_methods=["tio_nearest"],  ## or tio_linear
            unnorm=False,
            div_max=False,
            out_dtype=np.uint8)

        img_dict = {'H': img_H_trim, 'E': img_E_trim, 'L': img_L_trim}
        slice_idx_list = np.linspace(D_trim // 4, D_trim - D_trim // 4, 5)  # Save 5 slices even spaced inside the middle of the last scan dimension
        for slice_idx in slice_idx_list:
            grid_image = baseline_comparison_tool.get_comparison_image(img_dict, slice_idx=int(slice_idx))
            file_name = os.path.join(image_dir, "full_slice_comparisons/full_sample_comparison_%d_%s_%dx.png" % (
            slice_idx, opt['model_opt']['model_architecture'], opt['up_factor']))
            grid_image = Image.fromarray(grid_image)
            grid_image.save(file_name)

    # Create files with performance metric statistics
    create_metric_files(metric_vals, wandb_path, opt)

    print("Done")

if __name__ == "__main__":
    main()