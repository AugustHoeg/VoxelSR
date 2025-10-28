import argparse
import os

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from SaliencyModel.BackProp import GaussianBlurPath, GaussianBlurPath_2d
from SaliencyModel.BackProp import attribution_objective, Path_gradient, Path_gradient_ArSSR, make_coord
from SaliencyModel.BackProp import attribution_objective_2d, Path_gradient_2d
from SaliencyModel.BackProp import saliency_map_PG as saliency_map
from SaliencyModel.attributes import attr_grad, attr_grad_2d
from SaliencyModel.utils import cv2_to_pil, pil_to_cv2, gini, pad_to_shape_numpy
from SaliencyModel.utils import vis_saliency, vis_saliency_log, grad_abs_norm, grad_abs_norm_2d

import config
from utils.load_options import load_json, load_options_from_experiment_id


def parse_options(options_file=None):

    # Load default experiment option from config.py
    if config.MODEL_ARCHITECTURE == "mDCSRN_GAN":
        opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_mDCSRN_GAN.json')
    elif config.MODEL_ARCHITECTURE == "mDCSRN":
        opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_mDCSRN.json')
    elif config.MODEL_ARCHITECTURE == "SuperFormer":
        opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_SuperFormer.json')
    elif config.MODEL_ARCHITECTURE == "ESRGAN3D":
        opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_ESRGAN3D.json')
    elif config.MODEL_ARCHITECTURE == "RRDBNet3D":
        opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_RRDBNet3D.json')
    elif config.MODEL_ARCHITECTURE == "EDDSR":
        opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_EDDSR.json')
    elif config.MODEL_ARCHITECTURE == "MFER":
        opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_MFER.json')
    elif config.MODEL_ARCHITECTURE == "MTVNet":
        opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_MTVNet.json')
    elif config.MODEL_ARCHITECTURE == "ArSSR":
        opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_ArSSR.json')
    elif config.MODEL_ARCHITECTURE == "RCAN":
        opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_RCAN.json')
    elif config.MODEL_ARCHITECTURE == "SwinIR":
        opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_SwinIR.json')
    elif config.MODEL_ARCHITECTURE == "HAT":
        opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_HAT.json')
    elif config.MODEL_ARCHITECTURE == "DRCT":
        opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_DRCT.json')
    elif config.MODEL_ARCHITECTURE == "DUMMY":
        opt_path = os.path.join(config.ROOT_DIR, 'options', 'train_DUMMY.json')
    else:
        raise NotImplementedError('Model architecture %s not implemented.' % config.MODEL_ARCHITECTURE)

    # Load options
    opt = load_json(opt_path)

    return opt

def parse_LAM_arguments():
    # Parse command-line arguments for LAM analysis
    parser = argparse.ArgumentParser(description="Run LAM_3d with specified model.")

    # parser.add_argument("--experiment_id", type=str, required=True, help="Experiment ID for the trained model to use.")

    # Parse command-line arguments for LAM analysis
    parser.add_argument("--cube_no", type=str, required=False, default='001', help="ID of cube for LAM analysis")
    parser.add_argument("--h", type=int, required=False, default=28, help="Height of LAM ROI")
    parser.add_argument("--w", type=int, required=False, default=28, help="Width of LAM ROI")
    parser.add_argument("--d", type=int, required=False, default=28, help="Depth of LAM ROI")
    parser.add_argument("--window_size", type=int, required=False, default=8, help="Size of LAM ROI.")
    parser.add_argument("--use_new_cube_dir", type=int, required=False, default=1, help="use new cube directory")
    #parser.add_argument("--up_factor", default=1, type=int, required=False)

    args = parser.parse_args()

    return args


@hydra.main(version_base=None, config_path="../options", config_name=config.MODEL_ARCHITECTURE)
def main(opt: DictConfig):

    # Load LAM arguments from command line
    cube_no = f"{opt['LAM_opt']['cube_no']}"
    h = opt['LAM_opt']['h']
    w = opt['LAM_opt']['w']
    d = opt['LAM_opt']['d']
    window_size = opt['LAM_opt']['window_size']
    use_new_cube_dir = opt['LAM_opt']['use_new_cube_dir']
    dataset = opt['dataset_opt']['datasets']

    # Load options file from experiment ID
    experiment_id = opt['experiment_id']
    print("Experiment ID:", experiment_id)
    opt_path = load_options_from_experiment_id(experiment_id, root_dir=config.ROOT_DIR, file_type="yaml")
    opt = OmegaConf.load(opt_path)
    wandb_path = opt_path.rsplit("files", 1)[0]

    # Override datasets from command line
    opt['dataset_opt']['datasets'] = dataset

    # Set input type to 3D if not specified
    if 'input_type' not in opt:
        opt['input_type'] = '3D'

    # Define universal SR model using the KAIR define_Model framework
    from models.select_model import define_Model
    model = define_Model(opt, mode='test')

    model.init_test(experiment_id)

    # Override seed for reproducibility, as LAM is sensitive to random initializations
    seed_value = 8339
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)

    # Define parameters
    model_name = opt['model_opt']['model_architecture']
    up_factor = opt['up_factor']
    input_size = opt['dataset_opt']['patch_size']

    # %% Load test image
    if use_new_cube_dir:
        root_dir = config.ROOT_DIR
        lr_cube_dir = f"{root_dir}/saved_image_cubes/{dataset}/LR"
        hr_cube_dir = f"{root_dir}/saved_image_cubes/{dataset}/HR"
        img_lr = np.load(f"{lr_cube_dir}/cube_{input_size}_{cube_no}.npy")
        img_lr_full = np.load(f"{lr_cube_dir}/cube_{128}_{cube_no}.npy")
        img_hr = np.load(f"{hr_cube_dir}/cube_{cube_no}.npy")
        print(f"LR shape: {img_lr.shape}, HR shape: {img_hr.shape}, LR full shape: {img_lr_full.shape}")
    else:
        img_lr = np.load(f"saved_image_cubes/{opt['datasets']['name']}_{up_factor}x/LR/cube_{input_size}_{cube_no}.npy")
        img_hr = np.load(f"saved_image_cubes/{opt['datasets']['name']}_{up_factor}x/HR/cube_{cube_no}.npy")

    print(f"Using dataset: {dataset}")
    tensor_lr = torch.from_numpy(img_lr) ; tensor_hr = torch.from_numpy(img_hr)
    cv2_lr = np.moveaxis(tensor_lr.numpy(), 0, 3) ; cv2_hr = np.moveaxis(tensor_hr.numpy(), 0, 3)

    print(img_hr.shape)
    print(img_lr.shape)

    # %% Show image
    if model_name == "MTVNet":
        lr_pred_area = opt['netG']['context_sizes'][-1]
        z_idx_lr = (2 * d + window_size) // (2 * up_factor) + (input_size - lr_pred_area) // 2
    else:
        z_idx_lr = d // up_factor + window_size // (2 * up_factor)
    z_idx_hr = d+window_size//2
    plt.imshow(cv2_hr[:,:,d+window_size//2,:],cmap='gray')
    plt.imshow(cv2_lr[:,:,z_idx_lr,:],cmap='gray')

    # %% Draw rectangle on slice
    pil_hr = Image.fromarray((img_hr[0,:,:,z_idx_hr] * 255).astype(np.uint8))
    # pil_lr = Image.fromarray((img_lr[0,:,:,z_idx_lr] * 255).astype(np.uint8))
    draw_img_slice = pil_to_cv2(pil_hr)
    cv2.rectangle(draw_img_slice, (w, h), (w + window_size, h + window_size), (0, 0, 255), 1)
    position_pil = cv2_to_pil(draw_img_slice)


    if opt['input_type'] == "2D":
        # %% Calculate LAM 2D
        sigma = 1.2 ; fold = 25 ; l = 9
        alpha = 0.25 if dataset == "VoDaSuRe" else 0.25
        attr_objective = attribution_objective_2d(attr_grad_2d, h, w, window=window_size)
        gaus_blur_path_func = GaussianBlurPath_2d(sigma, fold, l)

        interpolated_grad_numpy, result_numpy, interpolated_numpy = Path_gradient_2d(tensor_lr[:,:,z_idx_lr].numpy(), model.netG, attr_objective, gaus_blur_path_func, cuda=True)
        grad_numpy, result = saliency_map(interpolated_grad_numpy, result_numpy)
        abs_normed_grad_numpy = grad_abs_norm_2d(grad_numpy)

        abs_normed_grad_numpy = pad_to_shape_numpy(abs_normed_grad_numpy, (128, 128))
        crop_idx = (128 - 32) // 2

        saliency_image_abs = vis_saliency(abs_normed_grad_numpy[crop_idx:-crop_idx, crop_idx:-crop_idx], zoomin=up_factor)
        angn_mean = abs_normed_grad_numpy
        angn_mean = (angn_mean - np.min(angn_mean)) / (np.max(angn_mean) - np.min(angn_mean))

        angn_mean_no_pad = angn_mean[crop_idx:-crop_idx, crop_idx:-crop_idx]
        angn_mean_no_pad = (angn_mean_no_pad - np.min(angn_mean_no_pad)) / (np.max(angn_mean_no_pad) - np.min(angn_mean_no_pad))

        saliency_image_abs_mean = vis_saliency(angn_mean[crop_idx:-crop_idx, crop_idx:-crop_idx], zoomin=up_factor)
        saliency_image_abs_mean_full = vis_saliency(angn_mean, zoomin=up_factor)
        saliency_image_abs_mean_full_log = vis_saliency_log(angn_mean, zoomin=up_factor)
        saliency_image_abs_full = vis_saliency(abs_normed_grad_numpy, zoomin=up_factor)
        saliency_image_abs_full_log = vis_saliency_log(abs_normed_grad_numpy, zoomin=up_factor)
        saliency_image_abs_zoom = vis_saliency(abs_normed_grad_numpy[crop_idx:-crop_idx, crop_idx:-crop_idx], zoomin=up_factor)
        # blend_abs_and_sr = cv2_to_pil(pil_to_cv2(saliency_image_abs_zoom) * (1.0 - alpha) + pil_to_cv2(result_im) * alpha)
        print("Shape of pil_hr", pil_hr.size)
        print("Shape of saliency_image_abs_zoom", saliency_image_abs_zoom.size)
        blend_abs_and_hr = cv2_to_pil(pil_to_cv2(saliency_image_abs_zoom) * (1.0 - alpha) + pil_to_cv2(pil_hr) * alpha)
        blend_mean = cv2_to_pil(pil_to_cv2(saliency_image_abs_mean) * (1.0 - alpha) + pil_to_cv2(pil_hr) * alpha)

        z_idx_lr = (2 * d + window_size) // (2 * up_factor) + (128 - 32) // 2

        gini_index = gini(abs_normed_grad_numpy)
        gini_index_no_pad = gini(abs_normed_grad_numpy[crop_idx:-crop_idx, crop_idx:-crop_idx])

        pil_lr_full = Image.fromarray((img_lr_full[0, :, :, z_idx_lr] * 255).astype(np.uint8))
        pil_lr_full_cv2 = pil_to_cv2(pil_lr_full)
        # start = ((128 - 32) // 2) + (window_size // up_factor) // 2
        start = ((128 - 32) // 2) + ((32 - (window_size // up_factor)) // 2)
        cv2.rectangle(pil_lr_full_cv2, (start, start), (start + window_size // up_factor, start + window_size // up_factor), (0, 0, 255), 1)
        position_lr_full = cv2_to_pil(pil_lr_full_cv2)
        blend_full_log = cv2_to_pil(pil_to_cv2(vis_saliency_log(abs_normed_grad_numpy, zoomin=1)) * (1.0 - alpha) + pil_to_cv2(pil_lr_full) * alpha)


    else:
        # %% Calculate LAM
        sigma = 1.2 ; fold = 25 ; l = 9
        alpha = 0.25 if dataset == "VoDaSuRe" else 0.25
        attr_objective = attribution_objective(attr_grad, h, w, d, window=window_size)
        gaus_blur_path_func = GaussianBlurPath(sigma, fold, l)
        if model_name == "ArSSR":
            xyz_hr = make_coord(list(tensor_hr.shape[1:])).unsqueeze(0)
            interpolated_grad_numpy, result_numpy, interpolated_numpy = Path_gradient_ArSSR(tensor_lr.numpy(), xyz_hr, model.netG, attr_objective, gaus_blur_path_func, cuda=True)
        else:
            interpolated_grad_numpy, result_numpy, interpolated_numpy = Path_gradient(tensor_lr.numpy(), model.netG, attr_objective, gaus_blur_path_func, cuda=True)
        grad_numpy, result = saliency_map(interpolated_grad_numpy, result_numpy)
        abs_normed_grad_numpy = grad_abs_norm(grad_numpy)


        # %% Make visualizations
        if model_name == "MTVNet" and (opt['netG']['num_levels'] > 1):
            abs_normed_grad_numpy = pad_to_shape_numpy(abs_normed_grad_numpy, (128, 128, 128))
            crop_idx = (128 - 32) // 2

            result_im = Image.fromarray((result[0, 0, :, :, z_idx_hr] * 255).astype(np.uint8))
            saliency_image_abs = vis_saliency(abs_normed_grad_numpy[crop_idx:-crop_idx, crop_idx:-crop_idx, z_idx_lr], zoomin=up_factor)
            angn_mean = np.mean(abs_normed_grad_numpy, axis=2)
            angn_mean = (angn_mean - np.min(angn_mean)) / (np.max(angn_mean) - np.min(angn_mean))

            angn_mean_no_pad = angn_mean[crop_idx:-crop_idx, crop_idx:-crop_idx]
            angn_mean_no_pad = (angn_mean_no_pad - np.min(angn_mean_no_pad)) / (np.max(angn_mean_no_pad) - np.min(angn_mean_no_pad))

            saliency_image_abs_mean = vis_saliency(angn_mean[crop_idx:-crop_idx, crop_idx:-crop_idx], zoomin=up_factor)
            saliency_image_abs_mean_full = vis_saliency(angn_mean, zoomin=up_factor)
            saliency_image_abs_mean_full_log = vis_saliency_log(angn_mean, zoomin=up_factor)
            saliency_image_abs_full = vis_saliency(abs_normed_grad_numpy[:, :, z_idx_lr], zoomin=up_factor)
            saliency_image_abs_full_log = vis_saliency_log(abs_normed_grad_numpy[:, :, z_idx_lr], zoomin=up_factor)
            saliency_image_abs_zoom = vis_saliency(abs_normed_grad_numpy[crop_idx:-crop_idx, crop_idx:-crop_idx, z_idx_lr], zoomin=up_factor)
            #blend_abs_and_sr = cv2_to_pil(pil_to_cv2(saliency_image_abs_zoom) * (1.0 - alpha) + pil_to_cv2(result_im) * alpha)
            print("Shape of pil_hr", pil_hr.size)
            print("Shape of saliency_image_abs_zoom", saliency_image_abs_zoom.size)
            blend_abs_and_hr = cv2_to_pil(pil_to_cv2(saliency_image_abs_zoom) * (1.0 - alpha) + pil_to_cv2(pil_hr) * alpha)
            blend_mean = cv2_to_pil(pil_to_cv2(saliency_image_abs_mean) * (1.0 - alpha) + pil_to_cv2(pil_hr) * alpha)

            z_idx_lr = (2 * d + window_size) // (2 * up_factor) + (128 - 32) // 2
            gini_index = gini(abs_normed_grad_numpy[:, :, z_idx_lr])
            gini_index_no_pad = gini(abs_normed_grad_numpy[crop_idx:-crop_idx, crop_idx:-crop_idx, z_idx_lr])

            pil_lr_full = Image.fromarray((img_lr_full[0, :, :, z_idx_lr] * 255).astype(np.uint8))
            pil_lr_full_cv2 = pil_to_cv2(pil_lr_full)
            start = ((128 - 32) // 2) + ((32 - (window_size // up_factor)) // 2)
            cv2.rectangle(pil_lr_full_cv2, (start, start), (start + window_size//up_factor, start + window_size//up_factor), (0, 0, 255), 1)
            position_lr_full = cv2_to_pil(pil_lr_full_cv2)
            blend_full_log = cv2_to_pil(pil_to_cv2(vis_saliency_log(abs_normed_grad_numpy[:, :, z_idx_lr], zoomin=1)) * (1.0 - alpha) + pil_to_cv2(pil_lr_full) * alpha)
        else:

            z_idx_lr = (2 * d + window_size) // (2 * up_factor) + (128 - 32) // 2
            result_im = Image.fromarray((result[0, 0, :, :, z_idx_hr] * 255).astype(np.uint8))
            abs_normed_grad_numpy = pad_to_shape_numpy(abs_normed_grad_numpy, (128, 128, 128))
            crop_idx = (128 - 32) // 2

            saliency_image_abs = vis_saliency(abs_normed_grad_numpy[crop_idx:-crop_idx, crop_idx:-crop_idx, z_idx_lr], zoomin=up_factor)
            angn_mean = np.mean(abs_normed_grad_numpy,axis=2)
            angn_mean = (angn_mean - np.min(angn_mean)) / (np.max(angn_mean) - np.min(angn_mean))

            angn_mean_no_pad = angn_mean[crop_idx:-crop_idx, crop_idx:-crop_idx]
            angn_mean_no_pad = (angn_mean_no_pad - np.min(angn_mean_no_pad)) / (np.max(angn_mean_no_pad) - np.min(angn_mean_no_pad))

            saliency_image_abs_mean = vis_saliency(angn_mean[crop_idx:-crop_idx, crop_idx:-crop_idx], zoomin=up_factor)
            saliency_image_abs_mean_full = vis_saliency(angn_mean, zoomin=up_factor)
            saliency_image_abs_mean_full_log = vis_saliency_log(angn_mean, zoomin=up_factor)
            saliency_image_abs_full = vis_saliency(abs_normed_grad_numpy[:, :, z_idx_lr], zoomin=up_factor)
            saliency_image_abs_full_log = vis_saliency_log(abs_normed_grad_numpy[:, :, z_idx_lr], zoomin=up_factor)
            saliency_image_abs_zoom = vis_saliency(abs_normed_grad_numpy[crop_idx:-crop_idx, crop_idx:-crop_idx, z_idx_lr], zoomin=up_factor)
            #blend_abs_and_sr = cv2_to_pil(pil_to_cv2(saliency_image_abs_zoom) * (1.0 - alpha) + pil_to_cv2(result_im) * alpha)
            print("Shape of pil_hr", pil_hr.size)
            print("Shape of saliency_image_abs_zoom", saliency_image_abs_zoom.size)
            blend_abs_and_hr = cv2_to_pil(pil_to_cv2(saliency_image_abs_zoom) * (1.0 - alpha) + pil_to_cv2(pil_hr) * alpha)
            blend_mean = cv2_to_pil(pil_to_cv2(saliency_image_abs_mean) * (1.0 - alpha) + pil_to_cv2(pil_hr) * alpha)

            gini_index = gini(abs_normed_grad_numpy[:, :, z_idx_lr])
            gini_index_no_pad = gini(abs_normed_grad_numpy[crop_idx:-crop_idx, crop_idx:-crop_idx, z_idx_lr])

            pil_lr_full = Image.fromarray((img_lr_full[0, :, :, z_idx_lr] * 255).astype(np.uint8))
            pil_lr_full_cv2 = pil_to_cv2(pil_lr_full)
            #start = ((128 - 32) // 2) + (window_size // up_factor) // 2
            start = ((128 - 32) // 2) + ((32 - (window_size // up_factor)) // 2)
            cv2.rectangle(pil_lr_full_cv2, (start, start), (start + window_size // up_factor, start + window_size // up_factor), (0, 0, 255), 1)
            position_lr_full = cv2_to_pil(pil_lr_full_cv2)
            blend_full_log = cv2_to_pil(pil_to_cv2(vis_saliency_log(abs_normed_grad_numpy[:, :, z_idx_lr], zoomin=1)) * (1.0 - alpha) + pil_to_cv2(pil_lr_full) * alpha)

    diffusion_index = (1 - gini_index) * 100
    diffusion_index = diffusion_index * 2**3 # scale to padded input size of 128^3
    print(f"The DI of this case is {diffusion_index}")

    gini_index_mean = gini(angn_mean)
    diffusion_index_mean = (1 - gini_index_mean) * 100
    diffusion_index_mean = diffusion_index_mean * 2**3  # scale to padded input size of 128^3
    print(f"The DI_mean of this case is {diffusion_index_mean}")

    diffusion_index_no_pad = (1 - gini_index_no_pad) * 100
    print(f"The DI (no pad) of this case is {diffusion_index_no_pad}")

    gini_index_mean_no_pad = gini(angn_mean_no_pad)
    diffusion_index_mean_no_pad = (1 - gini_index_mean_no_pad) * 100
    print(f"The DI_mean (no pad) of this case is {diffusion_index_mean_no_pad}")

    # %% Show LAM
    fig, axs = plt.subplots(1,3,figsize=(14,4))
    axs[0].imshow(position_pil)
    axs[1].imshow(saliency_image_abs)
    axs[2].imshow(saliency_image_abs_mean)

    # %% Save results
    cube_dir = f"{dataset}_cube_{cube_no}_win{window_size}_h{h}-w{w}-d{d}"
    if use_new_cube_dir:
        cube_dir = cube_dir + "_new"
    if not os.path.exists("Results/" + cube_dir):
        os.makedirs("Results/" + cube_dir, exist_ok=True)
        os.makedirs("Results/" + cube_dir + "/lam_out", exist_ok=True)

    np.save(f'Results/{cube_dir}/lam_out/angn_{model_name}_{experiment_id}_{cube_dir}.npy',abs_normed_grad_numpy)

    plt.figure()
    plt.imshow(position_pil)
    plt.axis('off')
    plt.savefig(f'Results/{cube_dir}/selection_{cube_no}_h{h}-w{w}-d{d}.png', bbox_inches='tight', pad_inches=0)

    plt.figure()
    plt.imshow(position_lr_full)
    plt.axis('off')
    plt.savefig(f'Results/{cube_dir}/lr_full_{cube_no}_h{h}-w{w}-d{d}.png', bbox_inches='tight', pad_inches=0)

    plt.figure()
    plt.imshow(saliency_image_abs)
    plt.axis('off')
    plt.savefig(f'Results/{cube_dir}/{model_name}_{experiment_id}_{cube_no}_h{h}-w{w}-d{d}.png', bbox_inches='tight', pad_inches=0)

    plt.figure()
    plt.imshow(saliency_image_abs_full)
    plt.axis('off')
    plt.savefig(f'Results/{cube_dir}/{model_name}_{experiment_id}_{cube_no}_h{h}-w{w}-d{d}_full.png', bbox_inches='tight', pad_inches=0)

    plt.figure()
    plt.imshow(saliency_image_abs_full_log)
    plt.axis('off')
    plt.savefig(f'Results/{cube_dir}/{model_name}_{experiment_id}_{cube_no}_h{h}-w{w}-d{d}_full_log.png', bbox_inches='tight', pad_inches=0)

    plt.figure()
    plt.imshow(blend_full_log)
    plt.axis('off')
    plt.savefig(f'Results/{cube_dir}/{model_name}_{experiment_id}_{cube_no}_h{h}-w{w}-d{d}_full_log_blend.png', bbox_inches='tight', pad_inches=0)

    plt.figure()
    plt.imshow(saliency_image_abs_mean)
    plt.axis('off')
    plt.savefig(f'Results/{cube_dir}/{model_name}_{experiment_id}_{cube_no}_h{h}-w{w}-d{d}_mean.png', bbox_inches='tight', pad_inches=0)

    plt.figure()
    plt.imshow(blend_mean)
    plt.axis('off')
    plt.savefig(f'Results/{cube_dir}/{model_name}_{experiment_id}_{cube_no}_h{h}-w{w}-d{d}_mean_blend.png', bbox_inches='tight', pad_inches=0)

    plt.figure()
    plt.imshow(saliency_image_abs_mean_full)
    plt.axis('off')
    plt.savefig(f'Results/{cube_dir}/{model_name}_{experiment_id}_{cube_no}_h{h}-w{w}-d{d}_mean_full.png', bbox_inches='tight', pad_inches=0)

    plt.figure()
    plt.imshow(saliency_image_abs_mean_full_log)
    plt.axis('off')
    plt.savefig(f'Results/{cube_dir}/{model_name}_{experiment_id}_{cube_no}_h{h}-w{w}-d{d}_mean_full_log.png', bbox_inches='tight', pad_inches=0)

    plt.figure()
    plt.imshow(blend_abs_and_hr)
    plt.axis('off')
    plt.savefig(f'Results/{cube_dir}/{model_name}_{experiment_id}_{cube_no}_h{h}-w{w}-d{d}_overlay.png', bbox_inches='tight', pad_inches=0)

    # %% Write to file
    with open(f'Results/{cube_dir}/LAM_DI.txt', 'a') as f:
        f.write(f'Diffusion index for {model_name}, {experiment_id}: {diffusion_index} ({cube_no}; selection: h{h}-w{w}-d{d})\n')
        f.write(f'Diffusion index (MEAN) for {model_name}, {experiment_id}: {diffusion_index_mean} ({cube_no}; selection: h{h}-w{w}-d{d})\n')
        f.write(f'Diffusion index (no pad) for {model_name}, {experiment_id}: {diffusion_index_no_pad} ({cube_no}; selection: h{h}-w{w}-d{d})\n')
        f.write(f'Diffusion index (MEAN, no pad) for {model_name}, {experiment_id}: {diffusion_index_mean_no_pad} ({cube_no}; selection: h{h}-w{w}-d{d})\n')
        #f.write(f'Gradient magnitude sum over full input for {model_name}, {experiment_id}: {grad_mag_sum} ({cube_no}; selection: h{h}-w{w}-d{d})\n')
        #f.write(f'Gradient magnitude sum over SR ROI for {model_name}, {experiment_id}: {grad_mag_sum_roi} ({cube_no}; selection: h{h}-w{w}-d{d})\n')

    # %%


if __name__ == '__main__':
    main()
