import datetime
import os
import time

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

import numpy as np
import scipy.stats as stats
import torch

from tqdm import tqdm

import config
from utils.load_options import load_json, parse_arguments, load_options_from_experiment_id
from utils import utils_3D_image

def create_inference_speed_file(inference_time_list, wandb_path, opt):

    # Create a file with performance metric statistics
    file_dir = os.path.join(wandb_path, "files/")
    if not os.path.exists(file_dir + "inference_statistics/"):
        os.makedirs(file_dir + "inference_statistics/")

    inference_statistics_dir = os.path.join(file_dir, "inference_statistics/")

    inference_speed_mean, ci = get_mean_and_ci(inference_time_list)

    # Specify the file name
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"inference_statistics_{opt['experiment_id']}_{current_time}.txt"

    # Open the file in write mode and write the contents
    with open(inference_statistics_dir + file_name, 'w') as file:
        file.write("MODEL ARCHITECTURE: " + opt['model_opt']['model_architecture'] + "\n")
        file.write("TRAINABLE PARAMETERS: " + str(opt['params_netG']) + "\n")
        file.write("DATASET: " + opt['dataset_opt']['name'] + "\n")
        file.write("EXPERIMENT ID: " + opt['experiment_id'] + "\n")
        file.write("RUN NAME: " + opt['run_name'] + "\n")

        file.write("PATCH SIZE: " + str(opt['dataset_opt']['patch_size']) + "\n")
        file.write("UP FACTOR: " + str(opt['up_factor']) + "\n")

        # Write the lists to the file
        file.write("INFERENCE SPEED LIST: " + str(torch.tensor(inference_time_list).numpy()) + "\n")

        # Write the individual values to the file
        file.write("AVERAGE INFERENCE SPEED: " + str(inference_speed_mean) + "+-" + str(ci) + "\n")

    print(f"File '{file_name}' has been created and saved.")


def get_mean_and_ci(data_sequence, confidence=0.95):

    data = np.array(data_sequence)
    n = len(data)
    mean, se = np.mean(data), stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return mean, h

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

    # Overwrite gradient checkpointing
    opt['model_opt']['netG']['use_checkpoint'] = False

    print("Cuda is available", torch.cuda.is_available())
    print("Cuda device count", torch.cuda.device_count())
    print("Cuda current device", torch.cuda.current_device())
    print("Cuda device name", torch.cuda.get_device_name(0))

    from models.select_model import define_Model
    model = define_Model(opt)
    model.init_test(experiment_id)

    # Generate random input data for inference
    patch_size = opt['dataset_opt']['patch_size']
    if opt['input_type'] == "2D":
        input_data = torch.randn(1, 1, patch_size, patch_size).cuda()
    else:
        input_data = torch.randn(1, 1, patch_size, patch_size, patch_size).cuda()

    iterations = 100  # Number of iterations to run inference speed test

    # Warm-up runs
    for _ in range(5):
        model.feed_data({'L': input_data}, need_H=False)
        _ = model.netG_forward()

    inference_time_list = []

    # Measure inference time with GPU synchronization
    for _ in tqdm(range(iterations), desc=f"Running inference speed test", mininterval=1):
        model.feed_data({'L': input_data}, need_H=False)
        torch.cuda.synchronize()
        start_time = time.time()
        output = model.netG_forward()
        torch.cuda.synchronize()
        end_time = time.time()
        inference_time = end_time - start_time
        #print(f"Inference time with GPU synchronization: {inference_time:.4f} seconds")

        inference_time_list.append(inference_time)

    # Print the number of parameters in the model
    params_netG = utils_3D_image.numel(model.netG, only_trainable=True)
    opt['params_netG'] = params_netG

    # Create files with performance metric statistics
    create_inference_speed_file(inference_time_list, wandb_path, opt)

    print("Done")

if __name__ == "__main__":
    main()
