import os
import time
import numpy as np
import matplotlib.pyplot as plt
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from omegaconf import OmegaConf
import torch
from monai.data import SmartCacheDataset, DataLoader
from utils.utils_3D_image import crop_center

import config
from utils.load_options import init_options, set_seed

def test_plot(train_batch):
    size_hr = train_batch['H'].shape[-1]
    size_lr = train_batch['L'].shape[-1]
    batch_size = len(train_batch['H'])

    plt.figure(figsize=(3 * batch_size, 6))  # wider figure for multiple samples

    for i in range(batch_size):
        # Plot HR slice
        plt.subplot(2, batch_size, i + 1)
        plt.imshow(train_batch['H'][i, 0, :, :, size_hr // 2], cmap='gray', vmin=0.0, vmax=1.0)
        plt.title(f'HR #{i}')
        plt.axis('off')

        # Plot LR slice
        plt.subplot(2, batch_size, batch_size + i + 1)
        plt.imshow(train_batch['L'][i, 0, :, :, size_lr // 2], cmap='gray', vmin=0.0, vmax=1.0)
        plt.title(f'LR #{i}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    # To use, add this line to the training loop:
    # test_plot(train_batch)  # Uncomment to visualize training batches


def save_image_cubes(opt, number_of_cubes, image_dir, train_loader, test_loader, print_status=True):

    current_step = 0
    idx = 0

    n_train_batches = len(train_loader)  # number of batches per epoch in the training dataset
    n_test_batches = len(test_loader)   # number of batches per epoch in the test dataset

    checkpoint_print = opt['train_opt']['checkpoint_print']
    if checkpoint_print == 0: checkpoint_print = n_train_batches

    # Create directories for saving image cubes if they do not exist
    hr_dir = os.path.join(image_dir, f"HR/")
    lr_dir = os.path.join(image_dir, f"LR/")
    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(lr_dir, exist_ok=True)

    while idx < number_of_cubes:

        for batch_idx, test_batch in enumerate(test_loader):

            current_step += 1
            test_plot(test_batch)

            print("Cube idx: ", idx)
            for i in range(len(test_batch['L'])):
                HR_cube = test_batch['H'][i].float().detach().cpu().numpy()
                LR_cube_32 = crop_center(test_batch['L'], center_size=32)[i].float().detach().cpu().numpy()
                LR_cube_64 = crop_center(test_batch['L'], center_size=64)[i].float().detach().cpu().numpy()
                LR_cube_128 = test_batch['L'][i].float().detach().cpu().numpy()

                if LR_cube_128.sum() / np.prod(LR_cube_128.shape) < 0.1:
                    continue

                # Create the filename with the format "cube_xxx.npy"
                np.save(os.path.join(os.path.join(image_dir, f"HR/"), f"cube_{idx:03d}.npy"), HR_cube)
                np.save(os.path.join(os.path.join(image_dir, f"LR/"), f"cube_32_{idx:03d}.npy"), LR_cube_32)
                np.save(os.path.join(os.path.join(image_dir, f"LR/"), f"cube_64_{idx:03d}.npy"), LR_cube_64)
                np.save(os.path.join(os.path.join(image_dir, f"LR/"), f"cube_128_{idx:03d}.npy"), LR_cube_128)

                idx += 1

                # -------------------------------
                # 4) print training information
                # -------------------------------
                if current_step % checkpoint_print == 0 and opt['rank'] == 0:
                    print("Iteration %d / %d" % (current_step, number_of_cubes))

                if idx >= number_of_cubes:
                    print("Cube generation finished")
                    return 0

    return 0


@hydra.main(version_base=None, config_path="options", config_name=config.MODEL_ARCHITECTURE)
def main(opt: DictConfig):

    # Returns None if no arguments parsed, as when run in PyCharm
    #args = parse_arguments()

    # Enable changes to the configuration
    OmegaConf.set_struct(opt, False)

    # Initialize options
    opt_path = os.path.join(config.ROOT_DIR, 'options', f'{HydraConfig.get().job.config_name}.yaml')
    init_options(opt, opt_path)

    print(f"RUNNING TRAIN MODE: {opt['train_mode'].upper()}")

    print("Cuda is available", torch.cuda.is_available())
    print("Cuda device count", torch.cuda.device_count())
    print("Cuda current device", torch.cuda.current_device())
    print("Cuda device name", torch.cuda.get_device_name(0))

    # Reset seed
    set_seed(opt)

    # Define dataloaders
    from data.select_dataset import define_Dataset
    train_dataset, test_dataset, baseline_dataset = define_Dataset(opt, return_filepaths=False)  # optional to have baseline dataloader as final output

    dataloader_params_train = opt['dataset_opt']['train_dataloader_params']
    dataloader_params_test = opt['dataset_opt']['test_dataloader_params']

    train_loader = DataLoader(train_dataset,
                              batch_size=dataloader_params_train['dataloader_batch_size'],
                              shuffle=dataloader_params_train['dataloader_shuffle'],
                              num_workers=dataloader_params_train['num_load_workers'],
                              persistent_workers=dataloader_params_train['persist_workers'],
                              pin_memory=dataloader_params_train['pin_memory'],
                              drop_last=True)

    test_loader = DataLoader(test_dataset,
                             batch_size=dataloader_params_test['dataloader_batch_size'],
                             shuffle=dataloader_params_test['dataloader_shuffle'],
                             num_workers=dataloader_params_test['num_load_workers'],
                             persistent_workers=dataloader_params_test['persist_workers'],
                             pin_memory=dataloader_params_test['pin_memory'],
                             drop_last=True)

    # Create directory for test patch comparisons
    if "datasets" not in opt['dataset_opt']:
        image_dir = os.path.join("saved_image_cubes", opt['dataset_opt']['name'])
    else:
        image_dir = os.path.join("saved_image_cubes", list(opt['dataset_opt']['datasets'])[0])
        print(f"Using {opt['dataset_opt']['datasets']} for saving image cubes.")
        print(f"Synthetic: {opt['dataset_opt']['synthetic']}")

    print("Saving image comparisons to:", image_dir)

    if opt['dataset_opt']['dataset_type'] == "MonaiSmartCacheDataset":
        train_dataset.start()
        test_dataset.start()

    number_of_cubes = 100  # Number of cubes to save

    time_start = time.time()

    if opt['run_profile']:
        from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     profile_memory=True,
                     record_shapes=False,
                     on_trace_ready=tensorboard_trace_handler("./profiles")) as prof:
            out_dict = save_image_cubes(opt, number_of_cubes, image_dir, train_loader, test_loader, print_status=True)
    else:
        out_dict = save_image_cubes(opt, number_of_cubes, image_dir, train_loader, test_loader, print_status=True)
    time_end = time.time()
    print("Time taken to train: ", time_end - time_start)

    print("Done")


if __name__ == "__main__":
    main()

    time.sleep(5)  # sleep before attempting removing .log files
    # remove any .log files in root directory
    for file in os.listdir(config.ROOT_DIR):
        if file.endswith(".log"):
            try:
                os.remove(os.path.join(config.ROOT_DIR, file))
            except Exception as e:
                print(f"Could not remove file {file}: {e}")
