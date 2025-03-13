import os
import time

import monai.transforms
import numpy as np
import torch
from monai.data import SmartCacheDataset

import config
from train import crop_context, crop_center
from utils.load_options import load_json, parse_arguments


def save_image_cubes(model, opt, number_of_cubes, image_dir,  train_loader, test_loader, print_status=True):

    current_step = 0

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 matrix multiplications on Ampere GPUs and later
    torch.backends.cudnn.allow_tf32 = True  # Allow TF32 operations on Ampere GPUs and later

    n_train_batches = len(train_loader)  # number of batches per epoch in the training dataset

    checkpoint_print = opt['train']['checkpoint_print']
    checkpoint_save = opt['train']['checkpoint_save']
    checkpoint_test = opt['train']['checkpoint_test']
    if checkpoint_print == 0: checkpoint_print = n_train_batches
    if checkpoint_save == 0: checkpoint_save = n_train_batches
    if checkpoint_test == 0: checkpoint_test = n_train_batches

    start_time = time.time()
    save_time = opt['save_time']

    idx = 0

    # Create directories for saving image cubes if they do not exist
    hr_dir = os.path.join(image_dir, f"HR_{opt['datasets']['degradation_type']}/")
    lr_dir = os.path.join(image_dir, f"LR_{opt['datasets']['degradation_type']}/")
    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(lr_dir, exist_ok=True)

    # -------------------------------
    while current_step < number_of_cubes:

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        for batch_idx, test_batch in enumerate(test_loader):

            current_step += 1
            idx += 1

            # -------------------------------
            # 1) load batches of HR and LR images onto GPU and feed to model
            # -------------------------------
            #if opt['model_architecture'] == "MTVNet" and not opt['datasets']['enable_femur_padding']:
            #    test_batch['H'] = crop_context(test_batch['H'], L=model.opt['netG']['num_levels'], level_ratio=model.opt['netG']['level_ratio'])

            # Save cubes here:
            print("Cube idx: ", idx)
            # Save the tensor as a .npy file
            HR_cube = test_batch['H'][0].float().detach().cpu().numpy()
            LR_cube_32 = crop_center(test_batch['L'], center_size=32)[0].float().detach().cpu().numpy()
            LR_cube_64 = crop_center(test_batch['L'], center_size=64)[0].float().detach().cpu().numpy()
            LR_cube_128 = test_batch['L'][0].float().detach().cpu().numpy()
            # Create the filename with the format "cube_xxx.npy"

            np.save(os.path.join(os.path.join(image_dir, f"HR_{opt['datasets']['degradation_type']}/"),  f"cube_{idx:03d}.npy"), HR_cube)
            np.save(os.path.join(os.path.join(image_dir, f"LR_{opt['datasets']['degradation_type']}/"), f"cube_32_{idx:03d}.npy"), LR_cube_32)
            np.save(os.path.join(os.path.join(image_dir, f"LR_{opt['datasets']['degradation_type']}/"), f"cube_64_{idx:03d}.npy"), LR_cube_64)
            np.save(os.path.join(os.path.join(image_dir, f"LR_{opt['datasets']['degradation_type']}/"), f"cube_128_{idx:03d}.npy"), LR_cube_128)

            if idx >= number_of_cubes:
                break

            # -------------------------------
            # 4) print training information
            # -------------------------------
            if current_step % checkpoint_print == 0 and opt['rank'] == 0:
                print("Iteration %d / %d" % (current_step, number_of_cubes))

        # Update train_loader with new samples if SmartCacheDataset
        if type(train_loader.dataset) == SmartCacheDataset:
            train_dataset.update_cache()

        # -------------------------------
        # 7) Print maximum reserved GPU memory
        # -------------------------------
        max_memory_allocated = torch.cuda.max_memory_allocated()
        max_memory_reserved = torch.cuda.max_memory_reserved()
        print("Maximum memory reserved during training step: %0.3f Gb / %0.3f Gb" % (
        max_memory_reserved / 10 ** 9, opt['total_gpu_mem']))

        if print_status:
            print(f"Iteration: {current_step}/{number_of_cubes}")

    # Shutdown SmartCacheDatasets
    if type(train_loader.dataset) == SmartCacheDataset:
        # stop replacement thread of SmartCache
        train_dataset.shutdown()
        test_dataset.shutdown()

    print("Training finished")

    return 0

if __name__ == "__main__":

    print("GENERATE CUBES")

    print("Cuda is available", torch.cuda.is_available())
    print("Cuda device count", torch.cuda.device_count())
    print("Cuda current device", torch.cuda.current_device())
    print("Cuda device name", torch.cuda.get_device_name(0))

    # Returns None if no arguments parsed, as when run in PyCharm
    args = parse_arguments()

    # Define experiment parameters
    options_file = args.options_file
    print("options_file", options_file)

    # Load default experiment option
    if options_file is None:
        opt_path = os.path.join(config.ROOT_DIR, 'options', f'train_{config.MODEL_ARCHITECTURE}.json')
    else:
        opt_path = os.path.join(config.ROOT_DIR, 'options', options_file)

    # Load options
    opt = load_json(opt_path)

    # Overwrite dataset in options file if specified
    if args.dataset is not None:
        opt['datasets']['name'] = args.dataset

    if args.cluster is not None:
        opt['cluster'] = args.cluster
    else:  # Default is opt['cluster'] = "DTU_HPC"
        opt['cluster'] = "DTU_HPC"

    # Define universal SR model using the KAIR define_Model framework
    from models.select_model import define_Model
    model = define_Model(opt)
    # Run initialization of model for training
    model.init_train()

    from data.select_dataset import define_Dataset
    train_dataset, test_dataset, baseline_dataset = define_Dataset(opt)  # optional to have baseline dataloader as final output


    dataloader_params_train = opt['datasets']['train']['dataloader_params']
    dataloader_params_test = opt['datasets']['test']['dataloader_params']

    train_loader = monai.data.DataLoader(train_dataset,
                                         batch_size=dataloader_params_train['dataloader_batch_size'],
                                         shuffle=dataloader_params_train['dataloader_shuffle'],
                                         num_workers=dataloader_params_train['num_load_workers'],
                                         persistent_workers=dataloader_params_train['persist_workers'],
                                         pin_memory=dataloader_params_train['pin_memory'])

    test_loader = monai.data.DataLoader(test_dataset,
                                        batch_size=dataloader_params_test['dataloader_batch_size'],
                                        shuffle=dataloader_params_test['dataloader_shuffle'],
                                        num_workers=dataloader_params_test['num_load_workers'],
                                        persistent_workers=dataloader_params_test['persist_workers'],
                                        pin_memory=dataloader_params_test['pin_memory'])

    # Create directory for test patch comparisons
    image_dir = os.path.join("saved_image_cubes", opt['datasets']['name'])
    print("Saving image comparisons to:", image_dir)

    if opt['datasets']['dataset_type'] == "MonaiSmartCacheDataset":
        train_dataset.start()
        test_dataset.start()

    number_of_cubes = 50  # Number of cubes to save

    time_start = time.time()
    save_image_cubes(model, opt, number_of_cubes, image_dir,  train_loader, test_loader, print_status=True)
    time_end = time.time()
    print("Time taken: ", time_end - time_start)
    print("Done")