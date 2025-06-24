import os
import time
import matplotlib.pyplot as plt
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from omegaconf import OmegaConf
import torch
from monai.data import SmartCacheDataset, DataLoader

import config
from utils.load_options import save_yaml, init_options


# def test_plot(train_batch):
#     size_hr = train_batch['H'].shape[-1]
#     size_lr = train_batch['L'].shape[-1]
#     plt.figure()
#     batch_size = len(train_batch['H'])
#     c = 0
#     for i in range(batch_size):
#         plt.subplot(2, batch_size, 1 + c)
#         plt.imshow(train_batch['H'][i, 0, :, :, size_hr//2])
#         plt.subplot(2, batch_size, 2 + c)
#         plt.imshow(train_batch['L'][i, 0, :, :, size_lr//2])
#         plt.show()
#         c += 1

def test_plot(train_batch):
    size_hr = train_batch['H'].shape[-1]
    size_lr = train_batch['L'].shape[-1]
    batch_size = len(train_batch['H'])

    plt.figure(figsize=(3 * batch_size, 6))  # wider figure for multiple samples

    for i in range(batch_size):
        # Plot HR slice
        plt.subplot(2, batch_size, i + 1)
        plt.imshow(train_batch['H'][i, 0, :, :, size_hr // 2], cmap='gray')
        plt.title(f'HR #{i}')
        plt.axis('off')

        # Plot LR slice
        plt.subplot(2, batch_size, batch_size + i + 1)
        plt.imshow(train_batch['L'][i, 0, :, :, size_lr // 2], cmap='gray')
        plt.title(f'LR #{i}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    # To use, add this line to the training loop:
    # test_plot(train_batch)  # Uncomment to visualize training batches


def train_model(model, opt, iterations, validation_iterations, train_loader, test_loader, print_status=True):
    """
    Train function for universal SR model.
    :param model:
    :param opt_dataset:
    :param iterations:
    :param train_loader:
    :param test_loader:
    :param print_status:
    :return:
    """

    current_step = model.last_iteration if opt['train_mode'] == "resume" else 0

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 matrix multiplications on Ampere GPUs and later
    torch.backends.cudnn.allow_tf32 = True  # Allow TF32 operations on Ampere GPUs and later

    n_train_batches = len(train_loader)  # number of batches per epoch in the training dataset
    n_test_batches = len(test_loader)   # number of batches per epoch in the test dataset

    checkpoint_print = opt['train_opt']['checkpoint_print']
    checkpoint_save = opt['train_opt']['checkpoint_save']
    checkpoint_test = opt['train_opt']['checkpoint_test']
    if checkpoint_print == 0: checkpoint_print = n_train_batches
    if checkpoint_save == 0: checkpoint_save = n_train_batches
    if checkpoint_test == 0: checkpoint_test = n_train_batches

    start_time = time.time()
    save_time = opt['save_time']

    while current_step < iterations:
        idx_train = 0

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        for batch_idx, train_batch in enumerate(train_loader):

            current_step += 1
            idx_train += 1

            # -------------------------------
            # 1) load batches of HR and LR images onto GPU and feed to model
            # -------------------------------
            model.feed_data(train_batch)

            # -------------------------------
            # 2) optimize parameters
            # -------------------------------
            update = True if (current_step % len(train_loader) == 0) else False  # force parameter update on last batch
            if model.mixed_precision is not None:
                model.optimize_parameters_amp(current_step, update=update)
            else:
                model.optimize_parameters(current_step, update=update)

            # -------------------------------
            # 3) update learning rate
            # -------------------------------
            if model.update:
                model.update_learning_rate()  # removed current step here and moved line to after optimizer.step()

            # -------------------------------
            # 4) print training information
            # -------------------------------
            if current_step % checkpoint_print == 0 and opt['rank'] == 0:
                print("Iteration %d / %d" % (current_step, iterations))

            # -------------------------------
            # 5) record training log at the end of every epoch
            # -------------------------------
            if current_step % len(train_loader) == 0 and opt['rank'] == 0:
                model.record_train_log(current_step, idx_train)

            # -------------------------------
            # 6) save model
            # -------------------------------
            elapsed_time = (time.time() - start_time) / 3600  # Elapsed time in hours
            if ((current_step % checkpoint_save == 0) or (elapsed_time > save_time)) and opt['rank'] == 0:
                save_time = torch.inf  # disable time-based model saving
                print("SAVING NETWORK PARAMETERS AT STEP %d / %d" % (current_step, iterations))
                # logger.info('Saving the model.')
                model.save(current_step)

            # -------------------------------
            # 8) testing / validation
            # -------------------------------
            if current_step % checkpoint_test == 0 and opt['rank'] == 0:
                # Set model(s) to evaluation model
                model.set_eval_mode()

                torch.cuda.empty_cache()
                idx_test = 0
                with torch.inference_mode():
                    while idx_test < validation_iterations:
                        for batch_idx, test_batch in enumerate(test_loader):
                            idx_test += 1

                            if idx_test % 100 == 0 and opt['rank'] == 0:
                                print("Validation iteration %d / %d" % (idx_test, validation_iterations))

                            # -------------------------------
                            # 9) load batches of HR and LR images onto GPU and feed to model
                            # -------------------------------

                            model.feed_data(test_batch)

                            # -------------------------------
                            # 10) Test model using inference mode
                            # -------------------------------
                            if model.mixed_precision is not None:
                                model.validation_amp()
                            else:
                                model.validation()
                    # -------------------------------
                    # 12) Record early stopping
                    # -------------------------------
                    model.early_stopping(current_step, idx_test)

                    # -------------------------------
                    # 11) calculate and record validation log
                    # -------------------------------
                    model.record_test_log(current_step, idx_test)

                # -------------------------------
                # 13) Save visual comparison
                # -------------------------------
                print("Saving comparison: test image")

                visuals = model.current_visuals()
                model.log_comparison_image(visuals, current_step)

                # Update test_loader with new samples if SmartCacheDataset
                if type(test_loader.dataset) == SmartCacheDataset:
                    test_loader.dataset.update_cache()

                # Set model(s) to train model
                model.set_train_mode()

        # Update train_loader with new samples if SmartCacheDataset
        if type(train_loader.dataset) == SmartCacheDataset:
            train_loader.dataset.update_cache()

        # -------------------------------
        # 7) Print maximum reserved GPU memory
        # -------------------------------

        if print_status and opt['rank'] == 0:
            print(f"Iteration: {current_step}/{iterations}")

            max_memory_reserved = torch.cuda.max_memory_reserved()
            print("Max memory reserved: %0.3f Gb / %0.3f Gb" % (max_memory_reserved / 10 ** 9, opt['total_gpu_mem']))

        # Trigger early stopping
        if model.early_stop:
            print("EARLY STOPPING")
            break

    # Shutdown SmartCacheDatasets
    if type(train_loader.dataset) == SmartCacheDataset:
        train_loader.dataset.shutdown()
        test_loader.dataset.shutdown()

    if opt['rank'] == 0:

        # Save final model
        if opt['save_model']:
            print("SAVING NETWORK PARAMETERS AFTER TRAINING")
            model.save(current_step)

        # Close WandB run
        model.run.finish()

    print("Training finished")

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

    # Define universal SR model using the KAIR define_Model framework
    from models.select_model import define_Model
    model = define_Model(opt)

    # Define wandb run
    model.define_wandb_run()

    # Run initialization of model for training
    model.init_train()

    # Save copy of options file in wandb directory
    save_yaml(opt, wandb_path=model.run.dir)

    # Define number of iterations and validation iterations
    iterations = opt['train_opt']['iterations'] * opt['train_opt']['num_accum_steps_G']
    validation_iterations = opt['train_opt']['validation_iterations']

    if opt['train_mode'] == "resume":
        assert iterations > model.last_iteration, "Total iterations >= start iterations. No iterations to resume training."

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
                                         pin_memory=dataloader_params_train['pin_memory'])

    test_loader = DataLoader(test_dataset,
                                        batch_size=dataloader_params_test['dataloader_batch_size'],
                                        shuffle=dataloader_params_test['dataloader_shuffle'],
                                        num_workers=dataloader_params_test['num_load_workers'],
                                        persistent_workers=dataloader_params_test['persist_workers'],
                                        pin_memory=dataloader_params_test['pin_memory'])

    # Train model
    if opt['dataset_opt']['dataset_type'] == "MonaiSmartCacheDataset":
        train_dataset.start()
        test_dataset.start()

    time_start = time.time()

    if opt['run_profile']:
        from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     profile_memory=True,
                     record_shapes=False,
                     on_trace_ready=tensorboard_trace_handler("./profiles")) as prof:
            out_dict = train_model(model, opt, iterations, validation_iterations,  train_loader, test_loader, print_status=True)
    else:
        out_dict = train_model(model, opt, iterations, validation_iterations, train_loader, test_loader, print_status=True)

    time_end = time.time()
    print("Time taken to train: ", time_end - time_start)

    print("Done")


if __name__ == "__main__":
    main()

    # remove any .log files in root directory
    for file in os.listdir(config.ROOT_DIR):
        if file.endswith(".log"):
            os.remove(os.path.join(config.ROOT_DIR, file))
