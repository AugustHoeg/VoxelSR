import os
import time

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from omegaconf import OmegaConf
from monai.data import SmartCacheDataset, DataLoader

import config
from utils.load_options import save_yaml, init_options

from torch.profiler import profile, ProfilerActivity
from torch.distributed import init_process_group, destroy_process_group

from train import train_model


@hydra.main(version_base=None, config_path="options", config_name=config.MODEL_ARCHITECTURE)
def main(opt: DictConfig):

    # Returns None if no arguments parsed, as when run in PyCharm
    #args = parse_arguments()

    # Enable changes to the configuration
    OmegaConf.set_struct(opt, False)

    # Initialize options
    opt_path = os.path.join(config.ROOT_DIR, 'options', f'{HydraConfig.get().job.config_name}.yaml')
    init_options(opt, opt_path)

    print(f"RUNNING DISTRIBUTED TRAIN MODE: {opt['train_mode'].upper()}")

    # Define universal SR model using the KAIR define_Model framework
    from models.select_model import define_Model
    model = define_Model(opt)

    # Define wandb run
    if opt['rank'] == 0:
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
    print("OVERRIDE DDP CACHE DATASET")
    opt['datasets']['dataset_type'] = "DDP_CacheDataset"
    # in case of DDP_CacheDataset is used, the dataset is split into parts for each rank
    # meaning we do not need to use DistributedSampler
    train_dataset, test_dataset, baseline_dataset = define_Dataset(opt, return_filepaths=False)

    dataloader_params_train = opt['datasets']['train']['dataloader_params']
    dataloader_params_test = opt['datasets']['test']['dataloader_params']

    if opt['dist'] and (opt['datasets']['dataset_type'] != "DDP_CacheDataset"):
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_dataset,
                                           shuffle=dataloader_params_train['dataloader_shuffle'],
                                           drop_last=True,
                                           seed=opt['train']['manual_seed'])

        test_sampler = DistributedSampler(test_dataset,
                                          shuffle=dataloader_params_test['dataloader_shuffle'],
                                          drop_last=True,
                                          seed=opt['train']['manual_seed'])

        train_loader = DataLoader(train_dataset,
                                             batch_size=dataloader_params_train['dataloader_batch_size'],
                                             shuffle=False,
                                             num_workers=dataloader_params_train['num_load_workers'],
                                             persistent_workers=dataloader_params_train['persist_workers'],
                                             pin_memory=dataloader_params_train['persist_workers'],
                                             sampler=train_sampler)

        test_loader = DataLoader(test_dataset,
                                            batch_size=dataloader_params_test['dataloader_batch_size'],
                                            shuffle=False,
                                            num_workers=dataloader_params_test['num_load_workers'],
                                            persistent_workers=dataloader_params_test['persist_workers'],
                                            pin_memory=dataloader_params_test['pin_memory'],
                                            sampler=test_sampler)

    else:
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
    if opt['datasets']['dataset_type'] == "MonaiSmartCacheDataset":
        train_dataset.start()
        test_dataset.start()

    time_start = time.time()

    if opt['run_profile']:
        from torch.profiler import profile, tensorboard_trace_handler
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     profile_memory=True,
                     record_shapes=False,
                     on_trace_ready=tensorboard_trace_handler("./profiles")) as prof:
            out_dict = train_model(model, opt, iterations, validation_iterations, train_loader, test_loader, print_status=True)
    else:
        out_dict = train_model(model, opt, iterations, validation_iterations, train_loader, test_loader, print_status=True)

    time_end = time.time()
    print("Time taken to train: ", time_end - time_start)

    destroy_process_group()

    print("Done")


if __name__ == "__main__":
    main()

    # Run training:
    # python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 train_dist.py --opt options/train_mDCSRN.json  --dist True
    # or
    # torchrun --nproc_per_node=2 --master_port=29500 train_dist.py --opt options/train_mDCSRN.json --dist True
