import torch
import lightning as L
from monai.data import DataLoader

def train_lightning(opt, train_dataset, test_dataset, iterations):

    # Define datasets
    if opt['dist']:
        from data.dataset import DDP_CacheDataset
        train_dataset = DDP_CacheDataset(train_dataset)
        test_dataset = DDP_CacheDataset(test_dataset)

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

    # Define Lightning model

    # Set strategy
    strategy = "auto"
    if torch.cuda.device_count() > 1:
        strategy = 'ddp'

    # Trainer configuration
    trainer = L.Trainer(
        max_steps=iterations,
        precision="bf16-mixed",  # Mixed precision with bfloat16
        gradient_clip_val=opt['train']['G_optimizer_clipgrad'],  # Gradient clipping
        accumulate_grad_batches=opt['train']['num_accum_steps_G'],  # Gradient accumulation
        strategy=strategy,  # DDPStrategy(find_unused_parameters=False),  # DDP Strategy
        devices=1 if not torch.cuda.is_available() else torch.cuda.device_count(),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        benchmark=True,  # set torch.backends.cudnn.benchmark to True
        logger=False,
        use_distributed_sampler=False
    )

    # Train
    trainer.fit(model, train_loader, test_loader)

