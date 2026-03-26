import glob
import os
from collections import OrderedDict
from omegaconf import OmegaConf

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.optim import Adam, AdamW
from torch.optim import lr_scheduler
from torch.nn.parallel import DistributedDataParallel

import config
from models.model_base import ModelBase
from models.select_network import define_G
from utils import utils_3D_image

from utils.utils_dist import get_rank, reduce_sum, reduce_max

from performance_metrics.performance_metrics import compute_performance_metrics, PSNR_3D, SSIM_3D, NRMSE_3D, PSNR_2D, SSIM_2D, NRMSE_2D

class ModelVQVAE(ModelBase):
    """Train with pixel-VGG-GAN loss"""
    def __init__(self, opt, mode='train', data_parallel=True):
        super(ModelVQVAE, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.last_iteration = 0  # last iteration
        self.opt_train = self.opt['train_opt']    # training option
        self.netG = define_G(opt, mode=mode)
        self.netG = self.model_to_device(self.netG, data_parallel=data_parallel, compile=False)

        if opt['rank'] == 0 and mode == 'train':
            print("Number of trainable parameters", utils_3D_image.numel(self.netG, only_trainable=True))

        self.update = False  # Flag for gradient accumulation

        # ------------------------------------
        # define early stopping parameters
        # ------------------------------------
        self.early_stop = False
        self.min_validation_loss = float('inf')
        self.patience = self.opt_train['early_stop_patience']
        self.patience_counter = 0
        self.min_delta = 0

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """
    def set_eval_mode(self):
        self.netG.eval()

    def set_train_mode(self):
        self.netG.train()

    def init_test(self, experiment_id):
        # Loads model based on the ID specified.
        # If there exists several logs using the same ID, will load latest one.
        self.load(experiment_id, mode='test')  # load model
        self.netG.eval()  # set eval mode
        self.define_metrics()  # define metrics
        self.define_mixed_precision()  # enable automatic mixed precision
        self.define_visual_eval()
        # self.log_dict = OrderedDict()          # log

    # ----------------------------------------
    # initialize training
    # -----------------------------
    def init_train(self):
        self.load()                             # load model
        self.netG.train()                       # set training mode,for BN

        self.define_loss()                      # define loss
        self.define_metrics()                   # define metrics

        self.define_optimizer()                 # define optimizer
        self.load_optimizers()                  # load optimizer

        self.define_mixed_precision()           # define mixed precision
        self.load_gradscalers()                 # load gradscaler

        self.define_scheduler()                 # define scheduler
        self.load_schedulers()                  # load scheduler

        self.define_visual_eval()
        #self.log_dict = OrderedDict()          # log

    # ----------------------------------------
    # load pre-trained G and D model
    # ----------------------------------------
    def load(self, experiment_id=None, mode='train'):
        """
        Navigate to appropriate directory using dataset -> wandb -> run ID -> latest
        :param experiment_id: ID of the experiment to load, takes precedence over "pretrained_experiment_id" in config
        :return: None
        """
        pretrained_experiment_id_G = self.opt['path']['pretrained_experiment_id'] if experiment_id is None else experiment_id

        if mode == 'train':
            if self.opt['train_mode'] == 'scratch':
                pretrained_experiment_id_G = None
            elif self.opt['train_mode'] == 'finetune' or self.opt['train_mode'] == 'resume':
                assert pretrained_experiment_id_G is not None, f"Pretrained experiment ID must be specified for training mode: {self.opt['train_mode']}."
        elif mode == 'test':
            assert experiment_id is not None, f"Experiment ID must be specified for loading in test mode."

        if pretrained_experiment_id_G is not None:
            opt_files = glob.glob(os.path.join(config.ROOT_DIR, "logs/", "*/", "wandb/", "*" + pretrained_experiment_id_G, "files/saved_models/*G.h5"))
            opt_files.sort(key=os.path.getmtime, reverse=True)
            try:
                G_file = opt_files[0]  # Get latest modified directory with the specified experiment_id
            except:
                print("An exception occurred: No G file found, skipping loading of model...")
            else:
                if self.opt['rank'] == 0:
                    print('Loading pretrained model for G [{:s}] ...'.format(os.sep.join(os.path.normpath(G_file).split(os.sep)[-4:])))
                self.load_network(G_file, self.netG, strict=self.opt_train['G_param_strict'])
                self.last_iteration = int(os.path.basename(G_file).split('_')[0])

    # ----------------------------------------
    # load optimizerG and optimizerD
    # ----------------------------------------
    def load_optimizers(self, experiment_id=None):

        pretrained_experiment_id_G = self.opt['path']['pretrained_experiment_id'] if experiment_id is None else experiment_id

        if self.opt['train_mode'] == 'scratch':
            pretrained_experiment_id_G = None  # Do not load optimizer for training mode: 'scratch'

        elif self.opt['train_mode'] == 'finetune':
            pretrained_experiment_id_G = None  # Do not load optimizer for training mode: 'finetune'
            # if self.opt['train_opt']['G_optimizer_reuse']:
            #     assert pretrained_experiment_id_G is not None, f"Pretrained experiment ID must be specified for training mode: {self.opt['train_mode']} when reusing optimizer states."
            #     if self.model_param_mismatch:
            #         print("Warning: Model parameter mismatch detected, skipping loading of optimizer states...")
            #         pretrained_experiment_id_G = None  # Do not load optimizer if model parameters do not match

        elif self.opt['train_mode'] == 'resume':
            assert pretrained_experiment_id_G is not None, f"Pretrained experiment ID must be specified for training mode: {self.opt['train_mode']}."
            self.opt['train_opt']['G_optimizer_reuse'] = True  # Always load optimizer for training mode: 'resume'

        if pretrained_experiment_id_G is not None and self.opt['train_opt']['G_optimizer_reuse']:

            opt_files = glob.glob(os.path.join(config.ROOT_DIR, "logs/", "*/", "wandb/", "*" + pretrained_experiment_id_G, "files/saved_optimizers/*optimizerG.h5"))
            opt_files.sort(key=os.path.getmtime, reverse=True)
            try:
                G_opt_file = opt_files[0]  # Get latest modified directory with the specified experiment_id
            except:
                print("An exception occurred: No G optimizer found, skipping loading of optimizer...")
            else:
                if self.opt['rank'] == 0:
                    print('Loading optimizer states for G [{:s}] ...'.format(os.sep.join(os.path.normpath(G_opt_file).split(os.sep)[-4:])))
                self.load_optimizer(G_opt_file, self.G_optimizer)

    # ----------------------------------------
    # load schedulerG and schedulerD
    # ----------------------------------------
    def load_schedulers(self, experiment_id=None):

        pretrained_experiment_id_G = self.opt['path']['pretrained_experiment_id'] if experiment_id is None else experiment_id

        if self.opt['train_mode'] == 'scratch':
            pretrained_experiment_id_G = None

        elif self.opt['train_mode'] == 'finetune':
            pretrained_experiment_id_G = None

        elif self.opt['train_mode'] == 'resume':
            assert pretrained_experiment_id_G is not None, f"Pretrained experiment ID must be specified for training mode: {self.opt['train_mode']}."

        if pretrained_experiment_id_G is not None:
            opt_files = glob.glob(os.path.join(config.ROOT_DIR, "logs/", "*/", "wandb/", "*" + pretrained_experiment_id_G, "files/saved_schedulers/*schedulerG.h5"))
            opt_files.sort(key=os.path.getmtime, reverse=True)
            try:
                G_scheduler_file = opt_files[0]  # Get latest modified directory with the specified experiment_id
            except:
                print("An exception occurred: No G schedulers found, skipping loading of schedulers...")
            else:
                if self.opt['rank'] == 0:
                    print('Loading scheduler states for G [{:s}] ...'.format(os.sep.join(os.path.normpath(G_scheduler_file).split(os.sep)[-4:])))
                self.load_scheduler(G_scheduler_file, self.schedulers[0])

    def load_gradscalers(self, experiment_id=None):

        pretrained_experiment_id_G = self.opt['path']['pretrained_experiment_id'] if experiment_id is None else experiment_id

        if self.opt['train_mode'] == 'scratch':
            pretrained_experiment_id_G = None

        elif self.opt['train_mode'] == 'finetune':
            pretrained_experiment_id_G = None

        elif self.opt['train_mode'] == 'resume':
            assert pretrained_experiment_id_G is not None, f"Pretrained experiment ID must be specified for training mode: {self.opt['train_mode']}."

        if pretrained_experiment_id_G is not None:
            opt_files = glob.glob(os.path.join(config.ROOT_DIR, "logs/", "*/", "wandb/", "*" + pretrained_experiment_id_G, "files/saved_gradscalers/*gradscalerG.h5"))
            opt_files.sort(key=os.path.getmtime, reverse=True)
            try:
                G_gradscaler_file = opt_files[0]  # Get latest modified directory with the specified experiment_id
            except:
                print("An exception occurred: No G gradscaler found, skipping loading of gradscalers...")
            else:
                if self.opt['rank'] == 0:
                    print('Loading gradscaler states for G [{:s}] ...'.format(os.sep.join(os.path.normpath(G_gradscaler_file).split(os.sep)[-4:])))
                self.load_scheduler(G_gradscaler_file, self.gen_scaler)


    # ----------------------------------------
    # save model / optimizer (optional)
    # ----------------------------------------
    def save(self, iter_label):
        # WandB save directory
        model_save_dir = os.path.join(self.run.dir, "saved_models")
        self.save_network(model_save_dir, self.netG, 'G', iter_label)

        opt_save_dir = os.path.join(self.run.dir, "saved_optimizers")
        self.save_optimizer(opt_save_dir, self.G_optimizer, 'optimizerG', iter_label)

        scheduler_save_dir = os.path.join(self.run.dir, "saved_schedulers")
        self.save_scheduler(scheduler_save_dir, self.schedulers[0], 'schedulerG', iter_label)

        if self.mixed_precision is not None:
            gradscaler_save_dir = os.path.join(self.run.dir, "saved_gradscalers")
            self.save_gradscaler(gradscaler_save_dir, self.gen_scaler, 'gradscalerG', iter_label)

    def define_wandb_run(self):

        ######### INITIALIZE WEIGHTS AND BIASES RUN #########

        self.run = wandb.init(
            mode=self.opt["wandb_mode"],
            # set the wandb project where this run will be logged
            entity=self.opt['wandb_entity'],
            project=self.opt['wandb_project'],
            name=self.opt['run_name'],
            id=self.opt['experiment_id'],
            notes=self.opt['note'],
            dir="logs/" + self.opt['dataset_opt']['name'],

            # track hyperparameters and run metadata
            config={
                "iterations": self.opt['train_opt']['iterations'],
                "G_learning_rate": self.opt['train_opt']['G_optimizer_lr'],
                "batch_size": self.opt['dataset_opt']['train_dataloader_params']['dataloader_batch_size'],
                "dataset": self.opt['dataset_opt']['name'],
                "up_factor": self.opt['up_factor'],
                "architecture": self.opt['model_opt']['model_architecture'],
            })

        ######### CREATE DIRECTORY FOR SAVED MODELS, OPTIMIZERS, ECT. #########
        os.mkdir(os.path.join(wandb.run.dir, "saved_models"))
        os.mkdir(os.path.join(wandb.run.dir, "saved_optimizers"))
        os.mkdir(os.path.join(wandb.run.dir, "saved_schedulers"))
        os.mkdir(os.path.join(wandb.run.dir, "saved_gradscalers"))

        # Create model artifacts for logging of files
        # Look at this link for how to construct an artifact with a more neat file structure:
        # https://docs.wandb.ai/guides/artifacts/construct-an-artifact
        self.wandb_config = wandb.config

        self.model_artifact_G = wandb.Artifact(
            "Generator", type=self.opt['model_opt']['netG']['net_type'],
            description=self.opt['model_opt']['netG']['description'],
            metadata=OmegaConf.to_container(self.opt['model_opt']['netG'], resolve=True)
        )

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):

        # Define losses for VQVAE
        self.G_train_loss = 0.0
        self.G_valid_loss = 0.0
        self.G_train_grad_norm = 0.0
        self.G_valid_grad_norm = 0.0


    # ----------------------------------------
    # define metrics
    # ----------------------------------------
    def define_metrics(self):

        # Define losses for G and D
        self.metric_fn_dict = {}
        self.metric_val_dict = {}

        if "psnr" in self.opt_train['performance_metrics']:
            self.metric_val_dict["psnr"] = 0.0
            self.metric_fn_dict["psnr"] = PSNR_3D() if self.opt['input_type'] == '3D' else PSNR_2D()
        if "ssim" in self.opt_train['performance_metrics']:
            self.metric_val_dict["ssim"] = 0.0
            self.metric_fn_dict["ssim"] = SSIM_3D() if self.opt['input_type'] == '3D' else SSIM_2D()
        if "nrmse" in self.opt_train['performance_metrics']:
            self.metric_val_dict["nrmse"] = 0.0
            self.metric_fn_dict["nrmse"] = NRMSE_3D() if self.opt['input_type'] == '3D' else NRMSE_2D()


    # ----------------------------------------
    # define optimizer, G and D
    # ----------------------------------------
    def define_optimizer(self):

        # Testing gradient accumulation
        self.G_accum_count = 0
        self.num_accum_steps_G = self.opt_train['num_accum_steps_G']

        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))

        if self.opt_train['G_optimizer_type'] == 'adam':
            self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'], weight_decay=self.opt_train['G_optimizer_wd'], betas=(0.9, 0.999))
        elif self.opt_train['G_optimizer_type'] == 'adamw':
            self.G_optimizer = AdamW(G_optim_params, lr=self.opt_train['G_optimizer_lr'], weight_decay=self.opt_train['G_optimizer_wd'], betas=(0.9, 0.999))
        else:
            raise NotImplementedError('optimizer [{:s}] is not implemented.'.format(self.opt_train['G_optimizer_type']))

    # ----------------------------------------
    # define gradient scaler for G and D
    # ----------------------------------------
    def define_gradscaler(self):
        self.gen_scaler = torch.amp.GradScaler("cuda")

    # ----------------------------------------
    # Set working precision for use with PyTorch AMP
    # ----------------------------------------
    def define_mixed_precision(self):
        if self.opt_train['mixed_precision'] == "FP16":
            self.mixed_precision = torch.float16
            self.define_gradscaler()

        elif self.opt_train['mixed_precision'] == "FP32":
            self.mixed_precision = torch.float32
            self.define_gradscaler()

        elif self.opt_train['mixed_precision'] == "BF16":
            self.mixed_precision = torch.bfloat16
            self.define_gradscaler()

        else:
            self.mixed_precision = None

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    #def define_scheduler(self):
    #    self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
    #                                                    self.opt_train['G_scheduler_milestones'],
    #                                                    self.opt_train['G_scheduler_gamma']
    #                                                    ))

    def define_scheduler(self):
        # MultiStep scheduler
        multistep_scheduler = lr_scheduler.MultiStepLR(
            self.G_optimizer,
            milestones=self.opt_train['G_scheduler_milestones'],
            gamma=self.opt_train['G_scheduler_gamma']
        )

        if self.opt_train.get('G_warmup_steps', 0) > 0:
            # Linear warmup scheduler
            warmup_scheduler = lr_scheduler.LinearLR(
                self.G_optimizer,
                start_factor=1e-8,  # or 0.0 if you want to start from 0
                end_factor=1.0,
                total_iters=self.opt_train['G_warmup_steps']
            )

            # Combine them with SequentialLR
            scheduler = lr_scheduler.SequentialLR(
                self.G_optimizer,
                schedulers=[warmup_scheduler, multistep_scheduler],
                milestones=[self.opt_train['G_warmup_steps']]  # when to switch
            )
        else:
            # Fallback to MultiStepLR only
            scheduler = multistep_scheduler

        self.schedulers.append(scheduler)

    def define_visual_eval(self):

        if self.opt['input_type'] == '2D':
            from utils.utils_2D_image import ImageComparisonTool2D as comparison_tool

        elif self.opt['input_type'] == '3D':
            from utils.utils_3D_image import ImageComparisonTool3D as comparison_tool

        self.comparison_tool = comparison_tool(patch_size_hr=self.opt['dataset_opt']['patch_size_hr'],
                                               upscaling_methods=["tio_nearest", "tio_linear"],
                                               unnorm=self.opt['dataset_opt']['norm_type'] == 'znormalization',
                                               div_max=self.opt['dataset_opt']['norm_type'] == 'znormalization',
                                               out_dtype=np.uint8)

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data):
        self.L = data['L'].as_tensor().to(self.device, non_blocking=True)
        self.H = data['H'].as_tensor().to(self.device, non_blocking=True)

    def vq_forward(self):  # Returns reconstruction E and VQ loss given H, used for training
        self.E, self.vq_loss = self.netG(self.H)

    def netG_forward(self):  # Returns reconstruction E given L, only used for inference testing
        self.E, _ = self.netG(self.L)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters_amp(self, current_step, update=False):

        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            self.vq_forward()  # Reconstruct H and compute VQ loss

            self.gen_loss = self.vq_loss + F.mse_loss(self.E, self.H)  # VQ loss + reconstruction loss
            self.gen_loss = self.gen_loss / self.num_accum_steps_G  # Scale loss by number of accumulation steps

        self.G_train_loss = self.gen_loss  # Add VAE training loss to total loss

        if self.opt['rank'] == 0:
            print("G train loss:", self.G_train_loss.item())

        # -------------------------------------------------
        # update logic for gradient accumulation
        # -------------------------------------------------
        self.update = ((self.G_accum_count + 1) % self.num_accum_steps_G) == 0 or update

        # -------------------------------------------------
        # DDP optimization: skip gradient sync during accumulation
        # -------------------------------------------------
        if not self.update:
            if isinstance(self.netD, DistributedDataParallel):  # Check if the model is DDP and supports no_sync
                with self.netG.no_sync():  # avoid expensive all-reduce
                    self.gen_scaler.scale(self.gen_loss).backward()
            else:
                self.gen_scaler.scale(self.gen_loss).backward()
        else:
            # sync gradients
            self.gen_scaler.scale(self.gen_loss).backward()

        # ------------------------------------
        # Optimizer step
        # ------------------------------------
        if self.update:  # Gradient acculumation
            # ------------------------------------
            # clip_grad on G
            # ------------------------------------
            G_clipgrad_max = self.opt_train['G_optimizer_clipgrad']
            if G_clipgrad_max > 0:
                # Unscales the gradients of optimizer's assigned params in-place if AMP is enabled
                self.gen_scaler.unscale_(self.G_optimizer)
                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                self.G_train_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.netG.parameters(),
                    max_norm=G_clipgrad_max,
                    norm_type=2
                )

            self.gen_scaler.step(self.G_optimizer)  # update weights
            self.gen_scaler.update()
            self.G_optimizer.zero_grad()  # set parameter gradients to zero

            # Reset gradient accumulation count
            self.G_accum_count = 0

        else:  # Update gradient accumulation count
            self.G_accum_count += 1


    def record_train_log(self, current_step):
        # ------------------------------------
        # record log
        # ------------------------------------

        # Record training losses using wandb
        loss = self.G_train_loss.item() * self.num_accum_steps_G
        self.run.log({"step": current_step, "G_train_loss": loss})

        grad_norm = self.G_train_grad_norm.item() * self.num_accum_steps_G
        self.run.log({"step": current_step, "G_train_grad_norm": grad_norm})

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    def record_avg_train_log(self, current_step, idx_train):
        # ------------------------------------
        # record log
        # ------------------------------------

        # Record training losses using wandb
        avg_loss = (self.G_train_loss.item() / idx_train) * self.num_accum_steps_G
        self.run.log({"step": current_step, "G_train_loss": avg_loss})

        # Reset training losses
        self.G_train_loss = 0.0

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    def early_stopping(self, current_step, idx_train):
        validation_loss = self.G_valid_loss / idx_train  # calculate average validation loss

        if validation_loss.item() < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.patience_counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.patience_counter += 1
            if (self.patience_counter >= self.patience) and current_step > 75000:
                self.early_stop = True

        early_stop_tensor = torch.tensor(int(self.early_stop), device=self.device)
        early_stop_tensor = reduce_max(early_stop_tensor)  # Sync early stop decision across GPUs
        self.early_stop = early_stop_tensor.item() == 1  # If any GPU has early_stop, set early_stop for all


    def record_test_log(self, idx_test):

        idx_tensor = torch.tensor(idx_test, device=self.device)
        global_idx_tensor = reduce_sum(idx_tensor)

        # ------------------------------------
        # Reduce validation metrics across GPUs
        # ------------------------------------
        for key, value in self.metric_val_dict.items():
            metric_tensor = torch.tensor(float(value), device=self.device)
            global_average = reduce_sum(metric_tensor) / global_idx_tensor

            if get_rank() == 0:  # Only rank 0 logs
                metric_name = "Average " + key
                self.run.log({metric_name: global_average.item()})
                print(metric_name, global_average.item())

            # Reset local metric accumulator
            self.metric_val_dict[key] = 0.0

        # ------------------------------------
        # Reduce validation loss across GPUs
        # ------------------------------------
        loss_tensor = torch.tensor(float(self.G_valid_loss), device=self.device)
        global_valid_loss = reduce_sum(loss_tensor) / global_idx_tensor

        if get_rank() == 0:  # Only rank 0 logs
            self.run.log({"G_valid_loss": global_valid_loss.item()})
            print("G_valid_loss", global_valid_loss.item())

        # Reset validation loss
        self.G_valid_loss = 0.0


    # ----------------------------------------
    # test and inference
    # ----------------------------------------

    def validation(self):

        self.vq_forward()  # Reconstruct H and compute VQ loss

        self.gen_loss = self.vq_loss + F.mse_loss(self.E, self.H)  # VQ loss + reconstruction loss

        # Add generator validation loss to total loss
        self.G_valid_loss += self.gen_loss

        # Compute performance metrics
        rescale_images = True if self.opt['dataset_opt']['norm_type'] == "znormalization" else False
        compute_performance_metrics(self.E, self.H, self.metric_fn_dict, self.metric_val_dict, rescale_images)


    def validation_amp(self):

        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            self.vq_forward()  # Reconstruct H and compute VQ loss

            self.gen_loss = self.vq_loss + F.mse_loss(self.E, self.H)  # VQ loss + reconstruction loss

        # Add generator validation loss to total loss
        self.G_valid_loss += self.gen_loss

        rescale_images = True if self.opt['dataset_opt']['norm_type'] == "znormalization" else False
        compute_performance_metrics(self.E, self.H, self.metric_fn_dict, self.metric_val_dict, rescale_images)


    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H images
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()

        roi = int(self.opt['dataset_opt']['patch_size_hr'] / self.opt['up_factor'])
        if self.opt['dataset_opt']['patch_size'] > roi:
            out_dict['L'] = utils_3D_image.crop_center(self.L, center_size=roi).detach()[0].float().cpu()
        else:
            out_dict['L'] = self.L.detach()[0].float().cpu()

        out_dict['E'] = self.E.detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict

    def log_comparison_image(self, img_dict, current_step):

        grid_image = self.comparison_tool.get_comparison_image(img_dict)
        figure_string = "SR comparison: %s, step %d, %dx upscaling" % (self.opt['model_opt']['model_architecture'], current_step, self.opt['up_factor'])

        if self.opt['run_type'] == "HOME PC":
            height, width = grid_image.shape[:2]
            plt.figure(figsize=(4 * width / 100, 4 * height / 100), dpi=100)
            plt.imshow(grid_image, vmin=0, vmax=255)
            plt.title(figure_string)
            plt.xticks([])
            plt.yticks([])
            plt.show()

        wandb.log({"Comparisons training": wandb.Image(grid_image, caption=figure_string, mode="RGB")})  # WandB assumes channel last

    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg
