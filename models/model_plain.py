import glob
import os
from collections import OrderedDict
from omegaconf import OmegaConf

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.optim import Adam
from torch.optim import lr_scheduler

import config
from loss_functions.loss_functions_simple import compute_generator_loss
from models.model_base import ModelBase
from models.select_network import define_G
from utils import utils_3D_image

from performance_metrics.performance_metrics import compute_performance_metrics, PSNR_3D, SSIM_3D, NRMSE_3D

class ModelPlain(ModelBase):
    """Train with pixel-VGG-GAN loss"""
    def __init__(self, opt, mode='train'):
        super(ModelPlain, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.last_iteration = 0  # last iteration
        self.opt_train = self.opt['train_opt']    # training option
        self.netG = define_G(opt, mode=mode)
        self.netG = self.model_to_device(self.netG)
        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(opt).to(self.device).eval()

        if opt['rank'] == 0:
            print("Number of trainable parameters, G", utils_3D_image.numel(self.netG, only_trainable=True))

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
            opt_files = glob.glob(os.path.join(config.ROOT_DIR + "/logs/" + "/*/" "/wandb/" + "*" + pretrained_experiment_id_G + "/files/saved_models/*G.h5"))
            opt_files.sort(key=os.path.getmtime, reverse=True)
            try:
                G_file = opt_files[0]  # Get latest modified directory with the specified experiment_id
            except:
                print("An exception occurred: No G file found, skipping loading of model...")
            else:
                print('Loading pretrained model for G [{:s}] ...'.format(G_file))
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
            opt_files = glob.glob(os.path.join(config.ROOT_DIR + "/logs/" + "/*/" "/wandb/" + "*" + pretrained_experiment_id_G + "/files/saved_optimizers/*optimizerG.h5"))
            opt_files.sort(key=os.path.getmtime, reverse=True)
            try:
                G_opt_file = opt_files[0]  # Get latest modified directory with the specified experiment_id
            except:
                print("An exception occurred: No G optimizer found, skipping loading of optimizer...")
            else:
                print('Loading optimizer states for G [{:s}] ...'.format(G_opt_file))
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
            opt_files = glob.glob(os.path.join(config.ROOT_DIR + "/logs/" + "/*/" "/wandb/" + "*" + pretrained_experiment_id_G + "/files/saved_schedulers/*schedulerG.h5"))
            opt_files.sort(key=os.path.getmtime, reverse=True)
            try:
                G_scheduler_file = opt_files[0]  # Get latest modified directory with the specified experiment_id
            except:
                print("An exception occurred: No G schedulers found, skipping loading of schedulers...")
            else:
                print('Loading scheduler states for G [{:s}] ...'.format(G_scheduler_file))
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
            opt_files = glob.glob(os.path.join(config.ROOT_DIR + "/logs/" + "/*/" "/wandb/" + "*" + pretrained_experiment_id_G + "/files/saved_gradscalers/*gradscalerG.h5"))
            opt_files.sort(key=os.path.getmtime, reverse=True)
            try:
                G_gradscaler_file = opt_files[0]  # Get latest modified directory with the specified experiment_id
            except:
                print("An exception occurred: No G gradscaler found, skipping loading of gradscalers...")
            else:
                print('Loading gradscaler states for G [{:s}] ...'.format(G_gradscaler_file))
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

        if self.opt_train['E_decay'] > 0:
            self.save_network(model_save_dir, self.netE, 'E', iter_label)


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
            "Generator", type=self.opt['model_opt']['model_architecture'],
            description=self.opt['model_opt']['netG']['description'],
            metadata=OmegaConf.to_container(self.opt['model_opt']['netG'], resolve=True))

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):

        self.loss_fn_dict = {
            "MSE": nn.MSELoss(),
            "L1": nn.L1Loss(),
            "BCE_Logistic": nn.BCEWithLogitsLoss(),
            "BCE": nn.BCELoss()
            #"VGG": VGGLoss(layer_idx=36, device=self.device),
            #"VGG3D": VGGLoss3D(num_parts=2*self.opt['up_factor'], layer_idx=35, loss_func=nn.MSELoss(), device=self.device),
            #"GRAD": GradientLoss3D(kernel='diff', order=1, loss_func=nn.L1Loss(), sigma=None),  # sigma = 0.8,
            #"LAPLACE": LaplacianLoss3D(sigma=1.0, padding='valid', loss_func=nn.L1Loss()),
            #"TV3D": TotalVariationLoss3D(mode="L2"),  # or mode = "sum_of_squares", "L2",
            #"TEXTURE3D": TextureLoss3D(layer_idx=35),
            #"STRUCTURE_TENSOR": StructureLoss3D(sigma=0.5, rho=0.5)
        }

        self.loss_val_dict = self.opt_train['G_loss_weights']

        # Define losses for G and D
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
            self.metric_fn_dict["psnr"] = PSNR_3D()
        if "ssim" in self.opt_train['performance_metrics']:
            self.metric_val_dict["ssim"] = 0.0
            self.metric_fn_dict["ssim"] = SSIM_3D()
        if "nrmse" in self.opt_train['performance_metrics']:
            self.metric_val_dict["nrmse"] = 0.0
            self.metric_fn_dict["nrmse"] = NRMSE_3D()


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
            self.G_optimizer = torch.optim.AdamW(G_optim_params, lr=self.opt_train['G_optimizer_lr'], weight_decay=self.opt_train['G_optimizer_wd'], betas=(0.9, 0.999))
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
    def feed_data(self, data, need_H=True, add_key=None):
        if add_key is not None:
            self.L = data['L'][add_key].to(self.device)
            if need_H:
                self.H = data['H'][add_key].to(self.device)
        elif self.opt['dataset_opt']['dataset_type'] == 'MasterThesisDataset':
            self.L = data[1].to(self.device)
            if need_H:
                self.H = data[0].to(self.device)
        else:
            self.L = data['L'].to(self.device)
            if need_H:
                self.H = data['H'].to(self.device)

    # ----------------------------------------
    # feed L to netG and get E
    # ----------------------------------------
    def netG_forward(self):
        if self.mixed_precision is not None:
            # Evaluate using AMP
            with torch.amp.autocast("cuda", dtype=self.mixed_precision):
                self.E = self.netG(self.L)  # self.L
        else:  # Standard precision
            self.E = self.netG(self.L)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters_amp(self, current_step, update=False):

        # ------------------------------------
        # optimize G
        # ------------------------------------

        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            # Forward G
            self.netG_forward()
            #with torch.cuda.amp.autocast(dtype=torch.float64):
            self.gen_loss = compute_generator_loss(self.H, self.E, self.loss_fn_dict, self.loss_val_dict,None, self.device)
            self.gen_loss = self.gen_loss / self.num_accum_steps_G  # Scale loss by number of accumulation steps

        self.G_train_loss = self.gen_loss  # Add generator training loss to total loss
        if self.opt['rank'] == 0:
            print("G train loss:", self.G_train_loss.item())

        #self.G_optimizer.zero_grad()  # set parameter gradients to zero
        self.gen_scaler.scale(self.gen_loss).backward()  # backward-pass to compute gradients

        self.update = ((self.G_accum_count + 1) % self.num_accum_steps_G) == 0 or update
        if self.update:  # Gradient acculumation
            # ------------------------------------
            # clip_grad on G
            # ------------------------------------
            G_clipgrad_max = self.opt_train['G_optimizer_clipgrad']
            if G_clipgrad_max > 0:
                # Unscales the gradients of optimizer's assigned params in-place if AMP is enabled
                self.gen_scaler.unscale_(self.G_optimizer)
                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                self.G_train_grad_norm = torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=G_clipgrad_max, norm_type=2)
                # if self.opt['rank'] == 0:
                #     print("G gradient norm:", grad_norm.item())

            self.gen_scaler.step(self.G_optimizer)  # update weights
            self.gen_scaler.update()
            self.G_optimizer.zero_grad()  # set parameter gradients to zero

            # Reset gradient accumulation count
            self.G_accum_count = 0

        else:  # Update gradient accumulation count
            self.G_accum_count += 1


        # self.G_optimizer.zero_grad()
        #
        # # Forward G on fake image E with/without AMP
        # self.netG_forward()
        #
        # # Compute generator loss with/without AMP
        # if self.mixed_precision is not None:
        #     with torch.cuda.amp.autocast(dtype=self.mixed_precision):
        #         self.gen_loss = compute_generator_loss(self.H, self.E, self.loss_fn_dict, self.loss_val_dict, None, self.device)
        #         self.gen_scaler.scale(self.gen_loss).backward()
        # else:
        #     self.gen_loss = compute_generator_loss(self.H, self.E, self.loss_fn_dict, self.loss_val_dict, None, self.device)
        #     self.gen_loss.backward()  # Standard precision backward
        #
        # # Add generator training loss to total loss
        # self.G_train_loss += self.gen_loss
        #
        # # ------------------------------------
        # # clip_grad
        # # ------------------------------------
        # G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        # if G_optimizer_clipgrad > 0:
        #     # Unscales the gradients of optimizer's assigned params in-place if AMP is enabled
        #     if self.mixed_precision is not None:
        #         self.gen_scaler.unscale_(self.G_optimizer)
        #     # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        #     print("G gradient norm:", torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=G_optimizer_clipgrad, norm_type=2).item())
        #
        # # ------------------------------------
        # # update parameters for G
        # # ------------------------------------
        # if self.mixed_precision is not None:
        #     self.gen_scaler.step(self.G_optimizer)
        #     self.gen_scaler.update()
        # else:
        #     self.G_optimizer.step()
        #
        # # ------------------------------------
        # # TODO Regularizer as in SuperFormer
        # # ------------------------------------

    def optimize_parameters(self, current_step, update=False):

        # ------------------------------------
        # optimize G
        # ------------------------------------

        # Forward G
        self.netG_forward()
        self.gen_loss = compute_generator_loss(self.H, self.E, self.loss_fn_dict, self.loss_val_dict,None, self.device)
        self.gen_loss = self.gen_loss / self.num_accum_steps_G  # Scale loss by number of accumulation steps

        self.G_train_loss = self.gen_loss  # Add generator training loss to total loss
        if self.opt['rank'] == 0:
            print("G train loss:", self.G_train_loss.item())

        self.gen_loss.backward()  # backward-pass to compute gradients

        self.update = ((self.G_accum_count + 1) % self.num_accum_steps_G) == 0 or update
        if self.update:  # Gradient acculumation
            # ------------------------------------
            # clip_grad on G
            # ------------------------------------
            G_clipgrad_max = self.opt_train['G_optimizer_clipgrad']
            if G_clipgrad_max > 0:
                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                self.G_train_grad_norm = torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=G_clipgrad_max, norm_type=2)
                # if self.opt['rank'] == 0:
                #     print("G gradient norm:", grad_norm.item())

            self.G_optimizer.step()  # update weights
            self.G_optimizer.zero_grad()  # set parameter gradients to zero

            # Reset gradient accumulation count
            self.G_accum_count = 0

        else:  # Update gradient accumulation count
            self.G_accum_count += 1
        # # ------------------------------------
        # # TODO Regularizer as in SuperFormer
        # # ------------------------------------


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

    def record_test_log(self, current_step, idx_test):
        # ------------------------------------
        # record log
        # ------------------------------------

        for key, value in self.metric_val_dict.items():
            # Get metric name for logging
            metric_name = "Average " + key
            # Record metric value using wandb
            self.run.log({metric_name: self.metric_val_dict[key] / idx_test})
            print(metric_name, self.metric_val_dict[key] / idx_test)
            # Reset performance metric
            self.metric_val_dict[key] = 0.0

        self.run.log({"G_valid_loss": self.G_valid_loss.item() / idx_test})

        # Reset validation losses
        self.G_valid_loss = 0.0

    # ----------------------------------------
    # test and inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        #with torch.no_grad():
        with torch.inference_mode():
            self.netG_forward()
        self.netG.train()

    def validation(self):

        # Forward G
        self.netG_forward()

        # Compute loss for G
        self.gen_loss = compute_generator_loss(self.H, self.E, self.loss_fn_dict, self.loss_val_dict, None, self.device)

        # Add generator validation loss to total loss
        self.G_valid_loss += self.gen_loss

        # Compute performance metrics
        rescale_images = True if self.opt['dataset_opt']['norm_type'] == "znormalization" else False
        if self.opt['input_type'] == '2D':
            compute_performance_metrics_2D(self.E, self.H, self.metric_fn_dict, self.metric_val_dict, rescale_images)
        elif self.opt['input_type'] == '3D':
            compute_performance_metrics_3D(self.E, self.H, self.metric_fn_dict, self.metric_val_dict, rescale_images)


    def validation_amp(self):

        # Forward G
        self.netG_forward()

        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            # Compute loss for G
            self.gen_loss = compute_generator_loss(self.H, self.E, self.loss_fn_dict, self.loss_val_dict, None, self.device)

        # Add generator validation loss to total loss
        self.G_valid_loss += self.gen_loss

        rescale_images = True if self.opt['dataset_opt']['norm_type'] == "znormalization" else False
        compute_performance_metrics(self.E, self.H, self.metric_fn_dict, self.metric_val_dict, rescale_images)


        # self.netG.eval()
        #
        # with torch.inference_mode():
        #
        #     # Forward G on fake image E with/without AMP
        #     self.netG_forward()
        #
        #     # Compute generator validation loss with/without AMP
        #     if self.mixed_precision is not None:
        #         with torch.cuda.amp.autocast(dtype=self.mixed_precision):
        #             self.gen_loss = compute_generator_loss(self.H, self.E, self.loss_fn_dict, self.loss_val_dict, None, self.device)
        #     else:
        #         self.gen_loss = compute_generator_loss(self.H, self.E, self.loss_fn_dict, self.loss_val_dict, None, self.device)
        #
        #     # Add generator validation loss to total loss
        #     self.G_valid_loss += self.gen_loss
        #
        #     # Compute performance metrics:
        #     #self.E_img = utils_image.tensor2ufloat(self.E) # returns floats clamped between 0 and 1
        #     #self.H_img = utils_image.tensor2ufloat(self.H) # returns floats clamped between 0 and 1
        #     compute_performance_metrics(self.E, self.H, self.metric_fn_dict, self.metric_val_dict)
        #     #for key, value in self.metric_func_dict.items():
        #     #    self.metric_val_dict[key] += self.metric_func_dict[key](self.E_img, self.H_img, border=config.OPT['scale'])
        #
        #         #self.psnr += util.calculate_psnr(self.E_img, self.H_img, border=border)
        #         #self.nrmse += util.calculate_nrmse(self.H_img, self.E_img, border=border)
        #         #self.ssim += util.calculate_ssim_3d(self.H_img, self.E_img, border=border)
        #
        #     #save_img_path = os.path.join(img_dir, '{:s}_{:d}.nii.gz'.format(img_name, current_step))
        #     #output_nib = nib.Nifti1Image(E_img, np.eye(4))




    def full_reconstruction(self):
        pass
        # # Code for full sample reconstruction from SuperFormer
        # HR = self.H
        # #HR = test_data["H"]
        # H, W, D = HR.shape[2:]
        # patches = (HR.shape[2] // opt["datasets"]["test"]["train_size"]) * (
        #             HR.shape[3] // opt["datasets"]["test"]["train_size"]) * (
        #                       HR.shape[4] // opt["datasets"]["test"]["train_size"])
        # model.netG.eval()
        # output = torch.zeros_like(test_data['H'])
        # i = 0
        # for h in range(H // train_size):
        #     for w in range(W // train_size):
        #         for d in range(D // train_size):
        #             patch_L = test_data['L'][:, :, h * train_size:h * train_size + train_size,
        #                       w * train_size:w * train_size + train_size,
        #                       d * train_size:d * train_size + train_size]
        #             model.feed_data({'L': patch_L}, need_H=False)
        #             model.test()
        #             output[:, :, h * train_size:h * train_size + train_size,
        #             w * train_size:w * train_size + train_size,
        #             d * train_size:d * train_size + train_size] = model.E
        #             print(i)
        #             i += 1
        #
        # self.E_img = util.tensor2uint(output)
        # self.H_img = util.tensor2uint(HR)


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
            out_dict['L'] = crop_center(self.L, center_size=roi).detach()[0].float().cpu()
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


import lightning as L

class ModelPlainLit(L.LightningModule):
    def __init__(self, opt, model: ModelPlain):
        super().__init__()
        self.opt = opt
        self.model = model

    def training_step(self, train_batch, batch_idx):

        if self.opt['model_architecture'] == "MTVNet" and not self.opt['dataset_opt']['enable_femur_padding']:
            train_batch['H'] = crop_context(train_batch['H'], L=self.model.opt['netG']['num_levels'], level_ratio=self.model.opt['netG']['level_ratio'])

        self.model.feed_data(train_batch, need_H=True)

        self.model.netG_forward()

        loss = compute_generator_loss(self.model.H, self.model.E, self.model.loss_fn_dict, self.model.loss_val_dict, None, self.model.device)

        # Log training loss here
        self.log("G_train_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        return loss


    def validation_step(self, test_batch, batch_idx):

        if self.opt['model_architecture'] == "MTVNet" and not self.opt['dataset_opt']['enable_femur_padding']:
            test_batch['H'] = crop_context(test_batch['H'], L=self.model.opt['netG']['num_levels'], level_ratio=self.model.opt['netG']['level_ratio'])

        self.model.feed_data(test_batch, need_H=True)

        self.model.netG_forward()

        loss = compute_generator_loss(self.model.H, self.model.E, self.model.loss_fn_dict, self.model.loss_val_dict, None, self.model.device)

        # Log validation loss here
        self.log("G_valid_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        # Compute performance metrics
        rescale_images = True if self.opt['dataset_opt']['norm_type'] == "znormalization" else False
        compute_performance_metrics(self.E, self.H, self.metric_fn_dict, self.metric_val_dict, rescale_images)

        return loss


    def configure_optimizers(self):
        """defines model optimizer and scheduler"""
        return {
            "optimizer": self.model.G_optimizer,
            "lr_scheduler": {
                "scheduler": self.model.schedulers,
                "interval": "step",  # Step after every global_step
            },
        }

from lightning.pytorch.callbacks import Callback
from utils.utils_3D_image import crop_context, crop_center

class CustomCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training started!")

    def on_validation_end(self, trainer, pl_module):

        if pl_module.opt['model_architecture'] == "MTVNet":
            pl_module.model.L = crop_center(pl_module.model.L, center_size=pl_module.model.opt['netG']['context_sizes'][-1])

        visuals = pl_module.model.current_visuals()
        pl_module.model.log_comparison_image(visuals, trainer.global_step)

        print("Validation completed!")


from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
from lightning.pytorch.utilities import rank_zero_only
class MyLogger(Logger):
    @property
    def name(self):
        return "MyLogger"

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        pass

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        pass

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass