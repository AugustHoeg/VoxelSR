import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam, AdamW

from loss_functions.loss_functions_simple import FSCLoss3D, LPIPSLoss3D, compute_generator_loss
from models.model_base import ModelBase
from models.select_network import define_G
from performance_metrics.performance_metrics import compute_performance_metrics
from utils import utils_3D_image


class ModelDegradation(ModelBase):
    """Train with pixel-VGG-GAN loss"""
    def __init__(self, opt, mode='train', data_parallel=True):
        super(ModelDegradation, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.last_iteration = 0
        self.netG = define_G(opt, mode=mode)
        self.netG = self.model_to_device(self.netG, data_parallel=data_parallel, compile=False)
        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(opt).to(self.device).eval()

        if opt['rank'] == 0 and mode == 'train':
            print("Number of trainable parameters, G", utils_3D_image.numel(self.netG, only_trainable=True))

        self.update = False

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

    def init_test(self, experiment_id):
        self.load(experiment_id, mode='test')
        self.netG.eval()
        self.define_metrics()
        self.define_mixed_precision()
        self.define_visual_eval()

    def init_train(self):
        self.load()
        self.netG.train()

        self.define_loss()
        self.define_metrics()

        self.define_optimizer()
        self.load_optimizers()

        self.define_mixed_precision()
        self.load_gradscalers()

        self.define_scheduler()
        self.load_schedulers()

        self.define_visual_eval()

    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self, experiment_id=None, mode='train'):
        eid = self.opt['path']['pretrained_experiment_id'] if experiment_id is None else experiment_id

        if mode == 'train':
            if self.opt['train_mode'] == 'scratch':
                return
            assert eid is not None, f"Pretrained experiment ID required for train_mode='{self.opt['train_mode']}'."
        else:
            assert eid is not None, "Experiment ID required for test mode."

        path = self._find_latest_checkpoint(eid, "saved_models", "*G.h5")
        if path is None:
            print("No G checkpoint found, skipping loading...")
            return
        if self.opt['rank'] == 0:
            print(f"Loading G [{self._short_path(path)}] ...")
        self.load_network(path, self.netG, strict=self.opt_train['G_param_strict'])
        self.last_iteration = int(os.path.basename(path).split('_')[0])

    def define_wandb_run(self):

        self.run = wandb.init(
            mode=self.opt["wandb_mode"],
            entity=self.opt['wandb_entity'],
            project=self.opt['wandb_project'],
            name=self.opt['run_name'],
            id=self.opt['experiment_id'],
            notes=self.opt['note'],
            dir="logs/" + self.opt['dataset_opt']['name'],
            config={
                "iterations": self.opt['train_opt']['iterations'],
                "G_learning_rate": self.opt['train_opt']['G_optimizer_lr'],
                "batch_size": self.opt['dataset_opt']['train_dataloader_params']['dataloader_batch_size'],
                "dataset": self.opt['dataset_opt']['name'],
                "down_factor": self.opt['down_factor'],
                "architecture": self.opt['model_opt']['model_architecture'],
            })

        os.mkdir(os.path.join(wandb.run.dir, "saved_models"))
        os.mkdir(os.path.join(wandb.run.dir, "saved_optimizers"))
        os.mkdir(os.path.join(wandb.run.dir, "saved_schedulers"))
        os.mkdir(os.path.join(wandb.run.dir, "saved_gradscalers"))

        self.wandb_config = wandb.config

        self.model_artifact_G = wandb.Artifact(
            "Generator", type=self.opt['model_opt']['model_architecture'],
            description=self.opt['model_opt']['netG']['description'],
            metadata=OmegaConf.to_container(self.opt['model_opt']['netG'], resolve=True))

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):

        self.loss_fn_dict = {}
        self.loss_val_dict = self.opt_train['G_loss_weights']

        for key, value in self.loss_val_dict.items():
            if key == "MSE" and value > 0:
                self.loss_fn_dict["MSE"] = nn.MSELoss()
            elif key == "L1" and value > 0:
                self.loss_fn_dict["L1"] = nn.L1Loss()
            elif key == "BCE_Logistic" and value > 0:
                self.loss_fn_dict["BCE_Logistic"] = nn.BCEWithLogitsLoss()
            elif key == "BCE" and value > 0:
                self.loss_fn_dict["BCE"] = nn.BCELoss()
            elif key == "LPIPS" and value > 0:
                self.loss_fn_dict["LPIPS"] = LPIPSLoss3D(
                    net_type='alex', version='0.1', device=self.device, axes=(0, 1, 2)
                )
            elif key == "FSC" and value > 0:
                self.loss_fn_dict["FSC"] = FSCLoss3D(
                    size=self.opt['dataset_opt']['patch_size_hr'], delta=1, alpha=2.0,
                    drop_DC=False, device=self.device
                )
            elif key == "CSC" and value > 0:
                from loss_functions.loss_functions_simple import CSCLoss
                self.loss_fn_dict["CSC"] = CSCLoss(
                    eval_mode=True, verbose=True, feat_dist_func='FSC',
                    compare_input=False, device=self.device,
                    size=self.opt['dataset_opt']['patch_size_hr'],
                    experiment_id=self.opt_train['pretrained_G_loss_IDs']['CSC']
                )
            elif key == "AESOP3D" and value > 0:
                from loss_functions.loss_functions_simple import AESOPLoss3D
                self.loss_fn_dict["AESOP3D"] = AESOPLoss3D(
                    ae_criterion_type='L1', ae_weight=1.0,
                    experiment_id=self.opt_train['pretrained_G_loss_IDs']['AESOP3D'],
                )

        self.G_train_loss = 0.0
        self.G_valid_loss = 0.0
        self.G_train_grad_norm = 0.0
        self.G_valid_grad_norm = 0.0

    # ----------------------------------------
    # define optimizer G
    # ----------------------------------------
    def define_optimizer(self):

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

    def define_visual_eval(self):
        if self.opt['input_type'] == '2D':
            raise NotImplementedError("Visual evaluation not implemented for 2D images yet.")
        elif self.opt['input_type'] == '3D':
            from utils.utils_degradation import DegradationComparisonTool3D_V2 as comparison_tool

            self.comparison_tool = comparison_tool(
                patch_size_hr=self.opt['dataset_opt']['patch_size_hr'],
                down_factor=self.opt['down_factor'],
                upscaling_methods=["tio_nearest"],
                unnorm=self.opt['dataset_opt']['norm_type'] == 'znormalization',
                div_max=self.opt['dataset_opt']['norm_type'] == 'znormalization',
                plot_synth_LR=True,
                out_dtype=np.uint8
            )

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    def feed_data(self, data):
        self.L = data['L'].as_tensor().to(self.device, non_blocking=True)
        self.H = data['H'].as_tensor().to(self.device, non_blocking=True)

    def netG_forward(self):
        self.E = self.netG(self.H)

    def optimize_parameters_amp(self, current_step, update=False):

        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            self.netG_forward()
            self.gen_loss = compute_generator_loss(self.L, self.E, self.loss_fn_dict, self.loss_val_dict, device=self.device)
            self.gen_loss = self.gen_loss / self.num_accum_steps_G

        self.G_train_loss = self.gen_loss
        if self.opt['rank'] == 0:
            print("G train loss:", self.G_train_loss.item())

        self.update = ((self.G_accum_count + 1) % self.num_accum_steps_G) == 0 or update

        if not self.update:
            if isinstance(self.netG, DistributedDataParallel):
                with self.netG.no_sync():
                    self.gen_scaler.scale(self.gen_loss).backward()
            else:
                self.gen_scaler.scale(self.gen_loss).backward()
        else:
            self.gen_scaler.scale(self.gen_loss).backward()

        if self.update:
            G_clipgrad_max = self.opt_train['G_optimizer_clipgrad']
            if G_clipgrad_max > 0:
                self.gen_scaler.unscale_(self.G_optimizer)
                self.G_train_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.netG.parameters(), max_norm=G_clipgrad_max, norm_type=2
                )
            self.gen_scaler.step(self.G_optimizer)
            self.gen_scaler.update()
            self.G_optimizer.zero_grad()
            self.G_accum_count = 0
        else:
            self.G_accum_count += 1

    def optimize_parameters(self, current_step, update=False):

        self.netG_forward()
        self.gen_loss = compute_generator_loss(self.L, self.E, self.loss_fn_dict, self.loss_val_dict, device=self.device)
        self.gen_loss = self.gen_loss / self.num_accum_steps_G

        self.G_train_loss = self.gen_loss
        if self.opt['rank'] == 0:
            print("G train loss:", self.G_train_loss.item())

        self.update = ((self.G_accum_count + 1) % self.num_accum_steps_G) == 0 or update

        if not self.update:
            if isinstance(self.netG, DistributedDataParallel):
                with self.netG.no_sync():
                    self.gen_loss.backward()
            else:
                self.gen_loss.backward()
        else:
            self.gen_loss.backward()

        if self.update:
            G_clipgrad_max = self.opt_train['G_optimizer_clipgrad']
            if G_clipgrad_max > 0:
                self.G_train_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.netG.parameters(), max_norm=G_clipgrad_max, norm_type=2
                )
            self.G_optimizer.step()
            self.G_optimizer.zero_grad()
            self.G_accum_count = 0
        else:
            self.G_accum_count += 1

    def record_train_log(self, current_step):
        loss = self.G_train_loss.item() * self.num_accum_steps_G
        self.run.log({"step": current_step, "G_train_loss": loss})

        grad_norm = self.G_train_grad_norm.item() * self.num_accum_steps_G
        self.run.log({"step": current_step, "G_train_grad_norm": grad_norm})

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    def record_avg_train_log(self, current_step, idx_train):
        avg_loss = (self.G_train_loss.item() / idx_train) * self.num_accum_steps_G
        self.run.log({"step": current_step, "G_train_loss": avg_loss})

        self.G_train_loss = 0.0

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    # ----------------------------------------
    # test and inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        with torch.inference_mode():
            self.netG_forward()
        self.netG.train()

    def validation(self):
        self.netG_forward()

        self.gen_loss = compute_generator_loss(self.L, self.E, self.loss_fn_dict, self.loss_val_dict, device=self.device)
        self.G_valid_loss += self.gen_loss

        rescale_images = self.opt['dataset_opt']['norm_type'] == "znormalization"
        compute_performance_metrics(self.L, self.E, self.metric_fn_dict, self.metric_val_dict, rescale_images)

    def validation_amp(self):

        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            self.netG_forward()
            self.gen_loss = compute_generator_loss(self.L, self.E, self.loss_fn_dict, self.loss_val_dict, device=self.device)

        self.G_valid_loss += self.gen_loss

        rescale_images = self.opt['dataset_opt']['norm_type'] == "znormalization"
        compute_performance_metrics(self.L, self.E, self.metric_fn_dict, self.metric_val_dict, rescale_images)

    def current_log(self):
        return self.log_dict

    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()

        roi = int(self.opt['dataset_opt']['patch_size_hr'] / self.opt['down_factor'])
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
        figure_string = "Degradation comparison: %s, step %d, %dx downscaling" % (
            self.opt['model_opt']['model_architecture'], current_step, self.opt['down_factor']
        )

        if self.opt['run_type'] == "HOME PC":
            height, width = grid_image.shape[:2]
            plt.figure(figsize=(4 * width / 100, 4 * height / 100), dpi=100)
            plt.imshow(grid_image, vmin=0, vmax=255)
            plt.title(figure_string)
            plt.xticks([])
            plt.yticks([])
            plt.show()

        wandb.log({"Comparisons training": wandb.Image(grid_image, caption=figure_string, mode="RGB")})
