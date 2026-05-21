import os
from collections import OrderedDict

import torch
import torch.nn as nn
import wandb
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam, AdamW

from loss_functions.loss_functions_simple import compute_generator_loss
from models.model_base import ModelBase
from models.select_network import define_G
from performance_metrics.performance_metrics import compute_performance_metrics
from utils import utils_3D_image


class ModelVQVAE(ModelBase):
    """Train with pixel-VGG-GAN loss"""
    def __init__(self, opt, mode='train', data_parallel=True):
        super(ModelVQVAE, self).__init__(opt)
        self.last_iteration = 0
        self.netG = define_G(opt, mode=mode)
        self.netG = self.model_to_device(self.netG, data_parallel=data_parallel, compile=False)

        if opt['rank'] == 0 and mode == 'train':
            print("Number of trainable parameters", utils_3D_image.numel(self.netG, only_trainable=True))

        self.update = False

        self.early_stop = False
        self.min_validation_loss = float('inf')
        self.patience = self.opt_train['early_stop_patience']
        self.patience_counter = 0
        self.min_delta = 0

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
                "up_factor": self.opt['up_factor'],
                "architecture": self.opt['model_opt']['model_architecture'],
            })

        os.mkdir(os.path.join(wandb.run.dir, "saved_models"))
        os.mkdir(os.path.join(wandb.run.dir, "saved_optimizers"))
        os.mkdir(os.path.join(wandb.run.dir, "saved_schedulers"))
        os.mkdir(os.path.join(wandb.run.dir, "saved_gradscalers"))

        self.wandb_config = wandb.config

        self.model_artifact_G = wandb.Artifact(
            "Generator", type=self.opt['model_opt']['netG']['net_type'],
            description=self.opt['model_opt']['netG']['description'],
            metadata=OmegaConf.to_container(self.opt['model_opt']['netG'], resolve=True)
        )

    def define_loss(self):
        self.build_loss_fn_dict()
        self.init_G_loss_trackers()

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

    def feed_data(self, data):
        self.L = data['L'].as_tensor().to(self.device, non_blocking=True)
        self.H = data['H'].as_tensor().to(self.device, non_blocking=True)

    def vq_forward(self):
        self.E, self.vq_loss = self.netG(self.H)

    def netG_forward(self):
        self.E, _ = self.netG(self.L)

    def optimize_parameters_amp(self, current_step, update=False):

        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            self.vq_forward()
            recon_loss = compute_generator_loss(self.H, self.E, self.loss_fn_dict, self.loss_val_dict)
            self.gen_loss = self.vq_loss + recon_loss
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

    def validation(self):
        self.vq_forward()

        recon_loss = compute_generator_loss(self.H, self.E, self.loss_fn_dict, self.loss_val_dict)
        self.gen_loss = self.vq_loss + recon_loss

        self.G_valid_loss += self.gen_loss

        rescale_images = self.opt['dataset_opt']['norm_type'] == "znormalization"
        compute_performance_metrics(self.E, self.H, self.metric_fn_dict, self.metric_val_dict, rescale_images)

    def validation_amp(self):
        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            self.vq_forward()
            recon_loss = compute_generator_loss(self.H, self.E, self.loss_fn_dict, self.loss_val_dict)
            self.gen_loss = self.vq_loss + recon_loss

        self.G_valid_loss += self.gen_loss

        rescale_images = self.opt['dataset_opt']['norm_type'] == "znormalization"
        compute_performance_metrics(self.E, self.H, self.metric_fn_dict, self.metric_val_dict, rescale_images)

    def current_log(self):
        return self.log_dict

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
