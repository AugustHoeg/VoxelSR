import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam, AdamW

from loss_functions.loss_functions_simple import bce_dis_loss, compute_generator_loss
from models.model_base import ModelBase
from models.select_network import define_D, define_G
from performance_metrics.performance_metrics import compute_performance_metrics
from utils import utils_3D_image
from utils.utils_dist import get_rank, reduce_sum


class ModelGAN(ModelBase):
    """Train with pixel-VGG-GAN loss"""
    def __init__(self, opt, mode='train', data_parallel=True):
        super(ModelGAN, self).__init__(opt)
        self.last_iteration = 0
        self.netG = define_G(opt, mode=mode)
        self.netG = self.model_to_device(self.netG, data_parallel=data_parallel, compile=False)
        if mode == 'train':
            self.netD = define_D(opt, mode=mode)
            self.netD = self.model_to_device(self.netD, data_parallel=data_parallel, compile=False)
            if self.opt_train['E_decay'] > 0:
                self.netE = define_G(opt).to(self.device).eval()

        if opt['rank'] == 0 and mode == 'train':
            print("Number of trainable parameters, G", utils_3D_image.numel(self.netG, only_trainable=True))
            print("Number of trainable parameters, D", utils_3D_image.numel(self.netD, only_trainable=True))

        self.update = False

        self.early_stop = False
        self.min_validation_loss = float('inf')
        self.patience = self.opt_train['early_stop_patience']
        self.patience_counter = 0
        self.min_delta = 0

    def set_eval_mode(self):
        self.netG.eval()
        self.netD.eval()

    def set_train_mode(self):
        self.netG.train()
        self.netD.train()

    def init_test(self, experiment_id):
        self.load(experiment_id, mode='test')
        self.netG.eval()
        self.define_metrics()
        self.define_mixed_precision()
        self.define_visual_eval()

    def init_train(self):
        self.load()
        self.netG.train()
        self.netD.train()

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
    # Checkpoint load / save  (G + D)
    # ----------------------------------------

    def load(self, experiment_id=None, mode='train'):
        eid = self.opt['path']['pretrained_experiment_id'] if experiment_id is None else experiment_id

        if mode == 'train':
            if self.opt['train_mode'] == 'scratch':
                return
            assert eid is not None, f"Pretrained experiment ID required for train_mode='{self.opt['train_mode']}'."
        else:
            assert eid is not None, "Experiment ID required for test mode."

        G_path = self._find_latest_checkpoint(eid, "saved_models", "*G.h5")
        D_path = self._find_latest_checkpoint(eid, "saved_models", "*D.h5")
        if G_path is None or D_path is None:
            print("No G or D checkpoint found, skipping loading...")
            return
        if self.opt['rank'] == 0:
            print(f"Loading G [{self._short_path(G_path)}] ...")
            print(f"Loading D [{self._short_path(D_path)}] ...")
        self.load_network(G_path, self.netG, strict=self.opt_train['G_param_strict'])
        if mode == 'train':
            self.load_network(D_path, self.netD, strict=self.opt_train['D_param_strict'])
        self.last_iteration = int(os.path.basename(G_path).split('_')[0])

    def save(self, iter_label):
        super().save(iter_label)  # saves G, G_optimizer, schedulers[0], gen_scaler, netE
        self.save_network(self._run_dir("saved_models"), self.netD, 'D', iter_label)
        self.save_optimizer(self._run_dir("saved_optimizers"), self.D_optimizer, 'optimizerD', iter_label)
        self.save_scheduler(self._run_dir("saved_schedulers"), self.schedulers[1], 'schedulerD', iter_label)
        if self.mixed_precision is not None:
            self.save_gradscaler(self._run_dir("saved_gradscalers"), self.dis_scaler, 'gradscalerD', iter_label)

    def load_optimizers(self, experiment_id=None):
        if self.opt['train_mode'] != 'resume':
            return
        eid = self.opt['path']['pretrained_experiment_id'] if experiment_id is None else experiment_id
        assert eid is not None, "Pretrained experiment ID required for train_mode='resume'."
        self.opt['train_opt']['G_optimizer_reuse'] = True

        G_path = self._find_latest_checkpoint(eid, "saved_optimizers", "*optimizerG.h5")
        D_path = self._find_latest_checkpoint(eid, "saved_optimizers", "*optimizerD.h5")
        if G_path is None or D_path is None:
            print("No G or D optimizer checkpoint found, skipping...")
            return
        if self.opt['rank'] == 0:
            print(f"Loading G optimizer [{self._short_path(G_path)}] ...")
            print(f"Loading D optimizer [{self._short_path(D_path)}] ...")
        self.load_optimizer(G_path, self.G_optimizer)
        self.load_optimizer(D_path, self.D_optimizer)

    def load_schedulers(self, experiment_id=None):
        if self.opt['train_mode'] != 'resume':
            return
        eid = self.opt['path']['pretrained_experiment_id'] if experiment_id is None else experiment_id
        assert eid is not None, "Pretrained experiment ID required for train_mode='resume'."

        G_path = self._find_latest_checkpoint(eid, "saved_schedulers", "*schedulerG.h5")
        D_path = self._find_latest_checkpoint(eid, "saved_schedulers", "*schedulerD.h5")
        if G_path is None or D_path is None:
            print("No G or D scheduler checkpoint found, skipping...")
            return
        if self.opt['rank'] == 0:
            print(f"Loading G scheduler [{self._short_path(G_path)}] ...")
            print(f"Loading D scheduler [{self._short_path(D_path)}] ...")
        self.load_scheduler(G_path, self.schedulers[0])
        self.load_scheduler(D_path, self.schedulers[1])

    def load_gradscalers(self, experiment_id=None):
        if self.opt['train_mode'] != 'resume':
            return
        eid = self.opt['path']['pretrained_experiment_id'] if experiment_id is None else experiment_id
        assert eid is not None, "Pretrained experiment ID required for train_mode='resume'."

        G_path = self._find_latest_checkpoint(eid, "saved_gradscalers", "*gradscalerG.h5")
        D_path = self._find_latest_checkpoint(eid, "saved_gradscalers", "*gradscalerD.h5")
        if G_path is None or D_path is None:
            print("No G or D gradscaler checkpoint found, skipping...")
            return
        if self.opt['rank'] == 0:
            print(f"Loading G gradscaler [{self._short_path(G_path)}] ...")
            print(f"Loading D gradscaler [{self._short_path(D_path)}] ...")
        self.load_gradscaler(G_path, self.gen_scaler)
        self.load_gradscaler(D_path, self.dis_scaler)

    def define_gradscaler(self):
        self.define_G_gradscaler()
        self.define_D_gradscaler()

    def define_scheduler(self):
        self.define_G_scheduler()
        self.define_D_scheduler()

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
        self.model_artifact_D = wandb.Artifact(
            "Discriminator", type=self.opt['model_opt']['netD']['net_type'],
            description=self.opt['model_opt']['netD']['description'],
            metadata=OmegaConf.to_container(self.opt['model_opt']['netD'], resolve=True)
        )

    def define_loss(self):
        self.build_loss_fn_dict()
        self.init_G_loss_trackers()
        self.init_D_loss_trackers()

    def define_optimizer(self):

        self.D_accum_count = 0
        self.num_accum_steps_D = self.opt_train['num_accum_steps_D']
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

        if self.opt_train['D_optimizer_type'] == 'adam':
            self.D_optimizer = Adam(self.netD.parameters(), lr=self.opt_train['D_optimizer_lr'], weight_decay=self.opt_train['D_optimizer_wd'], betas=(0.9, 0.999))
        elif self.opt_train['D_optimizer_type'] == 'adamw':
            self.D_optimizer = AdamW(self.netD.parameters(), lr=self.opt_train['D_optimizer_lr'], weight_decay=self.opt_train['D_optimizer_wd'], betas=(0.9, 0.999))
        else:
            raise NotImplementedError('optimizer [{:s}] is not implemented.'.format(self.opt_train['D_optimizer_type']))

    def feed_data(self, data):
        self.L = data['L'].as_tensor().to(self.device, non_blocking=True)
        self.H = data['H'].as_tensor().to(self.device, non_blocking=True)

    def netG_forward(self):
        self.E = self.netG(self.L)

    def netD_forward(self, input):
        return self.netD(input)

    def optimize_parameters_amp(self, current_step, update=False):

        # optimize D
        self.netD.requires_grad_(True)

        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            self.netG_forward()
            self.prop_real = self.netD_forward(self.H)
            self.prop_fake = self.netD_forward(self.E.detach())
            self.dis_loss = bce_dis_loss(self.prop_real, self.prop_fake)
            self.dis_loss = self.dis_loss / self.num_accum_steps_D

        self.D_train_loss = self.dis_loss
        if self.opt['rank'] == 0:
            print("D train loss:", self.D_train_loss.item())

        self.D_update = ((self.D_accum_count + 1) % self.num_accum_steps_D) == 0 or update

        if not self.D_update:
            if isinstance(self.netD, DistributedDataParallel):
                with self.netD.no_sync():
                    self.dis_scaler.scale(self.dis_loss).backward()
            else:
                self.dis_scaler.scale(self.dis_loss).backward()
        else:
            self.dis_scaler.scale(self.dis_loss).backward()

        if self.D_update:
            D_clipgrad_max = self.opt_train['D_optimizer_clipgrad']
            if D_clipgrad_max > 0:
                self.dis_scaler.unscale_(self.D_optimizer)
                self.D_train_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.netD.parameters(), max_norm=D_clipgrad_max, norm_type=2
                )
            self.dis_scaler.step(self.D_optimizer)
            self.dis_scaler.update()
            self.D_optimizer.zero_grad()
            self.D_accum_count = 0
        else:
            self.D_accum_count += 1

        # optimize G
        self.netD.requires_grad_(False)

        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            self.prop_fake = self.netD_forward(self.E)
            recon_loss = compute_generator_loss(self.H, self.E, self.loss_fn_dict, self.loss_val_dict, self.device)
            adv_loss = F.binary_cross_entropy_with_logits(self.prop_fake, torch.ones_like(self.prop_fake))
            self.gen_loss = recon_loss + self.loss_val_dict['ADV'] * adv_loss
            self.gen_loss = self.gen_loss / self.num_accum_steps_G

        self.G_train_loss = self.gen_loss
        if self.opt['rank'] == 0:
            print("G train loss:", self.G_train_loss.item())

        self.G_update = ((self.G_accum_count + 1) % self.num_accum_steps_G) == 0 or update

        if not self.G_update:
            if isinstance(self.netG, DistributedDataParallel):
                with self.netG.no_sync():
                    self.gen_scaler.scale(self.gen_loss).backward()
            else:
                self.gen_scaler.scale(self.gen_loss).backward()
        else:
            self.gen_scaler.scale(self.gen_loss).backward()

        if self.G_update:
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

        # optimize D
        self.netD.requires_grad_(True)
        self.netG_forward()
        self.prop_real = self.netD_forward(self.H)
        self.prop_fake = self.netD_forward(self.E.detach())
        self.dis_loss = bce_dis_loss(self.prop_real, self.prop_fake)
        self.dis_loss = self.dis_loss / self.num_accum_steps_D

        self.D_train_loss = self.dis_loss
        if self.opt['rank'] == 0:
            print("D train loss:", self.D_train_loss.item())

        self.D_update = ((self.D_accum_count + 1) % self.num_accum_steps_D) == 0 or update

        if not self.D_update:
            if isinstance(self.netD, DistributedDataParallel):
                with self.netD.no_sync():
                    self.dis_loss.backward()
            else:
                self.dis_loss.backward()
        else:
            self.dis_loss.backward()

        if self.D_update:
            D_clipgrad_max = self.opt_train['D_optimizer_clipgrad']
            if D_clipgrad_max > 0:
                self.D_train_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.netD.parameters(), max_norm=D_clipgrad_max, norm_type=2
                )
            self.D_optimizer.step()
            self.D_optimizer.zero_grad()
            self.D_accum_count = 0
        else:
            self.D_accum_count += 1

        # optimize G
        self.netD.requires_grad_(False)
        self.prop_fake = self.netD_forward(self.E)
        recon_loss = compute_generator_loss(self.H, self.E, self.loss_fn_dict, self.loss_val_dict, self.device)
        adv_loss = F.binary_cross_entropy_with_logits(self.prop_fake, torch.ones_like(self.prop_fake))
        self.gen_loss = recon_loss + self.loss_val_dict['ADV'] * adv_loss
        self.gen_loss = self.gen_loss / self.num_accum_steps_G

        self.G_train_loss = self.gen_loss
        if self.opt['rank'] == 0:
            print("G train loss:", self.G_train_loss.item())

        self.G_update = ((self.G_accum_count + 1) % self.num_accum_steps_G) == 0 or update

        if not self.G_update:
            if isinstance(self.netG, DistributedDataParallel):
                with self.netG.no_sync():
                    self.gen_loss.backward()
            else:
                self.gen_loss.backward()
        else:
            self.gen_loss.backward()

        if self.G_update:
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
        G_loss = self.G_train_loss.item() * self.num_accum_steps_G
        self.run.log({"step": current_step, "G_train_loss": G_loss})

        D_loss = self.D_train_loss.item() * self.num_accum_steps_D
        self.run.log({"step": current_step, "D_train_loss": D_loss})

        G_grad_norm = self.G_train_grad_norm.item() * self.num_accum_steps_G
        self.run.log({"step": current_step, "G_train_grad_norm": G_grad_norm})

        D_grad_norm = self.D_train_grad_norm.item() * self.num_accum_steps_D
        self.run.log({"step": current_step, "D_train_grad_norm": D_grad_norm})

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    def record_avg_train_log(self, current_step, idx_train):
        avg_loss_G = (self.G_train_loss.item() / idx_train) * self.num_accum_steps_G
        self.run.log({"step": current_step, "G_train_loss": avg_loss_G})

        avg_loss_D = (self.D_train_loss.item() / idx_train) * self.num_accum_steps_D
        self.run.log({"step": current_step, "D_train_loss": avg_loss_D})

        self.G_train_loss = 0.0
        self.D_train_loss = 0.0

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    def record_test_log(self, idx_test):
        idx_tensor = torch.tensor(idx_test, device=self.device)
        global_idx_tensor = reduce_sum(idx_tensor)

        for key, value in self.metric_val_dict.items():
            metric_tensor = torch.tensor(float(value), device=self.device)
            global_average = reduce_sum(metric_tensor) / global_idx_tensor

            if get_rank() == 0:
                metric_name = "Average " + key
                self.run.log({metric_name: global_average.item()})
                print(metric_name, global_average.item())

            self.metric_val_dict[key] = 0.0

        G_loss_tensor = torch.tensor(float(self.G_valid_loss), device=self.device)
        G_global_valid_loss = reduce_sum(G_loss_tensor) / global_idx_tensor

        D_loss_tensor = torch.tensor(float(self.D_valid_loss), device=self.device)
        D_global_valid_loss = reduce_sum(D_loss_tensor) / global_idx_tensor

        if get_rank() == 0:
            self.run.log({"G_valid_loss": G_global_valid_loss.item()})
            self.run.log({"D_valid_loss": D_global_valid_loss.item()})
            print("G_valid_loss", G_global_valid_loss.item())
            print("D_valid_loss", D_global_valid_loss.item())

        self.G_valid_loss = 0.0
        self.D_valid_loss = 0.0

    def test(self):
        self.netG.eval()
        with torch.inference_mode():
            self.netG_forward()
        self.netG.train()

    def validation(self):
        self.netG_forward()
        self.prop_real = self.netD_forward(self.H)
        self.prop_fake = self.netD_forward(self.E)

        recon_loss = compute_generator_loss(self.H, self.E, self.loss_fn_dict, self.loss_val_dict, self.device)
        adv_loss = F.binary_cross_entropy_with_logits(self.prop_fake, torch.ones_like(self.prop_fake))
        self.gen_loss = recon_loss + self.loss_val_dict['ADV'] * adv_loss
        self.dis_loss = bce_dis_loss(self.prop_real, self.prop_fake)

        self.G_valid_loss += self.gen_loss
        self.D_valid_loss += self.dis_loss

        rescale_images = self.opt['dataset_opt']['norm_type'] == "znormalization"
        compute_performance_metrics(self.E, self.H, self.metric_fn_dict, self.metric_val_dict, rescale_images)

    def validation_amp(self):

        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            self.netG_forward()
            self.prop_real = self.netD_forward(self.H)
            self.prop_fake = self.netD_forward(self.E)
            recon_loss = compute_generator_loss(self.H, self.E, self.loss_fn_dict, self.loss_val_dict, self.device)
            adv_loss = F.binary_cross_entropy_with_logits(self.prop_fake, torch.ones_like(self.prop_fake))
            self.gen_loss = recon_loss + self.loss_val_dict['ADV'] * adv_loss

        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            self.dis_loss = bce_dis_loss(self.prop_real, self.prop_fake)

        self.G_valid_loss += self.gen_loss
        self.D_valid_loss += self.dis_loss

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
