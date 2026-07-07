import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim import Adam, AdamW
from torchvision.utils import make_grid

from loss_functions.loss_functions_simple import compute_generator_loss
from models.model_base import ModelBase
from models.select_network import define_D, define_G
from performance_metrics.performance_metrics import compute_performance_metrics
from utils import utils_3D_image
from utils.utils_dist import get_rank, reduce_sum


class ModelVQGAN(ModelBase):
    """Train with pixel-VGG-GAN loss"""
    def __init__(self, opt, mode='train', data_parallel=True):
        super(ModelVQGAN, self).__init__(opt)
        self.last_iteration = 0
        self.netG = define_G(opt, mode=mode)
        self.netG = self.model_to_device(self.netG, data_parallel=data_parallel)
        if mode == 'train':
            self.netD = define_D(opt, mode=mode)
            self.netD = self.model_to_device(self.netD, data_parallel=data_parallel)
            if self.opt_train['E_decay'] > 0:
                self.netE = define_G(opt).to(self.device).eval()

        if opt['rank'] == 0 and mode == 'train':
            print("Number of trainable parameters, G", utils_3D_image.numel(self.netG, only_trainable=True))
            print("Number of trainable parameters, D", utils_3D_image.numel(self.netD, only_trainable=True))

        self.vae_target = opt.get('vae_target', 'HR')
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
        eid = self._resolve_eid(experiment_id)
        if mode == 'train':
            if self.opt['train_mode'] == 'scratch':
                return
            assert eid is not None, f"Pretrained experiment ID required for train_mode='{self.opt['train_mode']}'."
        else:
            assert eid is not None, 'Experiment ID required for test mode.'
        self.load_G(eid, mode)
        if mode == 'train':
            self.load_D(eid)

    def save(self, iter_label):
        self.save_G(iter_label)
        self.save_D(iter_label)

    def load_optimizers(self, experiment_id=None):
        if self.opt['train_mode'] != 'resume':
            return
        eid = self._resolve_eid(experiment_id)
        assert eid is not None, "Pretrained experiment ID required for train_mode='resume'."
        self.opt['train_opt']['G_optimizer_reuse'] = True
        self.load_G_optimizer(eid)
        self.load_D_optimizer(eid)

    def load_schedulers(self, experiment_id=None):
        if self.opt['train_mode'] != 'resume':
            return
        eid = self._resolve_eid(experiment_id)
        assert eid is not None, "Pretrained experiment ID required for train_mode='resume'."
        self.load_G_scheduler(eid)
        self.load_D_scheduler(eid)

    def load_gradscalers(self, experiment_id=None):
        if self.opt['train_mode'] != 'resume':
            return
        eid = self._resolve_eid(experiment_id)
        assert eid is not None, "Pretrained experiment ID required for train_mode='resume'."
        self.load_G_gradscaler(eid)
        self.load_D_gradscaler(eid)

    def define_gradscaler(self):
        self.define_G_gradscaler()
        self.define_D_gradscaler()

    def define_scheduler(self):
        self.define_G_scheduler()
        self.define_D_scheduler()

    def update_learning_rate(self):
        self.schedulers[0].step()
        if self.current_step >= self.opt_train['D_start_iteration']:
            self.schedulers[1].step()

    def define_wandb_run(self):
        self._init_wandb_run(extra_config={"up_factor": self.opt['up_factor']})
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

        self.lambda_adv = self.loss_val_dict['ADV']  # Initial value, will be dynamically updated during training

    def define_optimizer(self):
        self.define_G_optimizer()
        self.define_D_optimizer()

    def feed_data(self, data):
        self.L = data['L'].as_tensor().to(self.device, non_blocking=True)
        self.H = data['H'].as_tensor().to(self.device, non_blocking=True)
        self.vae_in = self.H if self.vae_target == 'HR' else self.L

    def vq_forward(self):
        self.E, self.vq_loss, self.codes, self.z_e, self.frac_unique = self.netG(self.vae_in)

    def netG_forward(self):
        self.E, _, _, _, _ = self.netG(self.L)  # Always L as inference_zarr expects this

    def netD_forward(self, input):
        return self.netD(input)

    def find_last_conv(self, model):
        for layer in reversed(list(model.modules())):
            if isinstance(layer, torch.nn.Conv3d):
                return layer
        raise ValueError("No convolutional layer found in the model.")

    def calculate_lambda(self, recon_loss, adv_loss):
        net = self.netG.module if isinstance(self.netG, (DataParallel, DistributedDataParallel)) else self.netG
        last_layer_weight = self.find_last_conv(net.decoder.model).weight
        recon_loss_grads = torch.autograd.grad(recon_loss, last_layer_weight, retain_graph=True)[0]
        adv_loss_grads = torch.autograd.grad(adv_loss, last_layer_weight, retain_graph=True)[0]

        lambda_adv = torch.norm(recon_loss_grads.float()) / (torch.norm(adv_loss_grads.float()) + 1e-4)
        lambda_adv = torch.clamp(lambda_adv, 0, 1e4).detach()
        return 0.8 * lambda_adv

    def optimize_parameters_amp(self, current_step, update=False):
        self.current_step = current_step
        dis_factor = 1.0 if current_step >= self.opt_train['D_start_iteration'] else 0.0

        # optimize D
        self.netD.requires_grad_(True)

        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            self.vq_forward()
            if dis_factor > 0.0:
                self.prop_real = self.netD_forward(self.vae_in)
                self.prop_fake = self.netD_forward(self.E.detach())
                self.dis_loss = 0.5 * (torch.mean(F.relu(1. - self.prop_real)) + torch.mean(F.relu(1. + self.prop_fake)))
                self.dis_loss = self.dis_loss / self.num_accum_steps_D
            else:
                self.dis_loss = torch.zeros(1, device=self.device)

        self.D_train_loss = self.dis_loss
        if self.opt['rank'] == 0:
            print("D train loss:", self.D_train_loss.item())

        if dis_factor > 0.0:
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
            recon_loss = compute_generator_loss(self.vae_in, self.E, self.loss_fn_dict, self.loss_val_dict, self.device)
            if dis_factor > 0.0:
                self.prop_fake = self.netD_forward(self.E)
                adv_loss = -torch.mean(self.prop_fake)
                self.lambda_adv = self.calculate_lambda(recon_loss, adv_loss)
                self.gen_loss = self.vq_loss + recon_loss + self.lambda_adv * adv_loss
            else:
                self.lambda_adv = 0.0
                self.gen_loss = self.vq_loss + recon_loss
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
        self.current_step = current_step
        dis_factor = 1.0 if current_step >= self.opt_train['D_start_iteration'] else 0.0

        # optimize D
        self.netD.requires_grad_(True)

        self.vq_forward()
        if dis_factor > 0.0:
            self.prop_real = self.netD_forward(self.vae_in)
            self.prop_fake = self.netD_forward(self.E.detach())
            self.dis_loss = 0.5 * (torch.mean(F.relu(1. - self.prop_real)) + torch.mean(F.relu(1. + self.prop_fake)))
            self.dis_loss = self.dis_loss / self.num_accum_steps_D
        else:
            self.dis_loss = torch.zeros(1, device=self.device)

        self.D_train_loss = self.dis_loss
        if self.opt['rank'] == 0:
            print("D train loss:", self.D_train_loss.item())

        if dis_factor > 0.0:
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

        recon_loss = compute_generator_loss(self.vae_in, self.E, self.loss_fn_dict, self.loss_val_dict, self.device)
        if dis_factor > 0.0:
            self.prop_fake = self.netD_forward(self.E)
            adv_loss = -torch.mean(self.prop_fake)
            self.lambda_adv = self.calculate_lambda(recon_loss, adv_loss)
            self.gen_loss = self.vq_loss + recon_loss + self.lambda_adv * adv_loss
        else:
            self.lambda_adv = 0.0
            self.gen_loss = self.vq_loss + recon_loss
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

        table = wandb.Table(
            data=[[d, frac] for d, frac in enumerate(self.frac_unique)],
            columns=["depth", "frac_unique"],
        )
        self.run.log({
            "step": current_step,
            "codebook_utilization": wandb.plot.bar(table, "depth", "frac_unique", title="Codebook Utilization per RQ Depth"),
        })

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
        self.vq_forward()
        self.prop_real = self.netD_forward(self.vae_in)
        self.prop_fake = self.netD_forward(self.E)

        recon_loss = compute_generator_loss(self.vae_in, self.E, self.loss_fn_dict, self.loss_val_dict, self.device)
        adv_loss = -torch.mean(self.prop_fake)
        self.gen_loss = self.vq_loss + recon_loss + self.lambda_adv * adv_loss
        self.dis_loss = 0.5 * (torch.mean(F.relu(1. - self.prop_real)) + torch.mean(F.relu(1. + self.prop_fake)))

        self.G_valid_loss += self.gen_loss
        self.D_valid_loss += self.dis_loss

        rescale_images = self.opt['dataset_opt']['norm_type'] == "znormalization"
        compute_performance_metrics(self.E, self.vae_in, self.metric_fn_dict, self.metric_val_dict, rescale_images)

    def validation_amp(self):
        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            self.vq_forward()
            self.prop_real = self.netD_forward(self.vae_in)
            self.prop_fake = self.netD_forward(self.E)

            recon_loss = compute_generator_loss(self.vae_in, self.E, self.loss_fn_dict, self.loss_val_dict, self.device)
            adv_loss = -torch.mean(self.prop_fake)
            self.gen_loss = self.vq_loss + recon_loss + self.lambda_adv * adv_loss
            self.dis_loss = 0.5 * (torch.mean(F.relu(1. - self.prop_real)) + torch.mean(F.relu(1. + self.prop_fake)))

        self.G_valid_loss += self.gen_loss
        self.D_valid_loss += self.dis_loss

        rescale_images = self.opt['dataset_opt']['norm_type'] == "znormalization"
        compute_performance_metrics(self.E, self.vae_in, self.metric_fn_dict, self.metric_val_dict, rescale_images)

    def current_visuals(self):
        out_dict = OrderedDict()
        out_dict['H'] = self.H.detach()[0].float().cpu()
        out_dict['E_vq'] = self.E.detach()[0].float().cpu()
        net = self.get_bare_model(self.netG)
        if self.mixed_precision is not None:
            with torch.amp.autocast("cuda", dtype=self.mixed_precision):
                E_no_vq = net.decode(self.z_e)
        else:
            E_no_vq = net.decode(self.z_e)
        out_dict['E_no_vq'] = E_no_vq.detach()[0].float().cpu()
        return out_dict

    def log_comparison_image(self, img_dict, current_step, out_dtype=np.uint8):
        unnorm = self.opt['dataset_opt']['norm_type'] == 'znormalization'
        slice_idx = img_dict['H'].shape[-1] // 2

        E_vq_slice    = img_dict['E_vq'][:, :, :, slice_idx]
        E_no_vq_slice = img_dict['E_no_vq'][:, :, :, slice_idx]
        H_slice       = img_dict['H'][:, :, :, slice_idx]

        row = torch.stack([E_vq_slice, E_no_vq_slice, H_slice])
        grid = make_grid(row, nrow=len(row), padding=0).permute(1, 2, 0)
        grid_image = utils_3D_image.unnorm_and_rescale(grid, out_dtype, unnorm=unnorm)

        figure_string = "VQGAN: %s, step %d: VQ Recon | No-VQ Recon | HR" % (
            self.opt['model_opt']['model_architecture'], current_step
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

    def current_log(self):
        return self.log_dict
