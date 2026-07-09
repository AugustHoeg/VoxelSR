import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam, AdamW
from torchvision.utils import make_grid

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
        self.netG = self.model_to_device(self.netG, data_parallel=data_parallel)

        if opt['rank'] == 0 and mode == 'train':
            print("Number of trainable parameters", utils_3D_image.numel(self.netG, only_trainable=True))

        self.vae_target = opt.get('vae_target', 'HR')
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

    def define_wandb_run(self):
        self._init_wandb_run(extra_config={"up_factor": self.opt['up_factor']})
        self.model_artifact_G = wandb.Artifact(
            "Generator", type=self.opt['model_opt']['netG']['net_type'],
            description=self.opt['model_opt']['netG']['description'],
            metadata=OmegaConf.to_container(self.opt['model_opt']['netG'], resolve=True)
        )

    def define_loss(self):
        self.build_loss_fn_dict()
        self.init_G_loss_trackers()

    def define_optimizer(self):
        self.define_G_optimizer()

    def feed_data(self, data):
        self.L = data['L'].as_tensor().to(self.device, non_blocking=True)
        self.H = data['H'].as_tensor().to(self.device, non_blocking=True)
        self.vae_in = self.H if self.vae_target == 'HR' else self.L

    def vq_forward(self):
        self.E, self.vq_loss, self.codes, self.z_e, self.frac_unique = self.netG(self.vae_in)

    def netG_forward(self):
        self.E, _, _, _, _ = self.netG(self.L)

    def optimize_parameters_amp(self, current_step, update=False):

        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            self.vq_forward()
            recon_loss = compute_generator_loss(self.vae_in, self.E, self.loss_fn_dict, self.loss_val_dict)
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

        table = wandb.Table(
            data=[[d, frac.item()] for d, frac in enumerate(self.frac_unique)],
            columns=["depth", "frac_unique"],
        )
        self.run.log({
            "step": current_step,
            "codebook_utilization": wandb.plot.bar(table, "depth", "frac_unique", title="Codebook Utilization per RQ Depth"),
        })

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

        recon_loss = compute_generator_loss(self.vae_in, self.E, self.loss_fn_dict, self.loss_val_dict)
        self.gen_loss = self.vq_loss + recon_loss

        self.G_valid_loss += self.gen_loss

        rescale_images = self.opt['dataset_opt']['norm_type'] == "znormalization"
        compute_performance_metrics(self.E, self.vae_in, self.metric_fn_dict, self.metric_val_dict, rescale_images)

    def validation_amp(self):
        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            self.vq_forward()
            recon_loss = compute_generator_loss(self.vae_in, self.E, self.loss_fn_dict, self.loss_val_dict)
            self.gen_loss = self.vq_loss + recon_loss

        self.G_valid_loss += self.gen_loss

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

        figure_string = "VQVAE: %s, step %d: VQ Recon | No-VQ Recon | HR" % (
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
