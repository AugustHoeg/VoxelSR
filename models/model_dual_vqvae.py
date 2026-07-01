from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim import Adam, AdamW
from torchvision.utils import make_grid

from loss_functions.loss_functions_simple import compute_generator_loss
from models.model_vqvae import ModelVQVAE
from performance_metrics.performance_metrics import compute_performance_metrics
from utils import utils_3D_image


class ModelDualVQVAE(ModelVQVAE):
    """
    Joint training of DualRQVAE3D.

    Step 1 (main): x_down → E → Q (EMA updated) → D → recon_loss(x_down)
                   Updates: encoder, pre_quant_conv, decoder, post_quant_conv.

    Step 2 (star): x_star → E* → Q (eval, no EMA) → D (frozen grad) → recon_loss(x_down)
                             + lambda_distill * ||E*(x_star) - sg[E(x_down)]||_2
                   Updates: encoder_star, pre_quant_conv_star only.

    DDP note: both steps call self.netG(...) so DDP gradient sync hooks fire
    correctly. Requires find_unused_parameters=True in DDP because E is unused
    in the star step and E* is unused in the main step.
    """

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _unwrap(self):
        net = self.netG
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

    # -------------------------------------------------------------------------
    # Initialisation overrides
    # -------------------------------------------------------------------------

    def define_optimizer(self):
        self.G_accum_count = 0
        self.star_accum_count = 0
        self.num_accum_steps_G = self.opt_train["num_accum_steps_G"]
        self.num_accum_steps_star = self.opt_train["num_accum_steps_G"]  # Assume same no. of accumulation steps
        self.lambda_distill = self.opt_train["lambda_distill"]
        self.lambda_align = self.opt_train["lambda_align"]

        net = self._unwrap()
        star_prefixes = ('encoder_star.', 'pre_quant_conv_star.')
        main_params, star_params = [], []
        for name, p in net.named_parameters():
            if any(name.startswith(pfx) for pfx in star_prefixes):
                if p.requires_grad:
                    star_params.append(p)
            else:
                if p.requires_grad:
                    main_params.append(p)

        if self.opt['rank'] == 0:
            print(f"VQ model params: {sum(p.numel() for p in main_params):,}")
            print(f"E* params: {sum(p.numel() for p in star_params):,}")

        if self.opt_train["G_optimizer_type"] == "adam":
            self.G_optimizer = Adam(main_params, lr=self.opt_train["G_optimizer_lr"], weight_decay=self.opt_train["G_optimizer_wd"], betas=(0.9, 0.999))
            self.star_optimizer = Adam(star_params, lr=self.opt_train["G_optimizer_lr"], weight_decay=self.opt_train["G_optimizer_wd"], betas=(0.9, 0.999))
        elif self.opt_train["G_optimizer_type"] == "adamw":
            self.G_optimizer = AdamW(main_params, lr=self.opt_train["G_optimizer_lr"], weight_decay=self.opt_train["G_optimizer_wd"], betas=(0.9, 0.999))
            self.star_optimizer = AdamW(star_params, lr=self.opt_train["G_optimizer_lr"], weight_decay=self.opt_train["G_optimizer_wd"],  betas=(0.9, 0.999))
        else:
            raise NotImplementedError(f"Optimizer [{self.opt_train['G_optimizer_type']}] is not implemented.")

        self.G_train_grad_norm = torch.zeros(1)
        self.star_train_grad_norm = torch.zeros(1)

    def define_gradscaler(self):
        self.gen_scaler = torch.amp.GradScaler("cuda")
        self.star_scaler = torch.amp.GradScaler("cuda")

    def define_scheduler(self):
        self.define_G_scheduler()
        star_scheduler = self._build_scheduler(
            self.star_optimizer,
            milestones_key='G_scheduler_milestones',
            gamma_key='G_scheduler_gamma',
            warmup_steps_key='G_scheduler_warmup_steps',
        )
        self.schedulers.append(star_scheduler)

    # -------------------------------------------------------------------------
    # Checkpoint overrides — persist star optimiser and scaler alongside G
    # -------------------------------------------------------------------------

    def save_G(self, iter_label):
        super().save_G(iter_label)
        self.save_optimizer(
            self._run_dir("saved_optimizers"), self.star_optimizer, 'optimizerStar', iter_label
        )
        if self.mixed_precision is not None:
            self.save_gradscaler(
                self._run_dir("saved_gradscalers"), self.star_scaler, 'gradscalerStar', iter_label
            )

    def load_optimizers(self, experiment_id=None):
        super().load_optimizers(experiment_id)
        if self.opt['train_mode'] != 'resume':
            return
        eid = self._resolve_eid(experiment_id)
        path = self._find_latest_checkpoint(eid, "saved_optimizers", "*optimizerStar.h5")
        if path is not None:
            if self.opt['rank'] == 0:
                print(f"Loading star optimizer [{self._short_path(path)}] ...")
            self.load_optimizer(path, self.star_optimizer)

    def load_gradscalers(self, experiment_id=None):
        super().load_gradscalers(experiment_id)
        if self.opt['train_mode'] != 'resume':
            return
        eid = self._resolve_eid(experiment_id)
        path = self._find_latest_checkpoint(eid, "saved_gradscalers", "*gradscalerStar.h5")
        if path is not None:
            if self.opt['rank'] == 0:
                print(f"Loading star gradscaler [{self._short_path(path)}] ...")
            self.load_gradscaler(path, self.star_scaler)

    # -------------------------------------------------------------------------
    # Data and forward passes
    # -------------------------------------------------------------------------

    def feed_data(self, data):
        self.L = data['L'].as_tensor().to(self.device, non_blocking=True)
        if 'REG' in data:
            self.L_star = data['REG'].as_tensor().to(self.device, non_blocking=True)
        else:
            self.L_star = self.L  # fallback to downsampled LR if REG not provided
        self.H = data['H'].as_tensor().to(self.device, non_blocking=True)
        self.vae_in = self.L  # kept for compatibility with inherited validation logging

    def vq_forward(self):
        self.E, self.vq_loss, self.codes, self.z_e_down, self.dists = self.netG(self.L)

    def vq_forward_star(self):
        self.E_star, self.vq_loss_star, self.codes_star, self.z_e_star, self.dists_star = self.netG(self.L_star, star_mode=True)

    @staticmethod
    def get_code_match_rate(codes, codes_star):
        """Fraction of positions where E and E* select the same codebook entry.

        codes, codes_star: (..., n_rq_depth) LongTensor. Agreement is averaged
        over all leading dims, so this works regardless of spatial shape.

        Returns:
            overall_rate: scalar tensor
            per_depth_rate: (n_rq_depth,) tensor
        """
        code_match = (codes == codes_star).float()
        reduce_dims = tuple(range(code_match.ndim - 1))
        overall_rate = code_match.mean()
        per_depth_rate = code_match.mean(dim=reduce_dims)
        return overall_rate, per_depth_rate

    # -------------------------------------------------------------------------
    # Training step
    # -------------------------------------------------------------------------

    def optimize_parameters_amp(self, current_step, update=False):
        net = self._unwrap()

        # ── Step 1: update E + D from downsampled LR ─────────────────────────
        with (torch.amp.autocast("cuda", dtype=self.mixed_precision)):
            self.vq_forward()
            recon_loss = compute_generator_loss(self.L, self.E, self.loss_fn_dict, self.loss_val_dict)
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
            G_clip = self.opt_train['G_optimizer_clipgrad']
            if G_clip > 0:
                self.gen_scaler.unscale_(self.G_optimizer)
                self.G_train_grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for g in self.G_optimizer.param_groups for p in g['params']],
                    max_norm=G_clip,
                )
            self.gen_scaler.step(self.G_optimizer)
            self.gen_scaler.update()
            self.G_optimizer.zero_grad()
            self.G_accum_count = 0
        else:
            self.G_accum_count += 1

        # ── Step 2: update E* from real LR ───────────────────────────────────
        with (net.frozen_decoder()):
            with torch.amp.autocast("cuda", dtype=self.mixed_precision):
                self.vq_forward_star()
                recon_loss_star = compute_generator_loss(self.L, self.E_star, self.loss_fn_dict, self.loss_val_dict)
                distill_loss = F.mse_loss(self.z_e_star, self.z_e_down.detach())
                self.distill_train_loss = distill_loss.detach()
                code_align_loss = F.cross_entropy(-self.dists_star.permute(0, 4, 1, 2, 3, 5), self.codes.detach().long())
                self.star_loss = self.vq_loss_star + recon_loss_star + self.lambda_distill * distill_loss + self.lambda_align * code_align_loss
                self.star_loss = self.star_loss / self.num_accum_steps_star

                self.match_rate, self.match_rate_per_depth = self.get_code_match_rate(self.codes, self.codes_star)

            self.star_train_loss = self.star_loss
            if self.opt["rank"] == 0:
                print("Star train loss:", self.star_train_loss.item())

            self.star_update = ((self.star_accum_count + 1) % self.num_accum_steps_star) == 0 or update

            if not self.star_update:
                if isinstance(self.netG, DistributedDataParallel):
                    with self.netG.no_sync():
                        self.star_scaler.scale(self.star_loss).backward()
                else:
                    self.star_scaler.scale(self.star_loss).backward()
            else:
                self.star_scaler.scale(self.star_loss).backward()

        if self.star_update:
            star_clip = self.opt_train['G_optimizer_clipgrad']
            if star_clip > 0:
                self.star_scaler.unscale_(self.star_optimizer)
                self.star_train_grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for g in self.star_optimizer.param_groups for p in g['params']],
                    max_norm=star_clip,
                )
            self.star_scaler.step(self.star_optimizer)
            self.star_scaler.update()
            self.star_optimizer.zero_grad()
            self.star_accum_count = 0
        else:
            self.star_accum_count += 1

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validation_amp(self):
        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            self.vq_forward()
            recon_loss = compute_generator_loss(
                self.L, self.E, self.loss_fn_dict, self.loss_val_dict
            )
            self.gen_loss = self.vq_loss + recon_loss

        self.G_valid_loss += self.gen_loss

        rescale_images = self.opt['dataset_opt']['norm_type'] == "znormalization"
        compute_performance_metrics(
            self.E, self.L, self.metric_fn_dict, self.metric_val_dict, rescale_images
        )

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def record_train_log(self, current_step):
        loss = self.G_train_loss.item() * self.num_accum_steps_G
        self.run.log({"step": current_step, "G_train_loss": loss})

        star_loss = self.star_train_loss.item() * self.num_accum_steps_star
        self.run.log({"step": current_step, "star_train_loss": star_loss})

        grad_norm = self.G_train_grad_norm.item() * self.num_accum_steps_G
        self.run.log({"step": current_step, "G_train_grad_norm": grad_norm})

        star_grad_norm = self.star_train_grad_norm.item() * self.num_accum_steps_star
        self.run.log({"step": current_step, "star_train_grad_norm": star_grad_norm})

        self.run.log({"step": current_step, "distill_train_loss": self.distill_train_loss.item()})

        self.run.log({"step": current_step, "code_agreement_rate": self.match_rate.item()})
        for depth_idx, rate in enumerate(self.match_rate_per_depth):
            self.run.log({"step": current_step, f"code_agreement_rate_depth{depth_idx}": rate.item()})

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    def current_visuals(self):
        out_dict = OrderedDict()

        if self.mixed_precision is not None:
            with torch.amp.autocast("cuda", dtype=self.mixed_precision):
                self.vq_forward()
                self.vq_forward_star()
        else:
            self.vq_forward()
            self.vq_forward_star()

        out_dict['L'] = self.L.detach()[0].float().cpu()
        out_dict['L_star'] = self.L_star.detach()[0].float().cpu()
        out_dict['E'] = self.E.detach()[0].float().cpu()
        out_dict['E_star'] = self.E_star.detach()[0].float().cpu()

        return out_dict

    def log_comparison_image(self, img_dict, current_step, out_dtype=np.uint8):
        slice_idx = img_dict['L'].shape[-1] // 2
        E_slice = img_dict['E'][:, :, :, slice_idx]
        E_star_slice = img_dict['E_star'][:, :, :, slice_idx]
        L_star_slice = img_dict['L_star'][:, :, :, slice_idx]
        L_slice = img_dict['L'][:, :, :, slice_idx]

        row = torch.stack([E_slice, L_slice, E_star_slice, L_star_slice])
        grid = make_grid(row, nrow=len(row), padding=0).permute(1, 2, 0)
        grid_image = utils_3D_image.unnorm_and_rescale(grid, out_dtype)

        # Down recon, Down, REG recon, REG
        figure_string = "%s: step %d" % (self.opt["model_opt"]["model_architecture"], current_step)

        if self.opt['run_type'] == "HOME PC":
            height, width = grid_image.shape[:2]
            plt.figure(figsize=(2 * self.opt['up_factor'] * width / 100, 2 * self.opt['up_factor'] * height / 100), dpi=100)
            plt.imshow(grid_image, vmin=0, vmax=255)
            plt.title(figure_string)
            plt.xticks([])
            plt.yticks([])
            plt.show()

        wandb.log({"Comparisons training": wandb.Image(grid_image, caption=figure_string, mode="RGB")})
