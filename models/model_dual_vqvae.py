from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torchvision.utils import make_grid

from loss_functions.loss_functions_simple import compute_generator_loss
from models.model_base import ModelBase
from models.select_network import define_G
from performance_metrics.performance_metrics import compute_performance_metrics
from utils import utils_3D_image


class ModelDualVQVAE(ModelBase):
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
    # Initialisation
    # -------------------------------------------------------------------------

    def __init__(self, opt, mode='train', data_parallel=True):
        super(ModelDualVQVAE, self).__init__(opt)
        self.last_iteration = 0
        self.netG = define_G(opt, mode=mode)
        self.netG = self.model_to_device(self.netG, data_parallel=data_parallel)

        if opt['rank'] == 0 and mode == 'train':
            print("Number of trainable parameters, G", utils_3D_image.numel(self.netG, only_trainable=True))

        self.update = False

        self.early_stop = False
        self.min_validation_loss = float('inf')
        self.patience = self.opt_train['early_stop_patience']
        self.patience_counter = 0
        self.min_delta = 0

    def set_eval_mode(self):
        self.netG.eval()

    def set_train_mode(self):
        self.netG.train()

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
        eid = self._resolve_eid(experiment_id)
        if mode == 'train':
            if self.opt['train_mode'] == 'scratch':
                return
            assert eid is not None, f"Pretrained experiment ID required for train_mode='{self.opt['train_mode']}'."
        else:
            assert eid is not None, 'Experiment ID required for test mode.'
        self.load_G(eid, mode)

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

        self.lambda_recon = self.opt_train["lambda_recon"]
        self.lambda_feat = self.opt_train["lambda_feat"]
        self.lambda_distill = self.opt_train["lambda_distill"]
        self.lambda_align = self.opt_train["lambda_align"]

    def define_optimizer(self):

        main_params, star_params = self.get_bare_model(self.netG).split_optimizer_params()

        if self.opt['rank'] == 0:
            print(f"VQ model params: {sum(p.numel() for p in main_params):,}")
            print(f"E* params: {sum(p.numel() for p in star_params):,}")

        self.define_G_optimizer(main_params)
        self.define_star_optimizer(star_params)

    def define_gradscaler(self):
        self.gen_scaler = torch.amp.GradScaler("cuda")
        self.star_scaler = torch.amp.GradScaler("cuda")

    def define_scheduler(self):
        self.define_G_scheduler()
        self.define_star_scheduler()

    # -------------------------------------------------------------------------
    # Checkpoint overrides — persist star optimiser and scaler alongside G
    # -------------------------------------------------------------------------

    def save(self, iter_label):
        self.save_G(iter_label)
        self.save_star_optimizer(iter_label)
        if self.mixed_precision is not None:
            self.save_star_gradscaler(iter_label)

    def load_optimizers(self, experiment_id=None):
        if self.opt['train_mode'] != 'resume':
            return
        eid = self._resolve_eid(experiment_id)
        assert eid is not None, "Pretrained experiment ID required for train_mode='resume'."
        self.opt['train_opt']['G_optimizer_reuse'] = True
        self.load_G_optimizer(eid)
        self.load_star_optimizer(eid)

    def load_schedulers(self, experiment_id=None):
        if self.opt['train_mode'] != 'resume':
            return
        eid = self._resolve_eid(experiment_id)
        assert eid is not None, "Pretrained experiment ID required for train_mode='resume'."
        self.load_G_scheduler(eid)

    def load_gradscalers(self, experiment_id=None):
        if self.opt['train_mode'] != 'resume':
            return
        eid = self._resolve_eid(experiment_id)
        assert eid is not None, "Pretrained experiment ID required for train_mode='resume'."
        self.load_G_gradscaler(eid)
        self.load_star_gradscaler(eid)

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
        self.E, self.vq_loss, self.codes, self.z_e_down, self.frac_unique = self.netG(self.L)

    def vq_forward_star(self):
        self.E_star, self.vq_loss_star, self.codes_star, self.z_e_star, self.frac_unique_star = self.netG(self.L_star, star_mode=True)

    def netG_forward(self):
        self.E, _, _, _, _ = self.netG(self.L)  # Always L as inference_zarr expects this

    def compute_code_align_loss(self):
        """Differentiable per-depth code-alignment CE for E*.

        The RQ bottleneck computes distances under ``@torch.no_grad`` and
        detaches the residual, so distances returned by the quantizer carry no
        gradient to E*. Here we recompute distances against the codebook
        weights, teacher-forcing each depth's residual with E's codes so E*'s
        depth-d logits are scored against the same residual as E's target.

        """
        q = self.get_bare_model(self.netG).quantizer
        # (B, C, Dz, Dy, Dx) -> (B, Dz, Dy, Dx, C)
        x = self.z_e_star.permute(0, 2, 3, 4, 1).contiguous()
        spatial_shape = x.shape[:-1]           # (B, Dz, Dy, Dx)
        C = x.shape[-1]

        residual = x
        logits_per_depth = []
        codes_e = self.codes.detach().long()   # (B, Dz, Dy, Dx, n_rq_depth)

        for d, cb in enumerate(q.codebooks):
            w = cb.weight[:-1, :]                                        # (n_embed, C), no-grad
            r_flat = residual.reshape(-1, C).to(w.dtype)
            # Logits = -||r - e||^2. The ||r||^2 term is constant across n_embed
            # and cancels inside log_softmax, so we omit it.
            neg_dist_sq = 2.0 * (r_flat @ w.t()) - w.pow(2).sum(dim=1)   # (N, n_embed)
            logits_per_depth.append(neg_dist_sq.reshape(*spatial_shape, -1))

            # Teacher-forced residual for the next depth. Detach so gradient
            # only flows through the current depth's logits.
            chosen = w[codes_e[..., d]]                                  # (B, Dz, Dy, Dx, C)
            residual = residual - chosen.detach()

        logits = torch.stack(logits_per_depth, dim=-1)                   # (B, Dz, Dy, Dx, n_embed, n_rq_depth)
        logits = logits.permute(0, 4, 1, 2, 3, 5)                        # (B, n_embed, Dz, Dy, Dx, n_rq_depth)
        return F.cross_entropy(logits, codes_e)

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

        net = self.get_bare_model(self.netG)  # used to access model-level functions

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
        with net.frozen_decoder():
            with torch.amp.autocast("cuda", dtype=self.mixed_precision):
                self.vq_forward_star()
                self.star_loss = self.vq_loss_star

                # Auxiliary losses for E* training. Controlled by lambda_* hyperparameters.
                if self.lambda_recon > 0:
                    recon_loss_star = compute_generator_loss(self.L, self.E_star, self.loss_fn_dict, self.loss_val_dict)
                    self.star_loss += self.lambda_recon * recon_loss_star
                if self.lambda_feat > 0:
                    with net.frozen_encoder():
                        feat_loss_star = F.mse_loss(net.encode(self.E_star), self.z_e_down.detach())
                        self.star_loss += self.lambda_feat * feat_loss_star
                if self.lambda_distill > 0:
                    distill_loss = F.mse_loss(self.z_e_star, self.z_e_down.detach())
                    self.star_loss += self.lambda_distill * distill_loss
                if self.lambda_align > 0:
                    code_align_loss = self.compute_code_align_loss()
                    self.star_loss += self.lambda_align * code_align_loss

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

    def validation(self):
        self.vq_forward()

        recon_loss = compute_generator_loss(self.L, self.E, self.loss_fn_dict, self.loss_val_dict)
        self.gen_loss = self.vq_loss + recon_loss

        self.G_valid_loss += self.gen_loss

        rescale_images = self.opt['dataset_opt']['norm_type'] == "znormalization"
        compute_performance_metrics(self.E, self.L, self.metric_fn_dict, self.metric_val_dict, rescale_images)

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

        self.run.log({"step": current_step, "code_agreement_rate": self.match_rate.item()})
        for depth_idx, rate in enumerate(self.match_rate_per_depth):
            self.run.log({"step": current_step, f"code_agreement_rate_depth{depth_idx}": rate.item()})

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

    def current_log(self):
        return self.log_dict

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
        unnorm = self.opt['dataset_opt']['norm_type'] == 'znormalization'
        slice_idx = img_dict['L'].shape[-1] // 2
        E_slice = img_dict['E'][:, :, :, slice_idx]
        L_slice = img_dict['L'][:, :, :, slice_idx]
        E_star_slice = img_dict['E_star'][:, :, :, slice_idx]
        L_star_slice = img_dict['L_star'][:, :, :, slice_idx]

        row = torch.stack([E_slice, L_slice, E_star_slice, L_star_slice])
        grid = make_grid(row, nrow=len(row), padding=0).permute(1, 2, 0)
        grid_image = utils_3D_image.unnorm_and_rescale(grid, out_dtype, unnorm=unnorm)

        # Down VQ recon | Down | REG VQ recon | REG
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
