import torch
import wandb
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel

from loss_functions.loss_functions_simple import compute_generator_loss
from models.model_plain import ModelPlain
from models.resshift import build_resshift_diffusion
from performance_metrics.performance_metrics import compute_performance_metrics


class ModelResShift(ModelPlain):
    """ResShift 2D SR baseline (pixel-space, no autoencoder).

    Reuses the residual-shifting Markov-chain diffusion from
    https://github.com/zsyOAOA/ResShift (vendored under ``models/resshift``).

    Only the generator training/sampling differs from :class:`ModelPlain`:
      * training: predict x0 from a residual-shifted noisy sample at a random
        timestep and minimize the (optionally weighted) MSE.
      * inference: run the ``num_timesteps``-step reverse chain from the
        upsampled-LR prior.

    The low-res image ``self.L`` plays two roles, exactly as in ResShift: it is
    the Markov-chain endpoint (bicubically upsampled to HR size) *and* the
    conditioning ``lq`` concatenated inside the UNet. Both are the same upsampled
    tensor, produced here via ``diffusion.encode_first_stage(L, None,
    up_sample=True)`` so training and sampling stay consistent.
    """

    def __init__(self, opt, mode='train', data_parallel=True):
        super(ModelResShift, self).__init__(opt, mode=mode, data_parallel=data_parallel)

        diffusion_opt = OmegaConf.to_container(opt['model_opt']['diffusion'], resolve=True)
        diffusion_opt.setdefault('sf', opt['up_factor'])
        self.diffusion = build_resshift_diffusion(diffusion_opt)
        self.clip_denoised = diffusion_opt.get('clip_denoised', True)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _sampling_model(self):
        """Prefer the EMA weights for sampling (ResShift relies on them)."""
        if self.opt_train['E_decay'] > 0 and hasattr(self, 'netE'):
            return self.netE
        return self.netG

    def _lq_cond(self, L):
        """Upsampled-LR used as both endpoint and UNet conditioning."""
        return self.diffusion.encode_first_stage(L, None, up_sample=True)

    # ------------------------------------------------------------------
    # inference: reverse diffusion chain -> self.E
    # ------------------------------------------------------------------
    def netG_forward(self):
        model = self._sampling_model()
        lq_up = self._lq_cond(self.L)
        self.E = self.diffusion.p_sample_loop(
            y=self.L,
            model=model,
            first_stage_model=None,
            clip_denoised=self.clip_denoised,
            model_kwargs={'lq': lq_up},
            device=self.device,
        )

    # ------------------------------------------------------------------
    # diffusion training loss (replaces the feed-forward pixel loss)
    # ------------------------------------------------------------------
    def _diffusion_loss(self):
        b = self.H.shape[0]
        t = torch.randint(0, self.diffusion.num_timesteps, size=(b,), device=self.device).long()
        lq_up = self._lq_cond(self.L)
        terms, _, _ = self.diffusion.training_losses(self.netG, self.H, self.L, t, model_kwargs={'lq': lq_up})
        return terms['mse'].mean()

    def optimize_parameters_amp(self, current_step, update=False):
        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            self.gen_loss = self._diffusion_loss() / self.num_accum_steps_G

        self.G_train_loss = self.gen_loss
        if self.opt['rank'] == 0:
            print("G train loss:", self.G_train_loss.item())

        self.update = ((self.G_accum_count + 1) % self.num_accum_steps_G) == 0 or update

        if not self.update and isinstance(self.netG, DistributedDataParallel):
            with self.netG.no_sync():
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
        self.gen_loss = self._diffusion_loss() / self.num_accum_steps_G

        self.G_train_loss = self.gen_loss
        if self.opt['rank'] == 0:
            print("G train loss:", self.G_train_loss.item())

        self.update = ((self.G_accum_count + 1) % self.num_accum_steps_G) == 0 or update

        if not self.update and isinstance(self.netG, DistributedDataParallel):
            with self.netG.no_sync():
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
