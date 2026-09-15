import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from torch.optim import Adam, AdamW

from loss_functions.loss_functions_simple import compute_generator_loss
from models.model_base import ModelBase
from models.osediff import OSEDiff_gen, OSEDiff_reg
from performance_metrics.performance_metrics import compute_performance_metrics
from utils import utils_3D_image
from utils.utils_dist import get_rank, reduce_sum


class ModelOSEDiff(ModelBase):
    """OSEDiff one-step diffusion SR baseline (2D), wrapping the vendored
    ``models/osediff`` networks into VoxelSR's ModelBase training loop.

    Inherits :class:`ModelBase` directly (not the GAN model) and defines the two
    networks explicitly:

      * ``self.netG``   -- :class:`OSEDiff_gen`: LoRA VAE-encoder + LoRA UNet +
        frozen VAE-decoder; produces the one-step SR image.
      * ``self.netReg`` -- :class:`OSEDiff_reg`: the VSD regularizer (a trainable
        "fake-score" UNet + a frozen pretrained UNet). It is *not* a GAN
        discriminator; it provides the distribution-matching (score distillation)
        signal and its own diffusion (denoising) loss.

    Two optimizers are stepped every real update, mirroring upstream
    ``train_osediff.py``:
      * generator: recon (MSE + LPIPS, via ``compute_generator_loss``) + lambda_vsd * VSD
      * regularizer: lambda_vsd_lora * diff_loss(generated latent)

    Conventions bridged for the pretrained SD stack:
      * VoxelSR 2D data is single-channel in ``[0, 1]``; OSEDiff's VAE works on
        3-channel ``[-1, 1]``. The LR is bicubically pre-upsampled to HR size
        (OSEDiff refines at output resolution, like ResShift's ``_lq_cond``),
        grayscale is replicated to 3 channels, and ``[0,1]<->[-1,1]`` mapping is
        applied around the VAE. Outputs are collapsed back to 1 channel in
        ``[0, 1]`` so losses/metrics/visuals match the other baselines.
        -> use a ``[0, 1]`` norm (``scale_intensity``) in the dataset config.
      * Text conditioning is a fixed **null prompt** (``""``); RAM/DAPE captioning
        is undefined for the volumetric domain (with equal pos/neg prompts the
        classifier-free-guidance term in VSD simply vanishes).

    Scope notes (first version):
      * Single-device (networks placed on ``self.device``; no DataParallel/DDP
        wrapping -- DDP over the mostly-frozen SD + LoRA is a follow-up).
      * No EMA (OSEDiff has none); set ``train_opt.E_decay: 0`` in the config.
      * Checkpoints store LoRA (+ UNet ``conv_in``) only -- small and faithful to
        upstream ``save_model`` -- following ModelBase's checkpoint file layout so
        ``train_mode: resume`` works.
    """

    def __init__(self, opt, mode='train', data_parallel=True):
        super(ModelOSEDiff, self).__init__(opt)
        self.last_iteration = 0

        net = opt['model_opt']['netG']
        self.osediff_args = SimpleNamespace(
            pretrained_model_name_or_path=net['pretrained_model_name_or_path'],
            lora_rank=net.get('lora_rank', 4),
            cfg_vsd=net.get('cfg_vsd', 7.5),
            device=self.device,
            mixed_precision='no',
            use_tiled_vae=False,
            merge_and_unload_lora=False,
            latent_tiled_size=net.get('latent_tiled_size', 96),
            latent_tiled_overlap=net.get('latent_tiled_overlap', 32),
        )

        # null-prompt conditioning (see class docstring)
        self.prompt = net.get('prompt', "")
        self.neg_prompt = net.get('neg_prompt', "")

        # VSD loss weights (upstream defaults: lambda_vsd=1, lambda_vsd_lora=1)
        self.lambda_vsd = self.opt_train.get('lambda_vsd', 1.0)
        self.lambda_vsd_lora = self.opt_train.get('lambda_vsd_lora', 1.0)

        self.netG = OSEDiff_gen(self.osediff_args)
        if mode == 'train':
            self.netReg = OSEDiff_reg(self.osediff_args, device=self.device, weight_dtype=torch.float32)

        if opt['rank'] == 0 and mode == 'train':
            print("Number of trainable parameters, G", utils_3D_image.numel(self.netG, only_trainable=True))

        self.update = False
        self.early_stop = False
        self.min_validation_loss = float('inf')
        self.patience = self.opt_train['early_stop_patience']
        self.patience_counter = 0
        self.min_delta = 0

    # ------------------------------------------------------------------
    # range / channel helpers  ([0,1] data  <->  [-1,1] 3-channel SD VAE)
    # ------------------------------------------------------------------
    @staticmethod
    def _to_pm1(x):
        return x * 2.0 - 1.0

    @staticmethod
    def _to_01(x):
        return (x + 1.0) * 0.5

    def _prep_lr(self, L):
        """LR ([0,1], 1ch, LR size) -> SD input ([-1,1], 3ch, HR size)."""
        up = F.interpolate(L, scale_factor=self.opt['up_factor'], mode='bicubic', align_corners=False)
        up = up.clamp(0.0, 1.0)
        if up.shape[1] == 1:
            up = up.repeat(1, 3, 1, 1)
        return self._to_pm1(up).to(self.device)

    def _null_batch(self, b):
        return {'prompt': [self.prompt] * b, 'neg_prompt': [self.neg_prompt] * b}

    # ------------------------------------------------------------------
    # forward: one-step SR -> self.E (1ch, [0,1]); returns the SR prediction
    # ------------------------------------------------------------------
    def _gen_forward(self):
        """Run OSEDiff_gen; set self.E and cache latent/prompt embeds for VSD."""
        c_t = self._prep_lr(self.L)
        out, latent, prompt_embeds, neg_prompt_embeds = self.netG(
            c_t, batch=self._null_batch(c_t.shape[0]), args=self.osediff_args
        )
        # collapse replicated RGB back to 1 channel, map [-1,1] -> [0,1]
        self.E = self._to_01(out).mean(dim=1, keepdim=True).clamp(0.0, 1.0)
        return out, latent, prompt_embeds, neg_prompt_embeds

    def netG_forward(self):
        """Produce the SR prediction (``self.E``) and return it. This is the
        single entry point the strided-inference logic needs."""
        out, latent, prompt_embeds, neg_prompt_embeds = self._gen_forward()
        return self.E

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------
    def init_test(self, experiment_id):
        self.load(experiment_id, mode='test')
        self.netG.eval()
        self.define_metrics()
        self.define_mixed_precision()
        self.define_visual_eval()

    def init_train(self):
        self.load()
        self.get_bare_model(self.netG).set_train()
        self.get_bare_model(self.netReg).set_train()

        self.define_loss()
        self.define_metrics()

        self.define_optimizer()
        self.load_optimizers()

        self.define_mixed_precision()
        self.load_gradscalers()

        self.define_scheduler()
        self.load_schedulers()

        self.define_visual_eval()

    def set_train_mode(self):
        self.netG.train()
        self.netReg.train()

    def set_eval_mode(self):
        self.netG.eval()

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
        self.Reg_train_loss = 0.0
        self.Reg_train_grad_norm = torch.zeros(1)

    # ------------------------------------------------------------------
    # optimizers / schedulers / gradscalers  (generator + regularizer)
    # ------------------------------------------------------------------
    def define_optimizer(self):
        self.g_params = [p for p in self.get_bare_model(self.netG).parameters() if p.requires_grad]
        self.define_G_optimizer(self.g_params)

        self.reg_params = [p for p in self.get_bare_model(self.netReg).parameters() if p.requires_grad]
        opt_t = self.opt_train
        OptCls = AdamW if opt_t["G_optimizer_type"] == "adamw" else Adam
        self.reg_optimizer = OptCls(
            self.reg_params, lr=opt_t["G_optimizer_lr"],
            weight_decay=opt_t["G_optimizer_wd"], betas=opt_t["G_optimizer_betas"],
        )
        self.reg_accum_count = 0

    def define_scheduler(self):
        self.define_G_scheduler()  # schedulers[0]
        self.schedulers.append(
            self._build_scheduler(
                self.reg_optimizer,
                milestones_key="G_scheduler_milestones",
                gamma_key="G_scheduler_gamma",
                warmup_steps_key="G_warmup_steps",
                eta_min_key="G_eta_min",
                scheduler_type=self.opt_train.get("G_scheduler_type", "MultiStepLR"),
            )
        )

    def define_gradscaler(self):
        self.define_G_gradscaler()  # self.gen_scaler
        self.reg_scaler = torch.amp.GradScaler("cuda")

    # ------------------------------------------------------------------
    # checkpoint save / load  (LoRA-only, following ModelBase file layout)
    # ------------------------------------------------------------------
    def _save_lora(self, save_dir, network, label, iter_label, extra=()):
        module = self.get_bare_model(network)
        sd = {k: v.cpu() for k, v in module.state_dict().items()
              if ("lora" in k) or any(e in k for e in extra)}
        torch.save(sd, os.path.join(save_dir, '{}_{}.h5'.format(iter_label, label)))

    def _load_lora(self, eid, network, label):
        path = self._find_latest_checkpoint(eid, "saved_models", "*_{}.h5".format(label))
        if path is None:
            print("No {} checkpoint found, skipping loading...".format(label))
            return
        if self.opt['rank'] == 0:
            print("Loading {} [{}] ...".format(label, self._short_path(path)))
        sd = torch.load(path, weights_only=True)
        self.get_bare_model(network).load_state_dict(sd, strict=False)
        if label == 'G':
            self.last_iteration = int(os.path.basename(path).split('_')[0])

    def load(self, experiment_id=None, mode='train'):
        eid = self._resolve_eid(experiment_id)
        if mode == 'train':
            if self.opt['train_mode'] == 'scratch':
                return
            assert eid is not None, f"Pretrained experiment ID required for train_mode='{self.opt['train_mode']}'."
        else:
            assert eid is not None, "Experiment ID required for test mode."
        self._load_lora(eid, self.netG, 'G')
        if mode == 'train':
            self._load_lora(eid, self.netReg, 'Reg')

    def save(self, iter_label):
        self._save_lora(self._run_dir("saved_models"), self.netG, 'G', iter_label, extra=('conv_in',))
        self.save_optimizer(self._run_dir("saved_optimizers"), self.G_optimizer, 'optimizerG', iter_label)
        self.save_scheduler(self._run_dir("saved_schedulers"), self.schedulers[0], 'schedulerG', iter_label)
        if self.mixed_precision is not None:
            self.save_gradscaler(self._run_dir("saved_gradscalers"), self.gen_scaler, 'gradscalerG', iter_label)

        self._save_lora(self._run_dir("saved_models"), self.netReg, 'Reg', iter_label)
        self.save_optimizer(self._run_dir("saved_optimizers"), self.reg_optimizer, 'optimizerReg', iter_label)
        self.save_scheduler(self._run_dir("saved_schedulers"), self.schedulers[1], 'schedulerReg', iter_label)
        if self.mixed_precision is not None:
            self.save_gradscaler(self._run_dir("saved_gradscalers"), self.reg_scaler, 'gradscalerReg', iter_label)

    def load_optimizers(self, experiment_id=None):
        if self.opt['train_mode'] != 'resume':
            return
        eid = self._resolve_eid(experiment_id)
        assert eid is not None, "Pretrained experiment ID required for train_mode='resume'."
        self.opt['train_opt']['G_optimizer_reuse'] = True
        self.load_G_optimizer(eid)
        path = self._find_latest_checkpoint(eid, "saved_optimizers", "*optimizerReg.h5")
        if path is not None:
            self.load_optimizer(path, self.reg_optimizer)

    def load_schedulers(self, experiment_id=None):
        if self.opt['train_mode'] != 'resume':
            return
        eid = self._resolve_eid(experiment_id)
        assert eid is not None, "Pretrained experiment ID required for train_mode='resume'."
        self.load_G_scheduler(eid)
        path = self._find_latest_checkpoint(eid, "saved_schedulers", "*schedulerReg.h5")
        if path is not None:
            self.load_scheduler(path, self.schedulers[1])

    def load_gradscalers(self, experiment_id=None):
        if self.opt['train_mode'] != 'resume':
            return
        eid = self._resolve_eid(experiment_id)
        assert eid is not None, "Pretrained experiment ID required for train_mode='resume'."
        self.load_G_gradscaler(eid)
        path = self._find_latest_checkpoint(eid, "saved_gradscalers", "*gradscalerReg.h5")
        if path is not None:
            self.load_gradscaler(path, self.reg_scaler)

    # ------------------------------------------------------------------
    # optimization
    # ------------------------------------------------------------------
    def _clip(self, params, scaler=None, optimizer=None):
        clip_max = self.opt_train['G_optimizer_clipgrad']
        if clip_max <= 0:
            return torch.zeros(1)
        if scaler is not None:
            scaler.unscale_(optimizer)
        return torch.nn.utils.clip_grad_norm_(params, max_norm=clip_max, norm_type=2)

    def optimize_parameters_amp(self, current_step, update=False):
        # ---- generator: recon (MSE+LPIPS) + lambda_vsd * VSD ----
        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            out, latent, prompt_embeds, neg_prompt_embeds = self._gen_forward()
            recon = compute_generator_loss(self.H, self.E, self.loss_fn_dict, self.loss_val_dict, device=self.device)
            loss_vsd = self.netReg.distribution_matching_loss(latent, prompt_embeds, neg_prompt_embeds, self.osediff_args)
            gen_loss = (recon + self.lambda_vsd * loss_vsd) / self.num_accum_steps_G
        self.gen_scaler.scale(gen_loss).backward()
        self.G_train_loss = gen_loss

        # ---- regularizer: lambda_vsd_lora * diff_loss(generated latent) ----
        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            loss_d = self.netReg.diff_loss(latent, prompt_embeds, self.osediff_args)
            reg_loss = (self.lambda_vsd_lora * loss_d) / self.num_accum_steps_G
        self.reg_scaler.scale(reg_loss).backward()
        self.Reg_train_loss = reg_loss

        if self.opt['rank'] == 0:
            print("G train loss:", self.G_train_loss.item(), "| Reg train loss:", self.Reg_train_loss.item())

        self.update = ((self.G_accum_count + 1) % self.num_accum_steps_G) == 0 or update
        if self.update:
            self.G_train_grad_norm = self._clip(self.g_params, self.gen_scaler, self.G_optimizer)
            self.gen_scaler.step(self.G_optimizer)
            self.gen_scaler.update()
            self.G_optimizer.zero_grad()

            self.Reg_train_grad_norm = self._clip(self.reg_params, self.reg_scaler, self.reg_optimizer)
            self.reg_scaler.step(self.reg_optimizer)
            self.reg_scaler.update()
            self.reg_optimizer.zero_grad()
            self.G_accum_count = 0
        else:
            self.G_accum_count += 1

    def optimize_parameters(self, current_step, update=False):
        # ---- generator ----
        out, latent, prompt_embeds, neg_prompt_embeds = self._gen_forward()
        recon = compute_generator_loss(self.H, self.E, self.loss_fn_dict, self.loss_val_dict, device=self.device)
        loss_vsd = self.netReg.distribution_matching_loss(latent, prompt_embeds, neg_prompt_embeds, self.osediff_args)
        gen_loss = (recon + self.lambda_vsd * loss_vsd) / self.num_accum_steps_G
        gen_loss.backward()
        self.G_train_loss = gen_loss

        # ---- regularizer ----
        loss_d = self.netReg.diff_loss(latent, prompt_embeds, self.osediff_args)
        reg_loss = (self.lambda_vsd_lora * loss_d) / self.num_accum_steps_G
        reg_loss.backward()
        self.Reg_train_loss = reg_loss

        if self.opt['rank'] == 0:
            print("G train loss:", self.G_train_loss.item(), "| Reg train loss:", self.Reg_train_loss.item())

        self.update = ((self.G_accum_count + 1) % self.num_accum_steps_G) == 0 or update
        if self.update:
            self.G_train_grad_norm = self._clip(self.g_params)
            self.G_optimizer.step()
            self.G_optimizer.zero_grad()

            self.Reg_train_grad_norm = self._clip(self.reg_params)
            self.reg_optimizer.step()
            self.reg_optimizer.zero_grad()
            self.G_accum_count = 0
        else:
            self.G_accum_count += 1

    # ------------------------------------------------------------------
    # logging / test / validation
    # ------------------------------------------------------------------
    def record_train_log(self, current_step):
        self.run.log({"step": current_step, "G_train_loss": self.G_train_loss.item() * self.num_accum_steps_G})
        self.run.log({"step": current_step, "Reg_train_loss": self.Reg_train_loss.item() * self.num_accum_steps_G})
        self.run.log({"step": current_step, "G_train_grad_norm": self.G_train_grad_norm.item()})
        self.run.log({"step": current_step, "Reg_train_grad_norm": self.Reg_train_grad_norm.item()})

    def record_avg_train_log(self, current_step, idx_train):
        self.run.log({"step": current_step, "G_train_loss": (self.G_train_loss.item() / idx_train) * self.num_accum_steps_G})
        self.run.log({"step": current_step, "Reg_train_loss": (self.Reg_train_loss.item() / idx_train) * self.num_accum_steps_G})
        self.G_train_loss = 0.0
        self.Reg_train_loss = 0.0

    def test(self):
        self.netG.eval()
        with torch.inference_mode():
            self.netG_forward()
        self.netG.train()

    def validation(self):
        self.netG_forward()
        self.gen_loss = compute_generator_loss(self.H, self.E, self.loss_fn_dict, self.loss_val_dict, device=self.device)
        self.G_valid_loss += self.gen_loss
        compute_performance_metrics(self.E, self.H, self.metric_fn_dict, self.metric_val_dict, rescale_images=True)

    def validation_amp(self):
        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            self.netG_forward()
            self.gen_loss = compute_generator_loss(self.H, self.E, self.loss_fn_dict, self.loss_val_dict, device=self.device)
        self.G_valid_loss += self.gen_loss
        compute_performance_metrics(self.E, self.H, self.metric_fn_dict, self.metric_val_dict, rescale_images=True)
