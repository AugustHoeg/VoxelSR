import math
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torchvision.utils import make_grid

from models.model_base import ModelBase
from models.select_model import define_Model
from models.select_network import define_G
from performance_metrics.performance_metrics import compute_performance_metrics
from utils import utils_3D_image
from utils.load_options import load_options_from_experiment_id


class ModelMaskRQVSRT(ModelBase):
    """
    Masked Residual Quantized Volumetric Super-Resolution Transformer (MaskRQVSRT)

    Trains MaskRQTransformer3Dv2 with MaskGIT-style masking extended to both
    spatial positions (L) and RQ depths (D), yielding a 2D (L×D) token grid.

    At each training step a random fraction of the L×D (position, depth) pairs
    is independently masked and the model learns to reconstruct them from the
    unmasked context and LR conditioning.  Inference uses iterative confidence-
    based unmasking over the same 2D grid, starting from a fully masked code.
    """

    def __init__(self, opt, mode='train', data_parallel=True):
        super().__init__(opt)
        self.last_iteration = 0

        self.netG = define_G(opt, mode=mode)
        self.netG = self.model_to_device(self.netG, data_parallel=data_parallel)

        self.num_embeddings = opt['model_opt']['netG']['num_embeddings']
        self.n_rq_depth = opt['model_opt']['netG']['n_rq_depth']
        self.mask_schedule = opt['model_opt']['netG']['mask_schedule']
        # Controls how much extra masking deeper RQ depths receive (0 = uniform, 1 = depth D-1 always fully masked)
        self.depth_mask_scale = opt['model_opt']['netG'].get('depth_mask_scale', 0.25)

        self.mask_token_id = self.num_embeddings  # reuses the +1 tok_emb slot (never a prediction target)

        self.vq_model_hr = None
        self.vq_model_lr = None
        self.latent_shape_hr = None

        self.update = False
        self.early_stop = False
        self.min_validation_loss = float('inf')
        self.patience = self.opt_train['early_stop_patience']
        self.patience_counter = 0
        self.min_delta = 0

        if opt['rank'] == 0 and mode == 'train':
            print("Number of trainable parameters, G", utils_3D_image.numel(self.netG, only_trainable=True))

    # ----------------------------------------
    # VQ model loading
    # ----------------------------------------

    def _load_vq_model(self, eid):
        opt_path = load_options_from_experiment_id(eid, root_dir="", file_type="yaml")
        opt_vq = OmegaConf.load(opt_path)
        opt_vq['dist'] = False  # Disable DDP on VQ
        opt_vq['compile'] = False  # Disable overarching compile on VQ

        net = define_Model(opt_vq, mode='test', data_parallel=False)
        net.load(eid, mode='test')
        vq_model = net.get_bare_model(net.netG).to(self.device)
        vq_model.eval()
        for p in vq_model.parameters():
            p.requires_grad_(False)
        return vq_model

    def load_hr_vq_model(self):
        assert "pretrained_hr_vqmodel_id" in self.opt["path"], (
            "Must specify pretrained_hr_vqmodel_id in path for ModelMaskRQVSRT."
        )
        eid = self.opt["path"]["pretrained_hr_vqmodel_id"]
        self.vq_model_hr = self._load_vq_model(eid)
        self.vq_model_hr.encode = torch.compile(self.vq_model_hr.encode, mode="max-autotune-no-cudagraphs")
        self.vq_model_hr.decode_code = torch.compile(self.vq_model_hr.decode_code, mode="max-autotune-no-cudagraphs")

        assert self.num_embeddings == self.vq_model_hr.quantizer.codebooks[0].n_embed, (
            f"num_embeddings mismatch: transformer has {self.num_embeddings}, "
            f"HR VQ model has {self.vq_model_hr.quantizer.codebooks[0].n_embed}."
        )

    def load_lr_vq_model(self):
        assert "pretrained_lr_vqmodel_id" in self.opt["path"], (
            "Must specify pretrained_lr_vqmodel_id in path for ModelMaskRQVSRT."
        )
        eid = self.opt["path"]["pretrained_lr_vqmodel_id"]
        self.vq_model_lr = self._load_vq_model(eid)

    # ----------------------------------------
    # Encoding / decoding (VQ models always frozen)
    # ----------------------------------------

    @torch.no_grad()
    def encode_to_indices(self, x: torch.Tensor, vq_model: torch.nn.Module):
        """Encode a volume to RQ codes via the frozen VQ encoder.

        Args:
            x:        (B, C, D, H, W)
            vq_model: frozen RQVAE3D model
        Returns:
            codes:        (B, Dz, Dy, Dx, n_rq_depth) int64
            z_e:          (B, C, Dz, Dy, Dx) continuous encoder features
            latent_shape: (Dz, Dy, Dx)
        """
        z_e = vq_model.encode(x)
        latent_shape = tuple(z_e.shape[2:])
        _, _, codes, _, _ = vq_model.quantizer(z_e)   # codes: (B, Dz, Dy, Dx, D)
        return codes, z_e, latent_shape

    def _flatten_lr_embeddings(self, z_lr: torch.Tensor) -> torch.Tensor:
        """Reshape LR encoder output (B, C, Dz, Dy, Dx) → (B, N_lr, C)."""
        B, C = z_lr.shape[:2]
        return z_lr.view(B, C, -1).permute(0, 2, 1)

    # ----------------------------------------
    # Masking utilities
    # ----------------------------------------

    def _get_mask_ratios(self, r: torch.Tensor) -> torch.Tensor:
        if self.mask_schedule == "linear":
            return 1 - r
        elif self.mask_schedule == "square":
            return 1 - (r ** 2)
        elif self.mask_schedule == "cosine":
            return torch.cos(r * math.pi * 0.5)
        else:  # "arccos" — MaskGIT default, biased toward high masking ratios
            return torch.arccos(r) / (math.pi * 0.5)

    def _mask_tokens_rq(self, codes: torch.Tensor):
        """Apply 2D MaskGIT masking over the (L, D) token grid.

        Each (position, depth) pair is independently masked with a per-sample
        probability drawn from the configured mask schedule.  This lets the
        model learn to predict any depth at any position from the remaining
        unmasked context.

        Args:
            codes: (B, Dz, Dy, Dx, D) int64
        Returns:
            masked_codes: (B, Dz, Dy, Dx, D) — masked positions hold mask_token_id
            mask:         (B, L, D) bool       — True at masked positions
        """
        B, dz, dy, dx, D = codes.shape
        L = dz * dy * dx

        r = torch.rand(B, device=codes.device)
        mask_ratio = self._get_mask_ratios(r)                           # (B,)

        noise = torch.rand(B, L, D, device=codes.device)
        mask  = noise < mask_ratio[:, None, None]                       # (B, L, D)

        codes_flat = codes.reshape(B, L, D).clone()
        codes_flat[mask] = self.mask_token_id

        return codes_flat.reshape(B, dz, dy, dx, D), mask

    def _mask_tokens_coarse_to_fine(self, codes: torch.Tensor):
        """Depth-biased 2D MaskGIT masking that promotes coarse-to-fine generation.

        A base mask ratio is drawn per sample from the configured schedule.
        Depth d then receives a boosted per-depth ratio:

            rho_d = base + depth_frac[d] * depth_mask_scale * (1 - base)

        where depth_frac[d] = d / (D-1) ramps linearly from 0 (coarsest) to 1
        (finest).  Spatial tokens at each depth are then masked independently
        with Bernoulli(rho_d).

        When depth_mask_scale=0 this reduces to the original uniform masking
        from _mask_tokens_rq.  When depth_mask_scale=1 the finest depth (D-1)
        is always fully masked regardless of the base ratio.

        Args:
            codes: (B, Dz, Dy, Dx, D) int64
        Returns:
            masked_codes: (B, Dz, Dy, Dx, D) — masked positions hold mask_token_id
            mask:         (B, L, D) bool       — True at masked positions
        """
        B, dz, dy, dx, D = codes.shape
        L = dz * dy * dx

        r = torch.rand(B, device=codes.device)
        base = self._get_mask_ratios(r)                              # (B,)

        # depth_frac: 0 at d=0 (coarsest), 1 at d=D-1 (finest)
        depth_frac = torch.linspace(0, 1, D, device=codes.device)   # (D,)

        # rho: (B, D) — deeper depths get higher mask ratios
        rho = base[:, None] + self.depth_mask_scale * depth_frac[None, :] * (1 - base[:, None])
        rho = rho.clamp(0, 1)

        noise = torch.rand(B, L, D, device=codes.device)
        mask = noise < rho[:, None, :]                               # (B, L, D)

        codes_flat = codes.reshape(B, L, D).clone()
        codes_flat[mask] = self.mask_token_id

        return codes_flat.reshape(B, dz, dy, dx, D), mask

    def _make_unmask_schedule(self, n_tokens: int, n_steps: int) -> torch.Tensor:
        """Compute per-step cumulative unmasking counts for iterative decoding.

        Args:
            n_tokens: total number of tokens (L*D for the 2D grid)
            n_steps:  number of refinement iterations
        Returns:
            (n_steps,) int tensor — cumulative number of tokens to unmask by step i
        """
        r = torch.linspace(1 / n_steps, 1, n_steps)
        mask_ratio = 1 - self._get_mask_ratios(r)
        sche = (mask_ratio * n_tokens).int()
        return sche

    # ----------------------------------------
    # Loss helper
    # ----------------------------------------

    def _compute_rq_loss(self, logits, codes_flat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Cross-entropy loss averaged over all masked (position, depth) pairs.

        Args:
            logits:     list of D tensors, each (B, L, num_embeddings+1)
            codes_flat: (B, L, D) int64 ground-truth codes
            mask:       (B, L, D) bool   — True at positions to include in loss
        Returns:
            scalar loss
        """
        # Stack per-depth logits → (B, L, D, num_embeddings+1)
        logits_stack = torch.stack(logits, dim=2)
        B, L, D, V = logits_stack.shape

        logits_flat  = logits_stack.reshape(B * L * D, V)
        targets_flat = codes_flat.reshape(B * L * D)
        mask_flat    = mask.reshape(B * L * D)

        return F.cross_entropy(logits_flat[mask_flat], targets_flat[mask_flat])

    # ----------------------------------------
    # Inference: iterative masked decoding over the 2D (L, D) grid
    # ----------------------------------------

    @torch.inference_mode()
    def sample_E(self, z_lr: torch.Tensor, n_steps: int = 12,
                 temperature: float = 1.0) -> torch.Tensor:
        """Generate volumes via 2D iterative masked token prediction.

        At each of n_steps iterations the transformer predicts all (L×D) tokens
        and the t most confident currently-masked tokens are revealed.

        Args:
            z_lr:        (B, N_lr, C) flattened LR encoder embeddings
            n_steps:     refinement iterations
            temperature: softmax temperature (lower = sharper predictions)
        Returns:
            x_sampled: (B, C, D, H, W) reconstructed HR volume
        """
        assert self.latent_shape_hr is not None, "latent_shape_hr not set; call encode_to_indices first."
        dz, dy, dx = self.latent_shape_hr
        L = dz * dy * dx
        D = self.n_rq_depth
        B = z_lr.shape[0]

        # Start fully masked
        codes = torch.full((B, dz, dy, dx, D), self.mask_token_id, dtype=torch.long, device=self.device)
        mask = torch.ones(B, L, D, dtype=torch.bool, device=self.device)  # (B, L, D)

        schedule = self._make_unmask_schedule(L * D, n_steps)

        for step, t in enumerate(schedule):
            print(f"Sampling step {step + 1}/{n_steps}")
            if not mask.any():
                break

            t = max(t.item(), step + 1)  # reveal at least one new token per step

            logits = self.netG(codes, z_lr)          # list[D × (B, L, V)]

            # Stack → (B, L, D, V), compute per-token sampling distribution
            logits_stack = torch.stack(logits, dim=2)                           # (B, L, D, V)
            prob = torch.softmax(logits_stack / temperature, dim=-1)       # (B, L, D, V)
            pred_code = torch.distributions.Categorical(probs=prob).sample()    # (B, L, D)

            # Confidence = probability assigned to the sampled token
            conf = prob.gather(-1, pred_code.unsqueeze(-1)).squeeze(-1)         # (B, L, D)
            conf[~mask] = math.inf  # force retention of already-revealed tokens

            # Select the t highest-confidence masked (l, d) pairs
            conf_flat = conf.reshape(B, L * D)
            thresh, _ = torch.topk(conf_flat, k=int(t), dim=-1)
            thresh = thresh[:, [-1]]                                         # (B, 1) per-sample cutoff

            reveal = (conf_flat >= thresh).reshape(B, L, D)

            # Update codes and recompute mask
            codes_flat = codes.reshape(B, L, D)
            codes_flat[reveal] = pred_code[reveal]
            codes = codes_flat.reshape(B, dz, dy, dx, D)
            mask = (codes.reshape(B, L, D) == self.mask_token_id)

        codes = codes.clamp(0, self.num_embeddings - 1)
        return self.vq_model_hr.decode_code(codes)

    @torch.inference_mode()
    def sample_E_coarse_to_fine(self, z_lr: torch.Tensor, n_steps: int = 12,
                                 temperature: float = 1.0) -> torch.Tensor:
        """Generate volumes with explicit depth-ordered iterative decoding.

        n_steps are divided evenly across D depths.  Within each depth's
        allocation, confidence-based unmasking is restricted to spatial tokens
        at that depth — coarser depths are fully committed before finer depths
        begin, matching the training distribution of _mask_tokens_coarse_to_fine.

        Args:
            z_lr:        (B, N_lr, C) flattened LR encoder embeddings
            n_steps:     total refinement iterations (split evenly across D depths)
            temperature: softmax temperature
        Returns:
            x_sampled: (B, C, D, H, W) reconstructed HR volume
        """
        assert self.latent_shape_hr is not None, "latent_shape_hr not set; call encode_to_indices first."
        dz, dy, dx = self.latent_shape_hr
        L = dz * dy * dx
        D = self.n_rq_depth
        B = z_lr.shape[0]

        codes = torch.full((B, dz, dy, dx, D), self.mask_token_id, dtype=torch.long, device=self.device)
        mask = torch.ones(B, L, D, dtype=torch.bool, device=self.device)

        steps_per_depth = max(1, n_steps // D)

        for d in range(D):
            depth_schedule = self._make_unmask_schedule(L, steps_per_depth)
            for step, t in enumerate(depth_schedule):
                if not mask[:, :, d].any():
                    break

                t = max(t.item(), step + 1)

                logits = self.netG(codes, z_lr)                                       # list[D × (B, L, V)]
                logits_stack = torch.stack(logits, dim=2)                             # (B, L, D, V)
                prob = torch.softmax(logits_stack / temperature, dim=-1)
                pred_code = torch.distributions.Categorical(probs=prob).sample()      # (B, L, D)

                # Confidence over spatial positions at depth d only
                conf_d = prob[:, :, d, :].gather(-1, pred_code[:, :, d].unsqueeze(-1)).squeeze(-1)  # (B, L)
                conf_d[~mask[:, :, d]] = math.inf

                thresh, _ = torch.topk(conf_d, k=int(t), dim=-1)
                thresh = thresh[:, [-1]]
                reveal_d = conf_d >= thresh                                            # (B, L)

                codes_flat = codes.reshape(B, L, D)
                codes_flat[reveal_d, d] = pred_code[reveal_d, d]
                codes = codes_flat.reshape(B, dz, dy, dx, D)
                mask[:, :, d] = codes_flat[:, :, d] == self.mask_token_id

            print(f"Depth {d}/{D - 1} committed.")

        codes = codes.clamp(0, self.num_embeddings - 1)
        return self.vq_model_hr.decode_code(codes)

    # ----------------------------------------
    # Lifecycle
    # ----------------------------------------

    def init_train(self):
        self.load()
        self.load_hr_vq_model()
        self.load_lr_vq_model()
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

    def init_test(self, experiment_id):
        self.load(experiment_id, mode='test')
        self.load_hr_vq_model()
        self.load_lr_vq_model()
        self.netG.eval()
        self.define_metrics()
        self.define_mixed_precision()
        self.define_visual_eval()

    def set_eval_mode(self):
        self.netG.eval()

    def set_train_mode(self):
        self.netG.train()

    # ----------------------------------------
    # Loss / WandB
    # ----------------------------------------

    def define_loss(self):
        self.init_G_loss_trackers()

    def define_wandb_run(self):
        self._init_wandb_run()
        self.model_artifact_G = wandb.Artifact(
            "Generator", type=self.opt['model_opt']['netG']['net_type'],
            description=self.opt['model_opt']['netG']['description'],
            metadata=OmegaConf.to_container(self.opt['model_opt']['netG'], resolve=True)
        )

    def define_visual_eval(self):
        pass

    # ----------------------------------------
    # Training
    # ----------------------------------------

    def feed_data(self, data):
        self.H = data['H'].as_tensor().to(self.device, non_blocking=True)
        self.L = data['L'].as_tensor().to(self.device, non_blocking=True)

    def optimize_parameters_amp(self, current_step, update=False):
        codes, z_hr, self.latent_shape_hr = self.encode_to_indices(self.H, self.vq_model_hr)
        _, z_lr, _ = self.encode_to_indices(self.L, self.vq_model_lr)
        z_lr = self._flatten_lr_embeddings(z_lr)       # (B, N_lr, C)

        masked_codes, mask = self._mask_tokens_coarse_to_fine(codes)   # (B,dz,dy,dx,D), (B,L,D)
        codes_flat = codes.reshape(codes.shape[0], -1, self.n_rq_depth)  # (B, L, D)

        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            logits = self.netG(masked_codes, z_lr)     # list[D × (B, L, V)]
            self.gen_loss = (
                self._compute_rq_loss(logits, codes_flat, mask)
                / self.num_accum_steps_G
            )

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
        codes, z_hr, self.latent_shape_hr = self.encode_to_indices(self.H, self.vq_model_hr)
        _, z_lr, _ = self.encode_to_indices(self.L, self.vq_model_lr)
        z_lr = self._flatten_lr_embeddings(z_lr)       # (B, N_lr, C)

        masked_codes, mask = self._mask_tokens_coarse_to_fine(codes)   # (B,dz,dy,dx,D), (B,L,D)
        codes_flat = codes.reshape(codes.shape[0], -1, self.n_rq_depth)  # (B, L, D)

        logits = self.netG(masked_codes, z_lr)         # list[D × (B, L, V)]
        self.gen_loss = (
            self._compute_rq_loss(logits, codes_flat, mask)
            / self.num_accum_steps_G
        )

        self.G_train_loss = self.gen_loss
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

    # ----------------------------------------
    # Validation
    # ----------------------------------------

    def validation(self):
        codes, _, self.latent_shape_hr = self.encode_to_indices(self.H, self.vq_model_hr)
        _, z_lr, _ = self.encode_to_indices(self.L, self.vq_model_lr)
        z_lr = self._flatten_lr_embeddings(z_lr)

        masked_codes, mask = self._mask_tokens_coarse_to_fine(codes)
        codes_flat = codes.reshape(codes.shape[0], -1, self.n_rq_depth)

        logits = self.netG(masked_codes, z_lr)
        self.gen_loss = self._compute_rq_loss(logits, codes_flat, mask)
        self.G_valid_loss += self.gen_loss

    def validation_amp(self):
        codes, _, self.latent_shape_hr = self.encode_to_indices(self.H, self.vq_model_hr)
        _, z_lr, _ = self.encode_to_indices(self.L, self.vq_model_lr)
        z_lr = self._flatten_lr_embeddings(z_lr)

        masked_codes, mask = self._mask_tokens_coarse_to_fine(codes)
        codes_flat = codes.reshape(codes.shape[0], -1, self.n_rq_depth)

        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            logits = self.netG(masked_codes, z_lr)
            self.gen_loss = self._compute_rq_loss(logits, codes_flat, mask)
        self.G_valid_loss += self.gen_loss

    # ----------------------------------------
    # Logging / visuals
    # ----------------------------------------

    def record_train_log(self, current_step):
        loss = self.G_train_loss.item() * self.num_accum_steps_G
        self.run.log({"step": current_step, "G_train_loss": loss})
        grad_norm = self.G_train_grad_norm.item() * self.num_accum_steps_G
        self.run.log({"step": current_step, "G_train_grad_norm": grad_norm})

    def current_visuals(self):
        out_dict = OrderedDict()

        codes, _, self.latent_shape_hr = self.encode_to_indices(self.H, self.vq_model_hr)
        _, z_lr, _ = self.encode_to_indices(self.L, self.vq_model_lr)
        z_lr = self._flatten_lr_embeddings(z_lr)

        E_vq = self.vq_model_hr.decode_code(codes)
        E = self.sample_E(z_lr)

        out_dict['H'] = self.H.detach()[0].float().cpu()
        out_dict['E_vq'] = E_vq.detach()[0].float().cpu()
        out_dict['E'] = E.detach()[0].float().cpu()

        return out_dict

    def log_comparison_image(self, img_dict, current_step, out_dtype=np.uint8):
        slice_idx  = img_dict['H'].shape[-1] // 2
        H_slice = img_dict['H'][:, :, :, slice_idx]
        E_vq_slice = img_dict['E_vq'][:, :, :, slice_idx]
        E_slice = img_dict['E'][:, :, :, slice_idx]

        row = torch.stack([E_slice, E_vq_slice, H_slice])
        grid = make_grid(row, nrow=len(row), padding=0).permute(1, 2, 0)
        grid_image = utils_3D_image.unnorm_and_rescale(grid, out_dtype)

        figure_string = "MaskRQVSRT: %s, step %d: E, VQ Recon, HR" % (
            self.opt["model_opt"]["model_architecture"], current_step,
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
