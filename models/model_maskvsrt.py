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
from utils import utils_3D_image
from utils.load_options import load_options_from_experiment_id


class ModelMaskVSRT(ModelBase):
    """
    Trains masked volumetric super-resolution transformer (MaskVSRT).
    
    Uses separate encoder for HR and LR images with separate codebooks.
    Transformer trains to predict masked HR codebook indices conditioned
    on LR embedding vectors (not codebook indices). Inference uses MaskGIT-style
    iterative masked decoding to generate HR codebook indices, which are then 
    decoded to a volume via the frozen HR VQ decoder.
    """

    def __init__(self, opt, mode='train', data_parallel=True):
        super().__init__(opt)
        self.last_iteration = 0

        self.netG = define_G(opt, mode=mode)
        self.netG = self.model_to_device(self.netG, data_parallel=data_parallel, compile=False)

        self.num_embeddings = opt['model_opt']['netG']['num_embeddings']
        self.mask_schedule = opt['model_opt']['netG']['mask_schedule']

        self.mask_token_id = self.num_embeddings  # reuses the +1 tok_emb slot (never a prediction target)

        self.vq_model = None
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
        opt_vq['dist'] = False

        net = define_Model(opt_vq, mode='test', data_parallel=False)
        net.load(eid, mode='test')
        vq_model = net.get_bare_model(net.netG).to(self.device)
        vq_model.eval()
        for p in vq_model.parameters():
            p.requires_grad_(False)

        return vq_model
        
        
    def load_hr_vq_model(self):
        assert "pretrained_hr_vqmodel_id" in self.opt["path"], (
            "Must specify pretrained_hr_vqmodel_id in path for ModelMaskVSRT."
        )
        eid = self.opt["path"]["pretrained_hr_vqmodel_id"]
        self.vq_model_hr = self._load_vq_model(eid)

        assert self.num_embeddings == self.vq_model_hr.codebook.num_embeddings, (
            f"num_embeddings mismatch: transformer has {self.num_embeddings}, "
            f"HR VQ model has {self.vq_model_hr.codebook.num_embeddings}."
        )
        
    
    def load_lr_vq_model(self):
        assert "pretrained_lr_vqmodel_id" in self.opt["path"], (
            "Must specify pretrained_lr_vqmodel_id in path for ModelMaskVSRT."
        )
        eid = self.opt["path"]["pretrained_lr_vqmodel_id"]
        self.vq_model_lr = self._load_vq_model(eid)
        

    # ----------------------------------------
    # Encoding / decoding (VQ model always frozen)
    # ----------------------------------------

    @torch.no_grad()
    def encode_to_indices(self, x: torch.Tensor, vq_model: torch.nn.Module):
        """Encode a volume to flat codebook indices via the frozen VQ encoder.

        Args:
            x: (B, C, D, H, W)
            vq_model: Either LR or HR VQ model.
        Returns:
            q_indices:    codebook indices: (B, L) int64  where L = D'*H'*W' 
            z_e:          latent embedding vectors: (B, D', H', W')
            latent_shape: (D', H', W')
        """
        z_e = vq_model.encoder(x)
        latent_shape = tuple(z_e.shape[2:])
        _, _, q_indices, _ = vq_model.codebook(z_e)
        return q_indices, z_e, latent_shape

    @torch.no_grad()
    def decode_indices(self, q_indices: torch.Tensor, vq_model: torch.nn.Module, latent_shape: tuple) -> torch.Tensor:
        """Decode flat codebook indices to a volume via the frozen VQ decoder.

        Args:
            q_indices:    (B, L) int64
            vq_model: Either LR or HR VQ model.
            latent_shape: (D', H', W')
        Returns:
            x_hat: (B, C, D, H, W)
        """
        B = q_indices.shape[0]
        embed_dim = vq_model.codebook.embedding_dim
        z_q = vq_model.codebook.embedding(q_indices)          # (B, L, C)
        z_q = z_q.permute(0, 2, 1).view(B, embed_dim, *latent_shape)
        return vq_model.decode(z_q)

    # ----------------------------------------
    # Masking utilities
    # ----------------------------------------

    def _get_mask_ratios(self, r: torch.Tensor) -> torch.Tensor:

        if self.mask_schedule == "linear":
            mask_ratio = 1 - r
        elif self.mask_schedule == "square":
            mask_ratio = 1 - (r ** 2)
        elif self.mask_schedule == "cosine":
            mask_ratio = torch.cos(r * math.pi * 0.5)
        else:  # "arccos" - MaskGIT default, biased toward high masking ratios
            mask_ratio = torch.arccos(r) / (math.pi * 0.5)

        return mask_ratio


    def _mask_tokens(self, q_indices: torch.Tensor):
        """Apply MaskGIT masking to a batch of token sequences.

        A per-patch masking ratio is drawn from the configured schedule distribution,
        then a random subset of the patch codes is replaced with mask_value.

        Args:
            q_indices: (B, L) int64 with values in [0, vq_vocab - 1]
        Returns:
            masked_indices: (B, L) with masked positions set to mask_value
            mask:           (B, L) bool, True at masked positions
        """
        B, L = q_indices.shape
        r = torch.rand(B, device=q_indices.device)

        mask_ratio = self._get_mask_ratios(r)  # (0, 1) fraction of tokens to mask for each sample in the batch

        noise = torch.rand(B, L, device=q_indices.device)  # Sample noise
        mask = noise < mask_ratio.unsqueeze(1)  # Create mask bool of shape (B, L) by value according to schedule

        masked_indices = q_indices.clone()  # (B, L)
        masked_indices[mask] = self.mask_token_id

        return masked_indices, mask

    def _make_unmask_schedule(self, n_tokens: int, n_steps: int) -> torch.Tensor:
        """Compute per-step unmasking counts following an arccos schedule.

        Fewer tokens are revealed in early steps, more in later steps.

        Args:
            n_tokens: total sequence length L
            n_steps:  number of refinement iterations
        Returns:
            list of ints, length n_steps, with total number of tokens to unmask at each step
        """

        r = torch.linspace(1/n_steps, 1, n_steps)
        mask_ratio = 1 - self._get_mask_ratios(r)
        sche = mask_ratio * n_tokens
        sche = sche.int()
        return sche

    # ----------------------------------------
    # Inference: iterative masked decoding
    # ----------------------------------------

    @torch.no_grad()
    def sample_from_transformer(self, n_samples: int = 1, n_steps: int = 12,
                                 temperature: float = 1.0) -> torch.Tensor:
        """Generate volumes via iterative masked token prediction (MaskGIT inference).

        Starting from a fully masked sequence, at each of n_steps iterations the
        transformer predicts all positions and the t most confident masked tokens are revealed.
        NOTE: Positions of unmasked tokens are retained from iteration to iteration, however,
        the predicted values are still be updated. This is different from Halton-MaskGIT,
        see issue: https://github.com/valeoai/Halton-MaskGIT/issues/27#issuecomment-2890779342.
        According to authors, the model learns to simply copy-paste already unmasked values.

        Args:
            n_samples:   number of volumes to generate
            n_steps:     refinement iterations (12 is the MaskGIT paper default)
            temperature: softmax temperature (lower = sharper, higher = more stochastic)
        Returns:
            x_sampled: (n_samples, C, D, H, W)
        """
        assert self.latent_shape_hr is not None, "latent_shape not set; call encode_to_indices first."
        L = int(np.prod(self.latent_shape_hr))

        # Start fully masked
        code = torch.full((n_samples, L), self.mask_token_id, dtype=torch.long, device=self.device)
        mask = torch.ones(n_samples, L, dtype=torch.bool, device=self.device)

        schedule = self._make_unmask_schedule(L, n_steps)

        for index, t in enumerate(schedule):  # Beginning of sampling, t = total of tokens predicted a step "index"
            print(f"Sampling step {index + 1}/{n_steps}")
            if not mask.any():  # Break if mask is empty (all tokens revealed)
                break

            t = max(t, index + 1)  # make sure to predict at least 1 token at each step

            logits = self.netG(code)  # (n_samples, L, num_embeddings)
            prob = torch.softmax(logits / temperature, dim=-1)
            pred_code = torch.distributions.Categorical(probs=prob).sample()   # (n_samples, L)

            # Confidence = probability assigned to the sampled token
            conf = prob.gather(-1, pred_code.unsqueeze(-1)).squeeze(-1)  # (n_samples, L)
            conf[~mask] = math.inf  # Force retention of already-revealed token positions

            # Select the t highest-confidence masked positions via a threshold
            tresh_conf, index_mask = torch.topk(conf, k=int(t), dim=-1)
            tresh_conf = tresh_conf[:, [-1]]  # (n_samples, 1) cutoff value per sample

            # Update code and recompute mask from code
            f_mask = (conf >= tresh_conf)  # (n_samples, L)
            code[f_mask] = pred_code[f_mask]  # Already revealed tokens may still be updated
            mask = (code == self.mask_token_id)

        code = code.clamp(0, self.num_embeddings - 1)
        return self.decode_indices(code, self.vq_model_hr, self.latent_shape_hr)

    # ----------------------------------------
    # Lifecycle
    # ----------------------------------------

    def init_train(self):
        self.load()
        self.load_vq_model()
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
        self.load_vq_model()
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
        q_indices, z_hr, self.latent_shape_hr = self.encode_to_indices(self.H, self.vq_model_hr)
        q_lr, z_lr, _ = self.encode_to_indices(self.L, self.vq_model_lr)
        
        masked_indices, mask = self._mask_tokens(q_indices)

        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            logits = self.netG(masked_indices, z_lr)    # (B, L, num_embeddings)
            # Loss only on masked positions; logits[mask] -> (N_masked, num_embeddings)
            self.gen_loss = (
                F.cross_entropy(logits[mask], q_indices[mask])
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
        q_indices, z_hr, self.latent_shape_hr = self.encode_to_indices(self.H, self.vq_model_hr)
        q_lr, z_lr, _ = self.encode_to_indices(self.L, self.vq_model_lr)
        
        masked_indices, mask = self._mask_tokens(q_indices)

        logits = self.netG(masked_indices, z_lr)
        self.gen_loss = (
            F.cross_entropy(logits[mask], q_indices[mask])
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
        q_indices, z_hr, self.latent_shape_hr = self.encode_to_indices(self.H, self.vq_model_hr)
        q_lr, z_lr, _ = self.encode_to_indices(self.L, self.vq_model_lr)
        masked_indices, mask = self._mask_tokens(q_indices)
        logits = self.netG(masked_indices, z_lr)
        self.gen_loss = F.cross_entropy(logits[mask], q_indices[mask])
        self.G_valid_loss += self.gen_loss

    def validation_amp(self):
        q_indices, z_hr, self.latent_shape_hr = self.encode_to_indices(self.H, self.vq_model_hr)
        q_lr, z_lr, _ = self.encode_to_indices(self.L, self.vq_model_lr)
        masked_indices, mask = self._mask_tokens(q_indices)
        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            logits = self.netG(masked_indices, z_lr)
            self.gen_loss = F.cross_entropy(logits[mask], q_indices[mask])
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

        q_indices, z_hr, latent_shape = self.encode_to_indices(self.H, self.vq_model_hr)
        E_vq = self.decode_indices(q_indices, self.vq_model_hr, self.latent_shape_hr)
        E_maskgit = self.sample_from_transformer(n_samples=1)

        out_dict['H'] = self.H.detach()[0].float().cpu()
        out_dict['E_vq'] = E_vq.detach()[0].float().cpu()
        out_dict['E_maskgit'] = E_maskgit.detach()[0].float().cpu()

        return out_dict

    def log_comparison_image(self, img_dict, current_step, out_dtype=np.uint8):
        slice_idx = img_dict['H'].shape[-1] // 2
        H_slice = img_dict['H'][:, :, :, slice_idx]
        E_vq_slice = img_dict['E_vq'][:, :, :, slice_idx]
        E_maskgit_slice = img_dict['E_maskgit'][:, :, :, slice_idx]

        row = torch.stack([E_maskgit_slice, E_vq_slice, H_slice])
        grid = make_grid(row, nrow=len(row), padding=0).permute(1, 2, 0)
        grid_image = utils_3D_image.unnorm_and_rescale(grid, out_dtype)

        figure_string = "MaskGIT: %s, step %d: MaskGIT Sample, VQ Recon, HR" % (
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