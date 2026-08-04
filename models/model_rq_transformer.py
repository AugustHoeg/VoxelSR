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


class ModelTransformerRQ(ModelBase):
    """Trains a RQtransformer over VQ codebook indices from a frozen RQVAE.

    The VQ model is loaded from a pretrained experiment (path.pretrained_vqmodel_id) and kept frozen. 
    The transformer learns to predict the next codebook depth in an autoregressive manner, by teacher forcing.
    """

    def __init__(self, opt, mode='train', data_parallel=True):
        super().__init__(opt)
        self.last_iteration = 0

        self.netG = define_G(opt, mode=mode)
        self.netG = self.model_to_device(self.netG, data_parallel=data_parallel)

        self.num_embeddings = opt['model_opt']['netG']['num_embeddings']
        self.n_rq_depth = opt['model_opt']['netG']['n_rq_depth']

        # Unconditional generation: no LR VQ model, no LR encoding, transformer sees lr_tokens=None
        self.unconditional = opt['model_opt']['netG'].get('lr_input_len', None) is None

        self.vq_model_hr = None
        self.vq_model_lr = None
        self.latent_shape_hr = None
        self.latent_shape_lr = None

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

        if self.opt["compile"]:
            self.vq_model_hr.encode = torch.compile(self.vq_model_hr.encode, mode="max-autotune-no-cudagraphs")
            self.vq_model_hr.decode_code = torch.compile(self.vq_model_hr.decode_code, mode="max-autotune-no-cudagraphs")

        assert self.num_embeddings == self.vq_model_hr.quantizer.codebooks[0].n_embed, (
            f"num_embeddings mismatch: transformer has {self.num_embeddings}, "
            f"HR VQ model has {self.vq_model_hr.quantizer.codebooks[0].n_embed}."
        )

    def load_lr_vq_model(self):
        self.vq_model_lr = None
        if "pretrained_lr_vqmodel_id" in self.opt["path"]:
            if self.opt["path"]["pretrained_lr_vqmodel_id"] is not None:
                eid = self.opt["path"]["pretrained_lr_vqmodel_id"]
                self.vq_model_lr = self._load_vq_model(eid)

    # ----------------------------------------
    # Encoding / decoding / sampling (VQ model always frozen)
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
        _, _, codes, _ = vq_model.quantizer(z_e)   # codes: (B, Dz, Dy, Dx, D)
        return codes, z_e, latent_shape


    @torch.no_grad()
    def decode_to_image(self, codes: torch.Tensor, vq_model: torch.nn.Module):
        """Decode RQ codes to a volume via the frozen VQ decoder.
        """
        z_q = vq_model.embed_code(codes)
        return vq_model.decode(z_q)
    
    
    def _flatten_lr_embeddings(self, z_lr: torch.Tensor) -> torch.Tensor:
        """Reshape LR encoder output (B, C, Dz, Dy, Dx) → (B, N_lr, C)."""
        B, C = z_lr.shape[:2]
        return z_lr.view(B, C, -1).permute(0, 2, 1)

    def _needs_code_vectors(self) -> bool:
        return getattr(self.get_bare_model(self.netG), 'head_emb_vqvae', False)

    def _get_code_vectors(self, codes: torch.Tensor) -> torch.Tensor:
        """Look up frozen RQVAE per-depth codebook vectors for `codes`.

        Args:
            codes: (..., D) int64 — any shape ending in the RQ depth axis.
        Returns:
            (..., D, input_embed_dim) codebook vectors (stacked, no cumsum).
        """
        embs = self.vq_model_hr.quantizer.embed_code_with_depth(codes)  # list[D × (..., input_embed_dim)]
        return torch.stack(embs, dim=-2)


    @torch.inference_mode()
    def sample_E(self, z_lr: torch.Tensor = None, batch_size: int = None,
                 temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
        """Raster-scan spatial + causal depth AR sampling, then RQVAE decode.

        Args:
            z_lr:        (B, C_lr, Dz, Dy, Dx) pre-encoded LR embeddings, or None
                         for unconditional generation.
            batch_size:  required when z_lr is None; ignored otherwise.
            temperature: softmax temperature (higher = more random).
            top_k:       if set, restrict to top-k logits before sampling.
        Returns:
            x_sampled: (B, C, D, H, W) reconstructed HR volume.
        """
        assert self.latent_shape_hr is not None, "latent_shape_hr not set; call encode_to_indices first."
        dz, dy, dx = self.latent_shape_hr
        D = self.n_rq_depth
        if z_lr is not None:
            B = z_lr.shape[0]
        else:
            assert batch_size is not None, "batch_size must be provided when z_lr is None."
            B = batch_size

        transformer = self.get_bare_model(self.netG)
        code_emb_fn = self._get_code_vectors if self._needs_code_vectors() else None
        codes_flat = transformer.sample(lr_tokens=z_lr, batch_size=B,
                                temperature=temperature, top_k=top_k,
                                code_emb_fn=code_emb_fn)                # (B, L, D)
        codes = codes_flat.reshape(B, dz, dy, dx, D).clamp(0, self.num_embeddings - 1)
        return self.vq_model_hr.decode_code(codes)

    # ----------------------------------------
    # Lifecycle
    # ----------------------------------------

    def init_train(self):
        self.load()        # load transformer checkpoint if resuming
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
        pass  # no SR comparison tool needed

    # ----------------------------------------
    # Training
    # ----------------------------------------

    def feed_data(self, data):
        self.H = data['H'].as_tensor().to(self.device, non_blocking=True)
        self.L = data['L'].as_tensor().to(self.device, non_blocking=True)

    def optimize_parameters_amp(self, current_step, update=False):
        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            codes, z_hr, self.latent_shape_hr = self.encode_to_indices(self.H, self.vq_model_hr)

        if self.unconditional:
            z_lr = None
        else:
            z_lr = self.L

        codes_flat = codes.reshape(codes.shape[0], -1, self.n_rq_depth)  # (B, L, D)
        code_vectors = self._get_code_vectors(codes_flat) if self._needs_code_vectors() else None

        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            logits = self.netG(codes, z_lr, code_vectors=code_vectors)   # list[D × (B, L, V)]
            self.gen_loss = self._rq_loss(logits, codes_flat) / self.num_accum_steps_G

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

        if self.unconditional:
            z_lr = None
        else:
            z_lr = self.L

        codes_flat = codes.reshape(codes.shape[0], -1, self.n_rq_depth)  # (B, L, D)
        code_vectors = self._get_code_vectors(codes_flat) if self._needs_code_vectors() else None

        logits = self.netG(codes, z_lr, code_vectors=code_vectors)       # list[D × (B, L, V)]
        self.gen_loss = self._rq_loss(logits, codes_flat) / self.num_accum_steps_G

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

    def _rq_loss(self, logits, codes_flat: torch.Tensor) -> torch.Tensor:
        """Fully-observed teacher-forced cross-entropy over all L*D targets."""
        logits_stack = torch.stack(logits, dim=2)                # (B, L, D, V)
        V = logits_stack.shape[-1]
        return F.cross_entropy(logits_stack.reshape(-1, V), codes_flat.reshape(-1))

    def validation(self):
        codes, _, self.latent_shape_hr = self.encode_to_indices(self.H, self.vq_model_hr)
        z_lr = None if self.unconditional else self.L
        codes_flat = codes.reshape(codes.shape[0], -1, self.n_rq_depth)
        code_vectors = self._get_code_vectors(codes_flat) if self._needs_code_vectors() else None

        logits = self.netG(codes, z_lr, code_vectors=code_vectors)
        self.gen_loss = self._rq_loss(logits, codes_flat)
        self.G_valid_loss += self.gen_loss

    def validation_amp(self):
        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            codes, _, self.latent_shape_hr = self.encode_to_indices(self.H, self.vq_model_hr)
        z_lr = None if self.unconditional else self.L
        codes_flat = codes.reshape(codes.shape[0], -1, self.n_rq_depth)
        code_vectors = self._get_code_vectors(codes_flat) if self._needs_code_vectors() else None

        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            logits = self.netG(codes, z_lr, code_vectors=code_vectors)
            self.gen_loss = self._rq_loss(logits, codes_flat)
        self.G_valid_loss += self.gen_loss

    # ----------------------------------------
    # Logging / visuals
    # ----------------------------------------

    def record_train_log(self, current_step):
        loss = self.G_train_loss.item() * self.num_accum_steps_G
        self.run.log({"step": current_step, "G_train_loss": loss})
        grad_norm = self.G_train_grad_norm.item()
        self.run.log({"step": current_step, "G_train_grad_norm": grad_norm})

    def current_visuals(self):
        out_dict = OrderedDict()

        if self.mixed_precision is not None:
            with torch.amp.autocast("cuda", dtype=self.mixed_precision):
                codes, _, self.latent_shape_hr = self.encode_to_indices(self.H, self.vq_model_hr)
        else:
            codes, _, self.latent_shape_hr = self.encode_to_indices(self.H, self.vq_model_hr)

        z_lr = None if self.unconditional else self.L

        if self.mixed_precision is not None:
            with torch.amp.autocast("cuda", dtype=self.mixed_precision):
                E_vq = self.vq_model_hr.decode_code(codes)
                E = self.sample_E(z_lr, batch_size=self.H.shape[0])
        else:
            E_vq = self.vq_model_hr.decode_code(codes)
            E = self.sample_E(z_lr, batch_size=self.H.shape[0])

        out_dict['H'] = self.H.detach()[0].float().cpu()
        out_dict['E_vq'] = E_vq.detach()[0].float().cpu()
        out_dict['E'] = E.detach()[0].float().cpu()

        return out_dict

    def log_comparison_image(self, img_dict, current_step, out_dtype=np.uint8):
        slice_idx = img_dict['H'].shape[-1] // 2
        H_slice = img_dict['H'][:, :, :, slice_idx]
        E_vq_slice = img_dict['E_vq'][:, :, :, slice_idx]
        E_slice = img_dict['E'][:, :, :, slice_idx]

        row = torch.stack([E_slice, E_vq_slice, H_slice])
        grid = make_grid(row, nrow=len(row), padding=0).permute(1, 2, 0)
        grid_image = utils_3D_image.unnorm_and_rescale(grid, out_dtype)

        figure_string = "RQTransformer: %s, step %d: AR Sample, VQ Recon, HR" % (
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
