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


class ModelTransformerVQ(ModelBase):
    """Trains a GPT-style transformer over VQ codebook indices from a frozen VQ model.

    The VQ model is loaded from a pretrained experiment (path.pretrained_vqmodel_id) and kept frozen. 
    The transformer learns to predict the next codebook index in an autoregressive manner, given a 
    sequence of input indices from the encoder.
    """

    def __init__(self, opt, mode='train', data_parallel=True):
        super().__init__(opt)
        self.last_iteration = 0

        self.netG = define_G(opt, mode=mode)
        self.netG = self.model_to_device(self.netG, data_parallel=data_parallel, compile=False)

        self.num_embeddings = opt['model_opt']['netG']['num_embeddings']

        self.vq_model = None
        self.latent_shape = None 

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

    def load_vq_model(self):

        assert 'pretrained_vqmodel_id' in self.opt['path'], "Must specify pre-trained VQ model for ModelTransformerVQ."
        eid = self.opt['path']['pretrained_vqmodel_id']
        opt_path = load_options_from_experiment_id(eid, root_dir="", file_type="yaml")
        opt_vq = OmegaConf.load(opt_path)
        opt_vq['dist'] = False

        net = define_Model(opt_vq, mode='test', data_parallel=False)
        net.load(eid, mode='test')
        self.vq_model = net.get_bare_model(net.netG).to(self.device)
        self.vq_model.eval()
        for p in self.vq_model.parameters():
            p.requires_grad_(False)

        assert self.num_embeddings == self.vq_model.codebook.num_embeddings, (
            f"num_embeddings mismatch: GPT has {self.num_embeddings}, "
            f"loaded VQ model has {self.vq_model.codebook.num_embeddings}."
        )

    # ----------------------------------------
    # Encoding / decoding / sampling (VQ model always frozen)
    # ----------------------------------------

    @torch.no_grad()
    def encode_to_indices(self, x: torch.Tensor):
        """Encode a volume to flat codebook indices via the frozen VQ model encoder.

        Args:
            x: (B, C, D, H, W)
        Returns:
            q_indices:    (B, D'*H'*W') int64 codebook indices
            latent_shape: (D', H', W') spatial dims of the latent grid
        """
        z_e = self.vq_model.encoder(x)
        latent_shape = tuple(z_e.shape[2:])  # (D', H', W')
        _, _, q_indices, _ = self.vq_model.codebook(z_e)
        return q_indices, latent_shape

    @torch.no_grad()
    def decode_indices(self, q_indices: torch.Tensor, latent_shape: tuple) -> torch.Tensor:
        """Decode flat codebook indices to a volume via the frozen VQ decoder.

        Args:
            q_indices:    (B, D'*H'*W') int64 codebook indices
            latent_shape: (D', H', W') spatial dims of the latent grid
        Returns:
            x_hat: (B, C, D, H, W) reconstructed volume
        """
        B = q_indices.shape[0]
        embed_dim = self.vq_model.codebook.embedding_dim
        z_q = self.vq_model.codebook.embedding(q_indices)           # (B, L, C)
        z_q = z_q.permute(0, 2, 1).view(B, embed_dim, *latent_shape)
        return self.vq_model.decode(z_q)

    @torch.no_grad()
    def sample_from_transformer(self, n_samples: int = 1, temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
        """Autoregressively sample volumes by generating token sequences from the GPT and decoding.

        Args:
            n_samples:   number of volumes to generate
            temperature: softmax temperature (higher = more random)
            top_k:       if set, restrict to top-k logits before sampling
        Returns:
            x_sampled: (n_samples, C, D, H, W) generated volumes
        """
        assert self.latent_shape is not None, "latent_shape not set; run encode_to_indices first."
        gpt = self.get_bare_model(self.netG)
        q_indices = gpt.sample(n_samples=n_samples, temperature=temperature, top_k=top_k, device=self.device)
        return self.decode_indices(q_indices, self.latent_shape)

    # ----------------------------------------
    # Lifecycle
    # ----------------------------------------

    def init_train(self):
        self.load()        # load transformer checkpoint if resuming
        self.load_vq_model()  # always required
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
        pass  # no SR comparison tool needed

    # ----------------------------------------
    # Training
    # ----------------------------------------

    def feed_data(self, data):
        self.H = data['H'].as_tensor().to(self.device, non_blocking=True)

    def optimize_parameters_amp(self, current_step, update=False):
        q_indices, self.latent_shape = self.encode_to_indices(self.H)

        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            logits = self.netG(q_indices)  # (B, L, num_embeddings)
            self.gen_loss = (
                F.cross_entropy(logits.view(-1, self.num_embeddings), q_indices.view(-1))
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
        q_indices, self.latent_shape = self.encode_to_indices(self.H)

        logits = self.netG(q_indices)
        self.gen_loss = (
            F.cross_entropy(logits.view(-1, self.num_embeddings), q_indices.view(-1))
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
        q_indices, self.latent_shape = self.encode_to_indices(self.H)
        logits = self.netG(q_indices)
        self.gen_loss = F.cross_entropy(
            logits.view(-1, self.num_embeddings), q_indices.view(-1)
        )
        self.G_valid_loss += self.gen_loss

    def validation_amp(self):
        q_indices, self.latent_shape = self.encode_to_indices(self.H)
        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            logits = self.netG(q_indices)
            self.gen_loss = F.cross_entropy(
                logits.view(-1, self.num_embeddings), q_indices.view(-1)
            )
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

        q_indices, latent_shape = self.encode_to_indices(self.H)
        E_vq = self.decode_indices(q_indices, latent_shape)
        E_gpt = self.sample_from_transformer(n_samples=1)

        out_dict['H'] = self.H.detach()[0].float().cpu()
        out_dict['E_vq'] = E_vq.detach()[0].float().cpu()
        out_dict['E_gpt'] = E_gpt.detach()[0].float().cpu()

        return out_dict

    def log_comparison_image(self, img_dict, current_step, out_dtype=np.uint8):
        slice_idx = img_dict['H'].shape[-1] // 2
        H_slice   = img_dict['H'][:, :, :, slice_idx]
        E_vq_slice = img_dict['E_vq'][:, :, :, slice_idx]
        E_gpt_slice = img_dict['E_gpt'][:, :, :, slice_idx]

        row = torch.stack([H_slice, E_vq_slice, E_gpt_slice])
        grid = make_grid(row, nrow=len(row), padding=0).permute(1, 2, 0)
        grid_image = utils_3D_image.unnorm_and_rescale(grid, out_dtype)

        figure_string = "VQTransformer: %s, step %d: HR, VQ Recon, GPT Sample" % (
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
