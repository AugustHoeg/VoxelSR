import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel

from models.model_base import ModelBase
from models.select_model import define_Model
from models.select_network import define_G
from models.residual_self_correction_3d import ResidualSelfCorrection3D
from performance_metrics.performance_metrics import compute_performance_metrics
from utils import utils_3D_image
from utils.load_options import load_options_from_experiment_id


class ModelTransformerRQVAR(ModelBase):
    """Trains a RQVARtransformer over VQ codebook indices from a frozen RMSVQVAE.

    The VQ model is loaded from a pretrained experiment (path.pretrained_vqmodel_id) and kept frozen. 
    The transformer learns to predict the next codebook depth in an autoregressive manner, by teacher forcing.
    """

    def __init__(self, opt, mode='train', data_parallel=True):
        super(ModelTransformerRQVAR, self).__init__(opt)
        self.last_iteration = 0

        self.netG_wo_ddp = define_G(opt, mode=mode)
        self.netG = self.model_to_device(self.netG_wo_ddp, data_parallel=data_parallel, allow_compile=True)

        if self.opt_train['E_decay'] > 0:
            self.netE = self.init_netE(opt)

        if opt['rank'] == 0 and mode == 'train':
            print("Number of trainable parameters, G", utils_3D_image.numel(self.netG, only_trainable=True))

        # Parameters
        self.size_hr = self.opt['dataset_opt']['patch_size_hr']
        self.V = self.n_embed = self.opt['model_opt']['netG']['num_embeddings']
        self.D = self.n_rq_depth = self.opt['model_opt']['netG']['n_rq_depth']

        self.update = False
        self.early_stop = False
        self.min_validation_loss = float('inf')
        self.patience = self.opt_train['early_stop_patience']
        self.patience_counter = 0
        self.min_delta = 0

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
            "Must specify pretrained_hr_vqmodel_id in path for ModelTransformerRQVAR."
        )
        eid = self.opt["path"]["pretrained_hr_vqmodel_id"]
        self.vq_model_hr = self._load_vq_model(eid)

        if self.opt["compile"]:
            self.vq_model_hr.encode = torch.compile(self.vq_model_hr.encode, mode="max-autotune-no-cudagraphs")
            self.vq_model_hr.decode_multiscale = torch.compile(self.vq_model_hr.decode_multiscale, mode="max-autotune-no-cudagraphs")

        self.rsc = ResidualSelfCorrection3D(
            vae=self.vq_model_hr,
            noise_apply_layers=self.opt['model_opt']['netG']['rsc_noise_apply_layers'],
            noise_apply_strength=self.opt['model_opt']['netG']['rsc_noise_apply_strength'],
            noise_apply_requant=self.opt['model_opt']['netG']['rsc_noise_apply_requant'],
        )

    # ----------------------------------------
    # Encoding / decoding / sampling (VQ model always frozen)
    # ----------------------------------------

    @torch.no_grad()
    def encode_to_indices(self, x: torch.Tensor, vq_model: torch.nn.Module):
        """Encode a volume to codes via the frozen VQ encoder and creates RQVAR train input

        Args:
            x:        (B, C, D, H, W)
            vq_model: frozen RMSVQVAE3D model
        Returns:
            x_BLC_wo_prefix:         var input
            code_vectors:            multi-scale code vectors with added noise from RSC
            gt_ms_idx_Bl:            GT code vectors K list of (B, l_k, D) needed for RQVAR depth transformer
        """

        raw = vq_model.encode(x)
        x_BLC_wo_prefix, code_vectors, gt_ms_idx_Bl = self.rsc.flip_requant(raw, self.device)
        return x_BLC_wo_prefix, code_vectors, gt_ms_idx_Bl 


    @torch.inference_mode()
    def sample_E(self, vae, lr_tokens, cfg=0.0, temp=1.0, top_k=None, top_p=None):
        if cfg > 0.0:
            ms_idx = self.netG_wo_ddp.autoregressive_infer_cfg(
                vae, lr_tokens, cfg=cfg, temperature=temp, top_k=top_k, top_p=top_p)
        else:
            ms_idx = self.netG_wo_ddp.autoregressive_infer(
                vae, lr_tokens, temperature=temp, top_k=top_k, top_p=top_p)

        return vae.decode_multiscale(ms_idx)

    # ----------------------------------------
    # Lifecycle
    # ----------------------------------------

    def init_train(self):
        self.load()        # load transformer checkpoint if resuming
        self.load_hr_vq_model()

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

    # ----------------------------------------
    # Training
    # ----------------------------------------

    def feed_data(self, data):
        self.H = data['H'].as_tensor().to(self.device, non_blocking=True)
        self.L = data['L'].as_tensor().to(self.device, non_blocking=True)

    def netG_forward(self):  # This is the function run during inference
        self.E = self.sample_E(self.vq_model_hr, lr_tokens=self.L, cfg=0.0, temp=1.0, top_k=None, top_p=None)

    def optimize_parameters_amp(self, current_step, update=False):
        # TODO: figure out if encode_to_indices should be outside AMP
        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            x_BLC_wo_prefix, code_vectors, gt_ms_idx_Bl = self.encode_to_indices(self.H, self.vq_model_hr)
            logits = self.netG(x_BLC_wo_prefix, code_vectors, lr_tokens=self.L)  # list D of (B, L_total, V)

            target_codes = torch.cat(gt_ms_idx_Bl, dim=1)  # (B, L_total, D)
            self.gen_loss = self._rq_loss(logits, target_codes) / self.num_accum_steps_G

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
        x_BLC_wo_prefix, code_vectors, gt_ms_idx_Bl = self.encode_to_indices(self.H, self.vq_model_hr)
        logits = self.netG(x_BLC_wo_prefix, code_vectors, lr_tokens=self.L)  # list D of (B, L_total, V)

        target_codes = torch.cat(gt_ms_idx_Bl, dim=1)  # (B, L_total, D)
        self.gen_loss = self._rq_loss(logits, target_codes) / self.num_accum_steps_G

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

    def _rq_loss(self, logits, target_codes: torch.Tensor) -> torch.Tensor:
        """Fully-observed teacher-forced cross-entropy over all L*D targets."""
        logits_stack = torch.stack(logits, dim=2)  # list of D (B, L, V) -> (B, L, D, V)
        return F.cross_entropy(logits_stack.reshape(-1, self.V), target_codes.reshape(-1))

    def validation(self):
        gt_ms_idx_Bl = self.vq_model_hr.encode_multiscale(self.H)                          # list K of (B, l_k, D)
        x_BLC_wo_prefix = self.vq_model_hr.quantizer.idxBl_to_var_input(gt_ms_idx_Bl)       # (B, L_total - l0, Cvae)
        code_vectors = self.vq_model_hr.quantizer.idx_to_depth_code_vectors(gt_ms_idx_Bl)   # (B, L_total, D, Cvae)
        logits = self.netG(x_BLC_wo_prefix, code_vectors, lr_tokens=self.L)  # list D of (B, L_total, V)

        target_codes = torch.cat(gt_ms_idx_Bl, dim=1)  # (B, L_total, D)
        self.gen_loss = self._rq_loss(logits, target_codes)
        self.G_valid_loss += self.gen_loss

        # Sample image
        self.E = self.sample_E(self.vq_model_hr, lr_tokens=self.L, cfg=0.0, temp=1.0, top_k=None, top_p=None)

        rescale_images = self.opt["dataset_opt"]["norm_type"] == "znormalization"
        compute_performance_metrics(self.E, self.H, self.metric_fn_dict, self.metric_val_dict, rescale_images=True)

    def validation_amp(self):
        with torch.amp.autocast("cuda", dtype=self.mixed_precision):   # TODO: figure out if encode_multiscale should be outside AMP
            gt_ms_idx_Bl = self.vq_model_hr.encode_multiscale(self.H)                          # list K of (B, l_k, D)
            x_BLC_wo_prefix = self.vq_model_hr.quantizer.idxBl_to_var_input(gt_ms_idx_Bl)       # (B, L_total - l0, Cvae)
            code_vectors = self.vq_model_hr.quantizer.idx_to_depth_code_vectors(gt_ms_idx_Bl)   # (B, L_total, D, Cvae)
            logits = self.netG(x_BLC_wo_prefix, code_vectors, lr_tokens=self.L)  # list D of (B, L_total, V)

            target_codes = torch.cat(gt_ms_idx_Bl, dim=1)  # (B, L_total, D)
            self.gen_loss = self._rq_loss(logits, target_codes)
        self.G_valid_loss += self.gen_loss

        # Sample image
        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            self.E = self.sample_E(self.vq_model_hr, lr_tokens=self.L, cfg=0.0, temp=1.0, top_k=None, top_p=None)

        rescale_images = self.opt["dataset_opt"]["norm_type"] == "znormalization"
        compute_performance_metrics(self.E, self.H, self.metric_fn_dict, self.metric_val_dict, rescale_images=True)

    # ----------------------------------------
    # Logging / visuals
    # ----------------------------------------

    def record_train_log(self, current_step):
        loss = self.G_train_loss.item() * self.num_accum_steps_G
        self.run.log({"step": current_step, "G_train_loss": loss})
        grad_norm = self.G_train_grad_norm.item()
        self.run.log({"step": current_step, "G_train_grad_norm": grad_norm})
