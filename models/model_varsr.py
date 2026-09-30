import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel

from models.model_base import ModelBase
from models.select_model import define_Model
from models.select_network import define_G
from performance_metrics.performance_metrics import compute_performance_metrics
from utils import utils_3D_image
from utils.load_options import load_options_from_experiment_id


class ModelVARSR(ModelBase):
    """Train VARSR"""
    def __init__(self, opt, mode='train', data_parallel=True):
        super(ModelVARSR, self).__init__(opt)
        self.last_iteration = 0
        self.netG_wo_ddp = define_G(opt, mode=mode)
        self.netG = self.model_to_device(self.netG_wo_ddp, data_parallel=data_parallel)

        if self.opt_train['E_decay'] > 0:
            self.netE = self.init_netE(opt)

        if opt['rank'] == 0 and mode == 'train':
            print("Number of trainable parameters, G", utils_3D_image.numel(self.netG, only_trainable=True))

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
            "Must specify pretrained_hr_vqmodel_id in path for ModelVARSR."
        )
        eid = self.opt["path"]["pretrained_hr_vqmodel_id"]
        self.vq_model_hr = self._load_vq_model(eid)

        if self.opt["compile"]:
            self.vq_model_hr.fhat_to_img = torch.compile(self.vq_model_hr.fhat_to_img, mode="max-autotune-no-cudagraphs")
            self.vq_model_hr.img_to_idxBl = torch.compile(self.vq_model_hr.img_to_idxBl, mode="max-autotune-no-cudagraphs")
            self.vq_model_hr.quantize.idxBl_to_var_input = torch.compile(self.vq_model_hr.quantize.idxBl_to_var_input, mode="max-autotune-no-cudagraphs")

    # ----------------------------------------
    # Encoding / decoding / sampling (VQ model always frozen)
    # ----------------------------------------

    @torch.no_grad()
    def encode_to_indices(self, x: torch.Tensor, vq_model: torch.nn.Module):
        """Encode a volume to codes via the frozen VQ encoder and creates VAR train input

        Args:
            x:        patch (B, C, H, W)
            vq_model: frozen VARVQVAE2D model
        Returns:
            gt_idx_Bl:         TODO
            gt_BL:             TODO
            x_BLCv_wo_first_l: TODO

        """
        gt_idx_Bl, idx_N_list = vq_model.img_to_idxBl(x)
        gt_BL = torch.cat(gt_idx_Bl[0:len(vq_model.quantize.v_patch_nums)], dim=1)
        x_BLCv_wo_first_l = vq_model.quantize.idxBl_to_var_input(idx_N_list)

        return gt_idx_Bl, gt_BL, x_BLCv_wo_first_l

    def _const_label(self, batch_size):
        """Constant class label for VARSR's class token."""
        return torch.zeros(batch_size, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def sample_E(self, lr_inp, batch_size = None, top_k=1, top_p=0.75, cfg=0.0, more_smooth=False):

        # Sample (cfg=0 -> image-based CFG disabled, see _const_label)
        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            f_hat = self.netG_wo_ddp.autoregressive_infer_cfg(
                B=batch_size,
                cfg=cfg,
                top_k=top_k,
                top_p=top_p,
                text_hidden=None,
                lr_inp=lr_inp,
                negative_text=None,
                label_B=self._const_label(batch_size),
                lr_inp_scale=None,
                more_smooth=more_smooth,
                tile_flag=True, # forces return of f_hat for decoding
                vae_proxy=(self.vq_model_hr,),
                vae_quant_proxy=(self.vq_model_hr.quantize,)
            )

        E = self.vq_model_hr.fhat_to_img(f_hat)
        return E

    def init_test(self, experiment_id):
        self.load(experiment_id, mode='test')
        self.load_hr_vq_model()
        self.netG.eval()
        self.define_metrics()
        self.define_mixed_precision()
        self.define_visual_eval()

    def init_train(self):
        self.load(from_pretrained_orig=self.opt['path']['from_pretrained_orig'])
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

    def define_wandb_run(self):
        self._init_wandb_run(extra_config={"up_factor": self.opt['up_factor']})
        self.model_artifact_G = wandb.Artifact(
            "Generator", type=self.opt['model_opt']['netG']['net_type'],
            description=self.opt['model_opt']['netG']['description'],
            metadata=OmegaConf.to_container(self.opt['model_opt']['netG'], resolve=True)
        )

    def _from_pretrained_orig(self, var, state_dict):
        for k, v in var.state_dict().items():
            if ".cross_attn." in k:
                if "mat_q" in k:
                    key = k.replace(".cross_attn", ".attn").replace("mat_q", "mat_qkv")
                    state_dict[k] = state_dict[key][0 : state_dict[key].shape[0] // 3, :]
                elif "mat_kv" in k:
                    key = k.replace(".cross_attn", ".attn").replace("mat_kv", "mat_qkv")
                    state_dict[k] = state_dict[key][state_dict[key].shape[0] // 3 :, : v.shape[1]]
                else:
                    key = k.replace(".cross_attn", ".attn")
                    state_dict[k] = state_dict[key]
            elif "class_emb" in k:
                value = state_dict[k]
                if value.shape[0] > v.shape[0]:
                    state_dict[k] = state_dict[k][: v.shape[0], :]
                elif value.shape[0] < v.shape[0]:
                    state_dict[k] = torch.cat((state_dict[k][:3830, :], state_dict[k][:3830, :]), dim=0)
        for key, value in var.state_dict().items():
            if key in state_dict and state_dict[key].shape != value.shape:
                print(key)
                state_dict.pop(key)
        ret = var.load_state_dict(state_dict, strict=False)
        missing, unexpected = ret
        print(f"[VARTrainer.load_state_dict] missing:  {missing}")
        print(f"[VARTrainer.load_state_dict] unexpected:  {unexpected}")
        del state_dict

        return var

    def _import_pretrained_orig(self, eid):
        """imports official VAR pretrained weights"""
        filename = self.opt['path'].get('pretrained_orig_filename', None)
        path = self._find_latest_checkpoint(eid, "saved_models", f"{filename}*")

        if self.opt['rank'] == 0:
            print(f"Importing VAR official weights [{self._short_path(path)}] ...")
        state_dict = torch.load(path, map_location='cpu', weights_only=False)
        # Official checkpoints are sometimes nested; unwrap the raw model state dict.
        if isinstance(state_dict, dict) and 'trainer' in state_dict:
            state_dict = state_dict['trainer'].get('var_wo_ddp', state_dict)
        self._from_pretrained_orig(self.get_bare_model(self.netG), state_dict)
        self.last_iteration = 0

    def load(self, experiment_id=None, mode='train', from_pretrained_orig=False):
        eid = self._resolve_eid(experiment_id)
        if mode == 'train':
            if self.opt['train_mode'] == 'scratch':
                return
            if from_pretrained_orig:                # finetune from official weights
                self._import_pretrained_orig(eid)
                return
            assert eid is not None, f"Pretrained experiment ID required for train_mode='{self.opt['train_mode']}'."
        else:
            assert eid is not None, "Experiment ID required for test mode."
        self.load_G(eid, mode)

    def define_loss(self):
        self.build_loss_fn_dict()
        self.init_G_loss_trackers()

    def define_optimizer(self):
        self.define_G_optimizer()

    def feed_data(self, data):
        self.L = data['L'].as_tensor().to(self.device, non_blocking=True)
        self.H = data['H'].as_tensor().to(self.device, non_blocking=True)
        self.L_up = F.interpolate(self.L, size=self.H.shape[2:], mode='bicubic', align_corners=False)

    def netG_forward(self):
        self.E = self.sample_E(self.L_up, batch_size=self.H.shape[0])

    def optimize_parameters_amp(self, current_step, update=False):

        B, V = self.H.shape[0], self.vq_model_hr.vocab_size  # batch size, vocabulary size

        # Encode VQ under mixed-precision and no-grad
        # TODO: Test if full precision during VQ encoding is really necessary
        gt_idx_Bl, gt_BL, x_BLCv_wo_first_l = self.encode_to_indices(self.H, self.vq_model_hr)

        # Forward VARSR
        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            logits_BLV, self.diff_loss, out_rgbs, mask_wo_prev_stages = self.netG(
                x_BLCv_wo_first_l,
                label_B=self._const_label(B),  # constant label; image-based CFG disabled (see D2)
                lr_inp=self.L_up,  # Use bicubic upsampled LR image as input to VARSR
                text_hidden=None,
                last_layer_gt=gt_idx_Bl[-1],
                last_layer_gt_discrete=gt_idx_Bl[-2],
                lr_inp_scale=None,
                vae_quant_proxy=(self.vq_model_hr.quantize,)
            )
            gt_BL = torch.cat((gt_BL[:, :-self.netG_wo_ddp.last_level_pns], gt_BL[:, -self.netG_wo_ddp.last_level_pns:][mask_wo_prev_stages].view(B, -1)), dim=1)
            logits_loss = F.cross_entropy(logits_BLV.contiguous().view(-1, V), gt_BL.view(-1), reduction='none').view(B, -1)
            self.logits_loss = logits_loss.mean(dim=-1).mean()
            self.gen_loss = (self.logits_loss + self.diff_loss * 2.0) / self.num_accum_steps_G

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

        # Log logits loss and diff loss separately
        self.run.log({"step": current_step, "G_logits_loss": self.logits_loss.item()})
        self.run.log({"step": current_step, "G_diff_loss": self.diff_loss.item()})

        grad_norm = self.G_train_grad_norm.item()
        self.run.log({"step": current_step, "G_train_grad_norm": grad_norm})

    def record_avg_train_log(self, current_step, idx_train):
        avg_loss = (self.G_train_loss.item() / idx_train) * self.num_accum_steps_G
        self.run.log({"step": current_step, "G_train_loss": avg_loss})

        self.G_train_loss = 0.0

    def test(self):
        self.netG.eval()
        with torch.inference_mode():
            self.netG_forward()
        self.netG.train()

    def validation(self):

        B, V = self.H.shape[0], self.vq_model_hr.vocab_size

        # Encode VQ under mixed-precision and no-grad
        gt_idx_Bl, gt_BL, x_BLCv_wo_first_l = self.encode_to_indices(self.H, self.vq_model_hr)

        # Forward VARSR
        logits_BLV, self.diff_loss, out_rgbs, mask_wo_prev_stages = self.netG(
            x_BLCv_wo_first_l,
            label_B=self._const_label(B),  # constant label; image-based CFG disabled (see D2)
            lr_inp=self.L_up,  # Use bicubic upsampled LR image as input to VARSR
            text_hidden=None,
            last_layer_gt=gt_idx_Bl[-1],
            last_layer_gt_discrete=gt_idx_Bl[-2],
            lr_inp_scale=None,
            vae_quant_proxy=(self.vq_model_hr.quantize,)
        )
        gt_BL = torch.cat((gt_BL[:, : -self.netG_wo_ddp.last_level_pns], gt_BL[:, -self.netG_wo_ddp.last_level_pns :][mask_wo_prev_stages].view(B, -1)), dim=1)
        logits_loss = F.cross_entropy(logits_BLV.contiguous().view(-1, V), gt_BL.view(-1), reduction='none').view(B, -1)
        self.logits_loss = logits_loss.mean(dim=-1).mean()
        self.gen_loss = (self.logits_loss + self.diff_loss * 2.0) / self.num_accum_steps_G

        self.G_valid_loss += self.gen_loss

        # Sample image
        self.E = self.sample_E(self.L_up, batch_size=self.H.shape[0])

        rescale_images = self.opt["dataset_opt"]["norm_type"] == "znormalization"
        compute_performance_metrics(self.E, self.H, self.metric_fn_dict, self.metric_val_dict, rescale_images=True)

    def validation_amp(self):

        B, V = self.H.shape[0], self.vq_model_hr.vocab_size

        # Encode VQ under mixed-precision and no-grad
        # TODO: Test if full precision during VQ encoding is really necessary
        gt_idx_Bl, gt_BL, x_BLCv_wo_first_l = self.encode_to_indices(self.H, self.vq_model_hr)

        # Forward VARSR
        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            logits_BLV, self.diff_loss, out_rgbs, mask_wo_prev_stages = self.netG(
                x_BLCv_wo_first_l,
                label_B=self._const_label(B),  # constant label; image-based CFG disabled (see D2)
                lr_inp=self.L_up,  # Use bicubic upsampled LR image as input to VARSR
                text_hidden=None,
                last_layer_gt=gt_idx_Bl[-1],
                last_layer_gt_discrete=gt_idx_Bl[-2],
                lr_inp_scale=None,
                vae_quant_proxy=(self.vq_model_hr.quantize,)
            )
            gt_BL = torch.cat((gt_BL[:, :-self.netG_wo_ddp.last_level_pns], gt_BL[:, -self.netG_wo_ddp.last_level_pns:][mask_wo_prev_stages].view(B, -1)), dim=1)
            logits_loss = F.cross_entropy(logits_BLV.contiguous().view(-1, V), gt_BL.view(-1), reduction='none').view(B, -1)
            self.logits_loss = logits_loss.mean(dim=-1).mean()
            self.gen_loss = (self.logits_loss + self.diff_loss * 2.0) / self.num_accum_steps_G

        self.G_valid_loss += self.gen_loss
        
        # Sample image
        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            self.E = self.sample_E(self.L_up, batch_size=self.H.shape[0])

        rescale_images = self.opt['dataset_opt']['norm_type'] == "znormalization"
        compute_performance_metrics(self.E, self.H, self.metric_fn_dict, self.metric_val_dict, rescale_images=True)
