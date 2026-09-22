import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel

from loss_functions.loss_functions_simple import compute_generator_loss
from models.model_base import ModelBase
from models.osediff.osediff_dec_lora import OSEDiff_gen, OSEDiff_reg, OSEDiff_test
from performance_metrics.performance_metrics import compute_performance_metrics
from utils import utils_3D_image


class ModelOSEDiff(ModelBase):
    """Train OSEDiff"""

    def __init__(self, opt, mode="train", data_parallel=True):
        super(ModelOSEDiff, self).__init__(opt)
        self.last_iteration = 0

        opt_net = opt['model_opt']['netG']
        self.ose_args = SimpleNamespace(
            pretrained_model_name_or_path=opt_net["pretrained_model_name_or_path"],
            lora_rank=opt_net["lora_rank"],
            cfg_vsd=opt_net["cfg_vsd"],
            device=self.device,
            mixed_precision=opt["train_opt"]["mixed_precision"],
            use_tiled_vae=False,
            merge_and_unload_lora=False,
            add_decoder_lora=opt_net["add_decoder_lora"],
        )
        
        if mode == 'train':
            self.netG = OSEDiff_gen(self.ose_args)
            self.netG = self.model_to_device(self.netG, data_parallel=data_parallel)

            self.netReg = OSEDiff_reg(self.ose_args, device=self.device)
            self.Reg_optimizer = None
    
            # prompt conditioning
            self.sample_names_as_prompt = self.opt['model_opt']['netG'].get('sample_names_as_prompt', True)
            self.prompt = self.opt['model_opt']['netG'].get("prompt", "")
            self.neg_prompt = self.opt['model_opt']['netG'].get("neg_prompt", "")
    
            # VSD loss weights
            self.lambda_vsd = self.opt['model_opt']['netG']["lambda_vsd"]
            self.lambda_vsd_lora = self.opt['model_opt']['netG']["lambda_vsd_lora"]

        elif mode == 'test':
            self.netG = OSEDiff_test(self.ose_args)
            self.netG = self.model_to_device(self.netG, data_parallel=data_parallel)

        self.update = False

        self.early_stop = False
        self.min_validation_loss = float("inf")
        self.patience = self.opt_train["early_stop_patience"]
        self.patience_counter = 0
        self.min_delta = 0
        

    def init_test(self, experiment_id):
        self.load(experiment_id, mode="test")
        self.netG.eval()
        self.define_metrics()
        self.define_mixed_precision()
        self.define_visual_eval()

    def init_train(self):
        self.load()
        self.get_bare_model(self.netG).set_train()
        self.get_bare_model(self.netReg).set_train()

        if self.opt["rank"] == 0:
            param_G = utils_3D_image.numel(self.get_bare_model(self.netG), only_trainable=True)
            param_Reg = utils_3D_image.numel(self.get_bare_model(self.netReg), only_trainable=True)
            print("Number of trainable parameters, G", param_G + param_Reg)

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
        self._init_wandb_run(extra_config={"up_factor": self.opt["up_factor"]})
        self.model_artifact_G = wandb.Artifact(
            "Generator",
            type=self.opt["model_opt"]["netG"]["net_type"],
            description=self.opt["model_opt"]["netG"]["description"],
            metadata=OmegaConf.to_container(self.opt["model_opt"]["netG"], resolve=True),
        )

    def define_loss(self):
        self.build_loss_fn_dict()
        self.init_G_loss_trackers()
        self.Reg_train_loss = 0.0
        self.Reg_train_grad_norm = torch.zeros(1)

    def define_optimizer(self):
        self.gen_params = []
        for n, _p in self.get_bare_model(self.netG).unet.named_parameters():
            if "lora" in n:
                self.gen_params.append(_p)
        self.gen_params += list(self.get_bare_model(self.netG).unet.conv_in.parameters())
        for n, _p in self.get_bare_model(self.netG).vae.named_parameters():
            if "lora" in n:
                self.gen_params.append(_p)

        self.define_G_optimizer(self.gen_params)

        self.reg_params = []
        for n, _p in self.get_bare_model(self.netReg).unet_update.named_parameters():
            if "lora" in n:
                self.reg_params.append(_p)

        self.define_named_optimizer(
            name="Reg",
            params=self.reg_params,
            optimizer_type=self.opt_train["G_optimizer_type"],
            optimizer_lr=self.opt_train["G_optimizer_lr"],
            weight_decay=self.opt_train["G_optimizer_wd"],
            betas=self.opt_train["G_optimizer_betas"],
            num_accum_steps=self.opt_train["num_accum_steps_G"],
        )
        
    def define_scheduler(self):
        self.define_G_scheduler()  # schedulers[0]
        self.schedulers.append(
            self._build_scheduler(
                self.Reg_optimizer,
                milestones_key="G_scheduler_milestones",
                gamma_key="G_scheduler_gamma",
                warmup_steps_key="G_warmup_steps",
                eta_min_key="G_eta_min",
                scheduler_type=self.opt_train.get("G_scheduler_type", "MultiStepLR"),
            )
        )
        
    def define_gradscaler(self):
        self.define_G_gradscaler()  # self.gen_scaler
        self.Reg_scaler = torch.amp.GradScaler("cuda")


    def _load_ckpt(self, eid, network, label):
        path = self._find_latest_checkpoint(eid, "saved_models", "*_{}.h5".format(label))
        if path is None:
            print("No {} checkpoint found, skipping loading...".format(label))
            return
        if self.opt['rank'] == 0:
            print("Loading {} [{}] ...".format(label, self._short_path(path)))
        self.get_bare_model(network).load_ckpt(torch.load(path))
        if label == "G":
            self.last_iteration = int(os.path.basename(path).split('_')[0])

    def load(self, experiment_id=None, mode='train'):
        eid = self._resolve_eid(experiment_id)
        if mode == 'train':
            if self.opt['train_mode'] == 'scratch':
                return
            assert eid is not None, f"Pretrained experiment ID required for train_mode='{self.opt['train_mode']}'."
        else:
            assert eid is not None, "Experiment ID required for test mode."
        self._load_ckpt(eid, self.netG, 'G')
        if mode == 'train':
            self._load_ckpt(eid, self.netReg, 'Reg')

    def save(self, iter_label):
        save_filename = "{}_{}.h5".format(iter_label, 'G')
        save_path = os.path.join(self._run_dir("saved_models"), save_filename)
        self.get_bare_model(self.netG).save_model(save_path)
        self.save_optimizer(self._run_dir("saved_optimizers"), self.G_optimizer, 'optimizerG', iter_label)
        self.save_scheduler(self._run_dir("saved_schedulers"), self.schedulers[0], 'schedulerG', iter_label)
        if self.mixed_precision is not None:
            self.save_gradscaler(self._run_dir("saved_gradscalers"), self.gen_scaler, 'gradscalerG', iter_label)

        save_filename = "{}_{}.h5".format(iter_label, "Reg")
        save_path = os.path.join(self._run_dir("saved_models"), save_filename)
        self.get_bare_model(self.netReg).save_model(save_path)
        self.save_optimizer(self._run_dir("saved_optimizers"), self.Reg_optimizer, 'optimizerReg', iter_label)
        self.save_scheduler(self._run_dir("saved_schedulers"), self.schedulers[1], 'schedulerReg', iter_label)
        if self.mixed_precision is not None:
            self.save_gradscaler(self._run_dir("saved_gradscalers"), self.Reg_scaler, 'gradscalerReg', iter_label)

    def load_optimizers(self, experiment_id=None):
        if self.opt['train_mode'] != 'resume':
            return
        eid = self._resolve_eid(experiment_id)
        assert eid is not None, "Pretrained experiment ID required for train_mode='resume'."
        self.opt['train_opt']['G_optimizer_reuse'] = True
        self.load_G_optimizer(eid)
        path = self._find_latest_checkpoint(eid, "saved_optimizers", "*optimizerReg.h5")
        if path is not None:
            self.load_optimizer(path, self.Reg_optimizer)

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
            self.load_gradscaler(path, self.Reg_scaler)

    def feed_data(self, data):
        self.L = data["L"].as_tensor().to(self.device, non_blocking=True)
        self.H = data["H"].as_tensor().to(self.device, non_blocking=True)
        self.sample_names = data["sample_name"]  # [sample_name, sample_name, ...]

    def _split_filename(self, filename):
        str1 = filename.split("_")[0]
        return f"{str1.lower()}"

    def _prompt(self, b):
        neg_prompt = [self.neg_prompt] * b
        if self.sample_names_as_prompt:
            prompt = [f"Micro-CT scan of {self._split_filename(name)} sample" for name in self.sample_names]
        else:
            prompt = [self.prompt] * b
        return {'prompt': prompt, 'neg_prompt': neg_prompt}

    def gen_forward(self):
        self.L_up = F.interpolate(self.L, size=self.H.shape[2:], mode='bicubic', align_corners=False)
        batch_prompt = self._prompt(self.L_up.shape[0])
        self.E, self.x_denoised, self.prompt_embeds, self.neg_prompt_embeds = self.netG(self.L_up, batch=batch_prompt)

    def netG_forward(self):
        self.L_up = F.interpolate(self.L, size=self.H.shape[2:], mode='bicubic', align_corners=False)
        batch_prompt = self._prompt(self.L_up.shape[0])
        self.E = self.netG(self.L_up, batch=batch_prompt)

    def optimize_parameters_amp(self, current_step, update=False):
        
        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            self.gen_forward()
            self.recon_loss = compute_generator_loss(self.H, self.E, self.loss_fn_dict, self.loss_val_dict, device=self.device)
            self.loss_vsd = self.netReg.distribution_matching_loss(self.x_denoised, self.prompt_embeds, self.neg_prompt_embeds, self.ose_args)
            self.gen_loss = self.recon_loss + self.lambda_vsd * self.loss_vsd
            self.gen_loss = self.gen_loss / self.num_accum_steps_G

        self.G_train_loss = self.gen_loss
        if self.opt["rank"] == 0:
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
            G_clipgrad_max = self.opt_train["G_optimizer_clipgrad"]
            if G_clipgrad_max > 0:
                self.gen_scaler.unscale_(self.G_optimizer)
                self.G_train_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.gen_params, max_norm=G_clipgrad_max, norm_type=2
                )
            self.gen_scaler.step(self.G_optimizer)
            self.gen_scaler.update()
            self.G_optimizer.zero_grad()
            self.G_accum_count = 0
        else:
            self.G_accum_count += 1

        # ---- Diff loss ----
        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            loss_d = self.netReg.diff_loss(self.x_denoised, self.prompt_embeds, self.ose_args)
            self.reg_loss = (self.lambda_vsd_lora * loss_d) / self.num_accum_steps_G
        
        self.Reg_train_loss = self.reg_loss
        if self.opt["rank"] == 0:
            print("Reg train loss:", self.Reg_train_loss.item())

        self.update = ((self.Reg_accum_count + 1) % self.num_accum_steps_Reg) == 0 or update
        
        if not self.update:
            if isinstance(self.netReg, DistributedDataParallel):
                with self.netReg.no_sync():
                    self.Reg_scaler.scale(self.reg_loss).backward()
            else:
                self.Reg_scaler.scale(self.reg_loss).backward()
        else:
            self.Reg_scaler.scale(self.reg_loss).backward()

        if self.update:
            Reg_clipgrad_max = self.opt_train["G_optimizer_clipgrad"]
            if Reg_clipgrad_max > 0:
                self.Reg_scaler.unscale_(self.Reg_optimizer)
                self.Reg_train_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.reg_params, max_norm=Reg_clipgrad_max, norm_type=2
                )
            self.Reg_scaler.step(self.Reg_optimizer)
            self.Reg_scaler.update()
            self.Reg_optimizer.zero_grad()
            self.Reg_accum_count = 0
        else:
            self.Reg_accum_count += 1


    def record_train_log(self, current_step):
        self.run.log({"step": current_step, "G_train_loss": self.G_train_loss.item() * self.num_accum_steps_G})
        self.run.log({"step": current_step, "Reg_train_loss": self.Reg_train_loss.item() * self.num_accum_steps_G})
        self.run.log({"step": current_step, "G_train_grad_norm": self.G_train_grad_norm.item()})
        self.run.log({"step": current_step, "Reg_train_grad_norm": self.Reg_train_grad_norm.item()})
        self.run.log({"step": current_step, "G_recon_loss": self.recon_loss.item()})

    def record_avg_train_log(self, current_step, idx_train):
        self.run.log({"step": current_step, "G_train_loss": (self.G_train_loss.item() / idx_train) * self.num_accum_steps_G})
        self.run.log({"step": current_step, "Reg_train_loss": (self.Reg_train_loss.item() / idx_train) * self.num_accum_steps_G})
        self.G_train_loss = 0.0
        self.Reg_train_loss = 0.0

    def test(self):
        self.netG.eval()
        with torch.inference_mode():
            self.gen_forward()
        self.netG.train()

    def validation(self):
        self.gen_forward()
        self.gen_loss = compute_generator_loss(self.H, self.E, self.loss_fn_dict, self.loss_val_dict, device=self.device)
        self.G_valid_loss += self.gen_loss

        rescale_images = self.opt["dataset_opt"]["norm_type"] == "znormalization"
        compute_performance_metrics(self.E, self.H, self.metric_fn_dict, self.metric_val_dict, rescale_images=True)

    def validation_amp(self):
        with torch.amp.autocast("cuda", dtype=self.mixed_precision):
            self.gen_forward()
            self.gen_loss = compute_generator_loss(self.H, self.E, self.loss_fn_dict, self.loss_val_dict, device=self.device)

        self.G_valid_loss += self.gen_loss

        rescale_images = self.opt["dataset_opt"]["norm_type"] == "znormalization"
        compute_performance_metrics(self.E, self.H, self.metric_fn_dict, self.metric_val_dict, rescale_images=True)
