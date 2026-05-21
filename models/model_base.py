import glob
import os
import platform
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim import lr_scheduler

import config
from performance_metrics.performance_metrics import NRMSE_2D, NRMSE_3D, PSNR_2D, PSNR_3D, SSIM_2D, SSIM_3D
from utils.utils_3D_image import crop_center
from utils.utils_bnorm import merge_bn, tidy_sequential
from utils.utils_dist import get_rank, reduce_max, reduce_sum


class ModelBase():

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.train_mode = opt['train_mode']
        self.opt_train = self.opt.get('train_opt', {})
        self.schedulers = []
        self.model_param_mismatch = False

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    def init_train(self):
        pass

    def define_wandb_run(self):
        pass

    def load(self):
        pass

    # ----------------------------------------
    # Checkpoint discovery helpers
    # ----------------------------------------

    def _find_latest_checkpoint(self, experiment_id, subdir, filename_pattern):
        """Return the most recently modified checkpoint file matching the pattern, or None."""
        pattern = os.path.join(
            config.ROOT_DIR, "logs", "*", "wandb",
            "*" + experiment_id, "files", subdir, filename_pattern
        )
        matches = glob.glob(pattern)
        if not matches:
            return None
        matches.sort(key=os.path.getmtime, reverse=True)
        return matches[0]

    def _short_path(self, path, n=4):
        """Return the last n components of a path for concise log messages."""
        return os.sep.join(os.path.normpath(path).split(os.sep)[-n:])

    def _run_dir(self, subdir):
        """Return the path to a named subdirectory within the current WandB run."""
        return os.path.join(self.run.dir, subdir)

    # ----------------------------------------
    # Default G-only checkpoint load/save
    # GAN-style models override to also handle D
    # ----------------------------------------

    def save(self, iter_label):
        self.save_network(self._run_dir("saved_models"), self.netG, 'G', iter_label)
        self.save_optimizer(self._run_dir("saved_optimizers"), self.G_optimizer, 'optimizerG', iter_label)
        self.save_scheduler(self._run_dir("saved_schedulers"), self.schedulers[0], 'schedulerG', iter_label)
        if self.mixed_precision is not None:
            self.save_gradscaler(self._run_dir("saved_gradscalers"), self.gen_scaler, 'gradscalerG', iter_label)
        if hasattr(self, 'netE'):
            self.save_network(self._run_dir("saved_models"), self.netE, 'E', iter_label)

    def load_optimizers(self, experiment_id=None):
        if self.opt['train_mode'] != 'resume':
            return
        eid = self.opt['path']['pretrained_experiment_id'] if experiment_id is None else experiment_id
        assert eid is not None, "Pretrained experiment ID required for train_mode='resume'."
        self.opt['train_opt']['G_optimizer_reuse'] = True

        path = self._find_latest_checkpoint(eid, "saved_optimizers", "*optimizerG.h5")
        if path is None:
            print("No G optimizer checkpoint found, skipping...")
            return
        if self.opt['rank'] == 0:
            print(f"Loading G optimizer [{self._short_path(path)}] ...")
        self.load_optimizer(path, self.G_optimizer)

    def load_schedulers(self, experiment_id=None):
        if self.opt['train_mode'] != 'resume':
            return
        eid = self.opt['path']['pretrained_experiment_id'] if experiment_id is None else experiment_id
        assert eid is not None, "Pretrained experiment ID required for train_mode='resume'."

        path = self._find_latest_checkpoint(eid, "saved_schedulers", "*schedulerG.h5")
        if path is None:
            print("No G scheduler checkpoint found, skipping...")
            return
        if self.opt['rank'] == 0:
            print(f"Loading G scheduler [{self._short_path(path)}] ...")
        self.load_scheduler(path, self.schedulers[0])

    def load_gradscalers(self, experiment_id=None):
        if self.opt['train_mode'] != 'resume':
            return
        eid = self.opt['path']['pretrained_experiment_id'] if experiment_id is None else experiment_id
        assert eid is not None, "Pretrained experiment ID required for train_mode='resume'."

        path = self._find_latest_checkpoint(eid, "saved_gradscalers", "*gradscalerG.h5")
        if path is None:
            print("No G gradscaler checkpoint found, skipping...")
            return
        if self.opt['rank'] == 0:
            print(f"Loading G gradscaler [{self._short_path(path)}] ...")
        self.load_gradscaler(path, self.gen_scaler)

    # ----------------------------------------
    # Loss function construction
    # ----------------------------------------

    def build_loss_fn_dict(self):
        """Build loss_fn_dict and loss_val_dict from G_loss_weights config.
        All loss-specific imports are lazy so heavy libraries are only loaded when needed.
        """
        self.loss_fn_dict = {}
        self.loss_val_dict = self.opt_train['G_loss_weights']

        for key, value in self.loss_val_dict.items():
            if key == "MSE" and value > 0:
                self.loss_fn_dict["MSE"] = nn.MSELoss()
            elif key == "L1" and value > 0:
                self.loss_fn_dict["L1"] = nn.L1Loss()
            elif key == "BCE_Logistic" and value > 0:
                self.loss_fn_dict["BCE_Logistic"] = nn.BCEWithLogitsLoss()
            elif key == "BCE" and value > 0:
                self.loss_fn_dict["BCE"] = nn.BCELoss()
            elif key == "LPIPS" and value > 0:
                from loss_functions.loss_functions_simple import LPIPSLoss3D
                self.loss_fn_dict["LPIPS"] = LPIPSLoss3D(
                    net_type='alex', version='0.1', device=self.device, axes=(0, 1, 2)
                )
            elif key == "FSC" and value > 0:
                from loss_functions.loss_functions_simple import FSCLoss3D
                self.loss_fn_dict["FSC"] = FSCLoss3D(
                    size=self.opt['dataset_opt']['patch_size_hr'], delta=1, alpha=2.0,
                    drop_DC=False, device=self.device
                )
            elif key == "CSC" and value > 0:
                from loss_functions.loss_functions_simple import CSCLoss
                self.loss_fn_dict["CSC"] = CSCLoss(
                    eval_mode=True, verbose=True, feat_dist_func='FSC',
                    compare_input=False, device=self.device,
                    size=self.opt['dataset_opt']['patch_size_hr'],
                    experiment_id=self.opt_train['pretrained_G_loss_IDs']['CSC']
                )
            elif key == "AESOP3D" and value > 0:
                from loss_functions.loss_functions_simple import AESOPLoss3D
                self.loss_fn_dict["AESOP3D"] = AESOPLoss3D(
                    ae_criterion_type='L1', ae_weight=1.0,
                    experiment_id=self.opt_train['pretrained_G_loss_IDs']['AESOP3D'],
                )

    def init_G_loss_trackers(self):
        self.G_train_loss = 0.0
        self.G_valid_loss = 0.0
        self.G_train_grad_norm = 0.0
        self.G_valid_grad_norm = 0.0

    def init_D_loss_trackers(self):
        self.D_train_loss = 0.0
        self.D_valid_loss = 0.0
        self.D_train_grad_norm = 0.0
        self.D_valid_grad_norm = 0.0

    def define_loss(self):
        pass

    def define_optimizer(self):
        pass

    def define_G_gradscaler(self):
        self.gen_scaler = torch.amp.GradScaler("cuda")

    def define_D_gradscaler(self):
        self.dis_scaler = torch.amp.GradScaler("cuda")

    def define_gradscaler(self):
        """Default: G only. GAN models override to also call define_D_gradscaler()."""
        self.define_G_gradscaler()

    def define_mixed_precision(self):
        if self.opt_train['mixed_precision'] == "FP16":
            self.mixed_precision = torch.float16
            self.define_gradscaler()
        elif self.opt_train['mixed_precision'] == "FP32":
            self.mixed_precision = torch.float32
            self.define_gradscaler()
        elif self.opt_train['mixed_precision'] == "BF16":
            self.mixed_precision = torch.bfloat16
            self.define_gradscaler()
        else:
            self.mixed_precision = None

    def define_metrics(self):
        self.metric_fn_dict = {}
        self.metric_val_dict = {}

        if "psnr" in self.opt_train['performance_metrics']:
            self.metric_val_dict["psnr"] = 0.0
            self.metric_fn_dict["psnr"] = PSNR_3D() if self.opt['input_type'] == '3D' else PSNR_2D()
        if "ssim" in self.opt_train['performance_metrics']:
            self.metric_val_dict["ssim"] = 0.0
            self.metric_fn_dict["ssim"] = SSIM_3D() if self.opt['input_type'] == '3D' else SSIM_2D()
        if "nrmse" in self.opt_train['performance_metrics']:
            self.metric_val_dict["nrmse"] = 0.0
            self.metric_fn_dict["nrmse"] = NRMSE_3D() if self.opt['input_type'] == '3D' else NRMSE_2D()

    def _build_scheduler(self, optimizer, milestones_key, gamma_key, warmup_steps_key):
        """Build a MultiStepLR scheduler with an optional LinearLR warmup prefix."""
        multistep = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.opt_train[milestones_key],
            gamma=self.opt_train[gamma_key]
        )
        warmup_steps = self.opt_train.get(warmup_steps_key, 0)
        if warmup_steps > 0:
            warmup = lr_scheduler.LinearLR(
                optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
            )
            return lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, multistep], milestones=[warmup_steps]
            )
        return multistep

    def define_G_scheduler(self):
        self.schedulers.append(self._build_scheduler(
            self.G_optimizer,
            milestones_key='G_scheduler_milestones',
            gamma_key='G_scheduler_gamma',
            warmup_steps_key='G_warmup_steps'
        ))

    def define_D_scheduler(self):
        self.schedulers.append(self._build_scheduler(
            self.D_optimizer,
            milestones_key='D_scheduler_milestones',
            gamma_key='D_scheduler_gamma',
            warmup_steps_key='D_warmup_steps'
        ))

    def define_scheduler(self):
        """Default: G only. GAN models override to also call define_D_scheduler()."""
        self.define_G_scheduler()

    def define_visual_eval(self):
        if self.opt['input_type'] == '2D':
            from utils.utils_2D_image import ImageComparisonTool2D as comparison_tool
        elif self.opt['input_type'] == '3D':
            from utils.utils_3D_image import ImageComparisonTool3D as comparison_tool

        self.comparison_tool = comparison_tool(
            patch_size_hr=self.opt['dataset_opt']['patch_size_hr'],
            upscaling_methods=["tio_nearest", "tio_linear"],
            unnorm=self.opt['dataset_opt']['norm_type'] == 'znormalization',
            div_max=self.opt['dataset_opt']['norm_type'] == 'znormalization',
            out_dtype=np.uint8
        )

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def optimize_parameters_amp(self):
        pass

    def current_visuals(self):
        out_dict = OrderedDict()

        roi = int(self.opt['dataset_opt']['patch_size_hr'] / self.opt['up_factor'])
        if self.opt['dataset_opt']['patch_size'] > roi:
            out_dict['L'] = crop_center(self.L, center_size=roi).detach()[0].float().cpu()
        else:
            out_dict['L'] = self.L.detach()[0].float().cpu()

        out_dict['E'] = self.E.detach()[0].float().cpu()
        out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict

    def current_losses(self):
        pass

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def current_learning_rate(self):
        return self.schedulers[0].get_lr()[0]

    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def set_eval_mode(self):
        self.netG.eval()

    def set_train_mode(self):
        self.netG.train()

    def early_stopping(self, current_step, idx_train):
        validation_loss = self.G_valid_loss / idx_train

        if validation_loss.item() < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.patience_counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.patience_counter += 1
            if (self.patience_counter >= self.patience) and current_step > 75000:
                self.early_stop = True

        early_stop_tensor = torch.tensor(int(self.early_stop), device=self.device)
        early_stop_tensor = reduce_max(early_stop_tensor)
        self.early_stop = early_stop_tensor.item() == 1

    def record_test_log(self, idx_test):
        idx_tensor = torch.tensor(idx_test, device=self.device)
        global_idx_tensor = reduce_sum(idx_tensor)

        for key, value in self.metric_val_dict.items():
            metric_tensor = torch.tensor(float(value), device=self.device)
            global_average = reduce_sum(metric_tensor) / global_idx_tensor

            if get_rank() == 0:
                metric_name = "Average " + key
                self.run.log({metric_name: global_average.item()})
                print(metric_name, global_average.item())

            self.metric_val_dict[key] = 0.0

        loss_tensor = torch.tensor(float(self.G_valid_loss), device=self.device)
        global_valid_loss = reduce_sum(loss_tensor) / global_idx_tensor

        if get_rank() == 0:
            self.run.log({"G_valid_loss": global_valid_loss.item()})
            print("G_valid_loss", global_valid_loss.item())

        self.G_valid_loss = 0.0

    def log_comparison_image(self, img_dict, current_step):
        grid_image = self.comparison_tool.get_comparison_image(img_dict)
        figure_string = "SR comparison: %s, step %d, %dx upscaling" % (
            self.opt['model_opt']['model_architecture'], current_step, self.opt['up_factor']
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

    """
    # ----------------------------------------
    # Information of net
    # ----------------------------------------
    """

    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)

    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg

    def compile_network(self, network, mode="default"):

        if not hasattr(torch, "compile"):
            return network

        system = platform.system()

        try:
            if system == "Windows":
                print("Using torch.compile with backend='aot_eager' (Windows safe mode)")
                return torch.compile(network, backend="aot_eager", mode=mode)
            else:
                print("Using torch.compile with backend='inductor'")
                return torch.compile(network, backend="inductor", mode=mode)

        except Exception as e:
            print(f"torch.compile failed: {e}")
            print("Falling back to eager mode.")
            return network

    def get_bare_model(self, network):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(network, (DataParallel, DistributedDataParallel)):
            network = network.module
        return network

    def model_to_device(self, network, data_parallel=True, compile=False):
        """Model to device. It also warps models with DistributedDataParallel or DataParallel.
        Args:
            network (nn.Module)
        """
        network = network.to(self.device)

        if compile:
            network = self.compile_network(network)

        if data_parallel:
            if self.opt['dist']:
                find_unused_parameters = self.opt['find_unused_parameters']
                network = DistributedDataParallel(
                    network,
                    device_ids=[torch.cuda.current_device()],
                    find_unused_parameters=find_unused_parameters,
                    static_graph=self.opt['use_static_graph']
                )
            else:
                network = DataParallel(network)

        if self.opt['gpu_ids'] is None:
            network = network.module.to(self.device)

        return network

    def describe_network(self, network):
        network = self.get_bare_model(network)
        msg = '\n'
        msg += 'Networks name: {}'.format(network.__class__.__name__) + '\n'
        msg += 'Params number: {}'.format(sum(map(lambda x: x.numel(), network.parameters()))) + '\n'
        msg += 'Net structure:\n{}'.format(str(network)) + '\n'
        return msg

    def describe_params(self, network):
        network = self.get_bare_model(network)
        msg = '\n'
        msg += 'Params (name, shape, number):\n'
        for name, param in network.named_parameters():
            msg += '{:s}: {:s}, {:d}\n'.format(name, str(param.shape), param.numel())
        return msg

    """
    # ----------------------------------------
    # Save parameters
    # Load parameters
    # ----------------------------------------
    """

    def save_network(self, save_dir, network, network_label, iter_label):
        save_filename = '{}_{}.h5'.format(iter_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        network = self.get_bare_model(network)
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_network(self, load_path, network, strict=True, param_key='params', weights_only=True):
        network = self.get_bare_model(network)
        state_dict_old = torch.load(load_path, weights_only=weights_only)

        if param_key in state_dict_old:
            state_dict_old = state_dict_old[param_key]

        state_dict_new = network.state_dict()

        model_param_mismatch = False
        for k, v in state_dict_old.items():
            if k not in state_dict_new or state_dict_new[k].shape != v.shape:
                model_param_mismatch = True
                break
        for k in state_dict_new.keys():
            if k not in state_dict_old:
                model_param_mismatch = True
                break
        self.model_param_mismatch = model_param_mismatch

        if strict:
            network.load_state_dict(state_dict_old, strict=True)
        else:
            state_dict = network.state_dict()
            matched_params = {}
            for k, v in state_dict_old.items():
                if k in state_dict and state_dict[k].shape == v.shape:
                    matched_params[k] = v

            state_dict.update(matched_params)
            network.load_state_dict(state_dict, strict=False)

    def save_optimizer(self, save_dir, optimizer, optimizer_label, iter_label):
        save_filename = '{}_{}.h5'.format(iter_label, optimizer_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)

    def load_optimizer(self, load_path, optimizer, weights_only=True):
        optimizer.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()), weights_only=weights_only))

    def save_scheduler(self, save_dir, scheduler, scheduler_label, iter_label):
        save_filename = '{}_{}.h5'.format(iter_label, scheduler_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(scheduler.state_dict(), save_path)

    def load_scheduler(self, load_path, scheduler, weights_only=True):
        scheduler.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()), weights_only=weights_only))

    def save_gradscaler(self, save_dir, gradscaler, gradscaler_label, iter_label):
        save_filename = '{}_{}.h5'.format(iter_label, gradscaler_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(gradscaler.state_dict(), save_path)

    def load_gradscaler(self, load_path, gradscaler, weights_only=True):
        gradscaler.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()), weights_only=weights_only))

    def update_E(self, decay=0.999):
        netG = self.get_bare_model(self.netG)
        netG_params = dict(netG.named_parameters())
        netE_params = dict(self.netE.named_parameters())
        for k in netG_params.keys():
            netE_params[k].data.mul_(decay).add_(netG_params[k].data, alpha=1-decay)

    """
    # ----------------------------------------
    # Merge Batch Normalization for training
    # Merge Batch Normalization for testing
    # ----------------------------------------
    """

    def merge_bnorm_train(self):
        merge_bn(self.netG)
        tidy_sequential(self.netG)
        self.define_optimizer()
        self.define_scheduler()

    def merge_bnorm_test(self):
        merge_bn(self.netG)
        tidy_sequential(self.netG)
