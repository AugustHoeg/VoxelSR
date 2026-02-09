import os
import platform
import torch
import torch.nn as nn
from utils.utils_bnorm import merge_bn, tidy_sequential
from torch.nn.parallel import DataParallel, DistributedDataParallel

# This is an older version of model_base form SuperFormer, but another version can be found here:
# https://github.com/cszn/KAIR/blob/master/models/model_base.py#L129

class ModelBase():

    def __init__(self, opt):
        self.opt = opt                         # opt
        #self.save_dir = opt['path']['models']  # save models ## Not needed due to WandB handling saving of model
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.train_mode = opt['train_mode']        # training mode
        self.schedulers = []                   # schedulers
        self.model_param_mismatch = False  # Flag to indicate if model parameters mismatch during loading

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

    def save(self, label):
        pass

    def define_loss(self):
        pass

    def define_optimizer(self):
        pass

    def define_gradscaler(self):
        pass

    def define_mixed_precision(self):
        pass

    def define_scheduler(self):
        pass

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
        pass

    def current_losses(self):
        pass

    def update_learning_rate(self):  # Removed step n in scheduler.step() as its being deprecated.
        for scheduler in self.schedulers:
            scheduler.step()

    def current_learning_rate(self):
        return self.schedulers[0].get_lr()[0]

    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    """
    # ----------------------------------------
    # Information of net
    # ----------------------------------------
    """

    def print_network(self):
        pass

    def info_network(self):
        pass

    def print_params(self):
        pass

    def info_params(self):
        pass

    def compile_network(self, network, mode="default"):

        if not hasattr(torch, "compile"):
            # Older PyTorch
            return network

        system = platform.system()

        try:
            if system == "Windows":
                # Avoid Triton / Inductor on Windows
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


    def model_to_device(self, network):
        """Model to device. It also warps models with DistributedDataParallel or DataParallel.
        Args:
            network (nn.Module)
        """
        network = network.to(self.device)

        network = self.compile_network(network)

        if self.opt['dist']:
            find_unused_parameters = self.opt['find_unused_parameters']
            network = DistributedDataParallel(network, device_ids=[torch.cuda.current_device()], find_unused_parameters=find_unused_parameters)
        else:
            network = DataParallel(network)
        # If gpu_ids is none, force model to run on CPU
        if self.opt['gpu_ids'] is None:
            network = network.module.to(self.device)
        return network

    # ----------------------------------------
    # network name and number of parameters
    # ----------------------------------------
    def describe_network(self, network):
        network = self.get_bare_model(network)
        msg = '\n'
        msg += 'Networks name: {}'.format(network.__class__.__name__) + '\n'
        msg += 'Params number: {}'.format(sum(map(lambda x: x.numel(), network.parameters()))) + '\n'
        msg += 'Net structure:\n{}'.format(str(network)) + '\n'
        return msg


    """
    # ----------------------------------------
    # Save parameters
    # Load parameters
    # ----------------------------------------
    """

    # ----------------------------------------
    # save the state_dict of the network
    # ----------------------------------------
    def save_network(self, save_dir, network, network_label, iter_label):
        #save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_filename = '{}_{}.h5'.format(iter_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        network = self.get_bare_model(network)
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    # ----------------------------------------
    # load the state_dict of the network
    # ----------------------------------------
    # def load_network(self, load_path, network, strict=True, param_key='params', weights_only=True):
    #     network = self.get_bare_model(network)
    #     if strict:
    #         state_dict = torch.load(load_path, weights_only=weights_only)
    #         if param_key in state_dict.keys():
    #             state_dict = state_dict[param_key]
    #         network.load_state_dict(state_dict, strict=strict)
    #     else:
    #         state_dict_old = torch.load(load_path, weights_only=weights_only)
    #         if param_key in state_dict_old.keys():
    #             state_dict_old = state_dict_old[param_key]
    #         state_dict = network.state_dict()
    #         for ((key_old, param_old),(key, param)) in zip(state_dict_old.items(), state_dict.items()):
    #             state_dict[key] = param_old
    #         network.load_state_dict(state_dict, strict=True)
    #         del state_dict_old, state_dict

    def load_network(self, load_path, network, strict=True, param_key='params', weights_only=True):
        network = self.get_bare_model(network)
        state_dict_old = torch.load(load_path, weights_only=weights_only)

        if param_key in state_dict_old:
            state_dict_old = state_dict_old[param_key]

        state_dict_new = network.state_dict()

        # print("Checking for model parameter mismatch...")
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
                    matched_params[k] = v  # only load matching shape + same key

            # Update only the matched parameters
            state_dict.update(matched_params)
            network.load_state_dict(state_dict, strict=False)  # strict=False avoids errors

    # ----------------------------------------
    # save the state_dict of the optimizer
    # ----------------------------------------
    def save_optimizer(self, save_dir, optimizer, optimizer_label, iter_label):
        save_filename = '{}_{}.h5'.format(iter_label, optimizer_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)

    # ----------------------------------------
    # load the state_dict of the optimizer
    # ----------------------------------------
    def load_optimizer(self, load_path, optimizer, weights_only=True):
        optimizer.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()), weights_only=weights_only))

    # ----------------------------------------
    # save the state_dict of the scheduler
    # ----------------------------------------
    def save_scheduler(self, save_dir, scheduler, scheduler_label, iter_label):
        save_filename = '{}_{}.h5'.format(iter_label, scheduler_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(scheduler.state_dict(), save_path)

    # ----------------------------------------
    # load the state_dict of the scheduler
    # ----------------------------------------
    def load_scheduler(self, load_path, scheduler, weights_only=True):
        scheduler.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()), weights_only=weights_only))

    # ----------------------------------------
    # save the state_dict of the gradscaler
    # ----------------------------------------
    def save_gradscaler(self, save_dir, gradscaler, gradscaler_label, iter_label):
        save_filename = '{}_{}.h5'.format(iter_label, gradscaler_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(gradscaler.state_dict(), save_path)

    # ----------------------------------------
    # load the state_dict of the gradscaler
    # ----------------------------------------
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

    # ----------------------------------------
    # merge bn during training
    # ----------------------------------------
    def merge_bnorm_train(self):
        merge_bn(self.netG)
        tidy_sequential(self.netG)
        self.define_optimizer()
        self.define_scheduler()

    # ----------------------------------------
    # merge bn before testing
    # ----------------------------------------
    def merge_bnorm_test(self):
        merge_bn(self.netG)
        tidy_sequential(self.netG)