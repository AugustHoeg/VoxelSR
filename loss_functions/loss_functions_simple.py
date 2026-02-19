import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp
import lpips
from omegaconf import OmegaConf

from utils.fourier_ring_correlation import fourier_shell_correlation, get_shell_masks_3d
from utils.load_options import load_options_from_experiment_id


def compute_discriminator_loss(prop_real, prop_fake):
    dis_loss_fake = F.binary_cross_entropy_with_logits(prop_fake, torch.zeros_like(prop_fake))
    dis_loss_real = F.binary_cross_entropy_with_logits(prop_real, torch.ones_like(prop_real) - 0.1 * torch.ones_like(prop_real))
    # Formulate Discriminator loss: Max log(D(I_HR)) + log(1 - D(G(I_LR)))
    dis_loss = dis_loss_real + dis_loss_fake

    return dis_loss

def compute_gradient_penalty(interpolated_images, mixed_scores, device="cuda"):

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def compute_critic_loss(critic_real, critic_fake, scaled_gradient_penalty):

    loss_critic = -(torch.mean(critic_real.reshape(-1)) - torch.mean(critic_fake.reshape(-1))) + scaled_gradient_penalty
    return loss_critic

def compute_generator_loss(real_hi_res=None, fake_hi_res=None, loss_fn_dict=None, loss_val_dict=None, prop_real=None, prop_fake=None, model="plain", device="cuda"):

    gen_loss = torch.tensor(0.0).to(device)
    aux_loss = torch.tensor(0.0).to(device)

    for key, value in loss_val_dict.items():
        if value > 0 and key != 'ADV':
            aux_loss += value*loss_fn_dict[key](real_hi_res, fake_hi_res)

    gen_loss += aux_loss

    return gen_loss


def bce_loss(y_real, y_pred):
    """
    Simple binary cross entropy loss
    :param y_real: target value (label)
    :param y_pred: predicted value
    :return: loss
    """
    return torch.mean(y_pred - y_real*y_pred + torch.log(1 + torch.exp(-y_pred)))


class LPIPSLoss3D(torch.nn.Module):
    def __init__(self, net_type='alex', version='0.1', device="cuda", axes=(0, 1, 2)):
        super(LPIPSLoss3D, self).__init__()
        self.device = device
        self.loss_fn = lpips.LPIPS(net=net_type, version=version)
        self.loss_fn.to(self.device)
        self.axes = axes

    def forward(self, img_ref, img_pred):

        # Compute LPIPS loss for 3D images
        B, C, D, H, W = img_ref.shape

        # Tile to 3 channels, since LPIPS expects RGB images
        img1 = torch.tile(img_ref, dims=(1, 3, 1, 1, 1))
        img2 = torch.tile(img_pred, dims=(1, 3, 1, 1, 1))

        # Scale to [-1, 1]
        img1 = img1 * 2.0 - 1.0
        img2 = img2 * 2.0 - 1.0

        loss = 0.0
        #loss = torch.zeros(1).to(self.device)

        if 0 in self.axes:
            # Along D axis
            loss = loss + self.loss_fn.forward(
                img1.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, H, W),
                img2.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, H, W)
            ).mean()

        if 1 in self.axes:
            # Along H axis
            loss = loss + self.loss_fn.forward(
                img1.permute(0, 3, 2, 1, 4).contiguous().view(-1, C, D, W),
                img2.permute(0, 3, 2, 1, 4).contiguous().view(-1, C, D, W)
            ).mean()

        if 2 in self.axes:
            # Along W axis
            loss = loss + self.loss_fn.forward(
                img1.permute(0, 4, 2, 3, 1).contiguous().view(-1, C, D, H),
                img2.permute(0, 4, 2, 3, 1).contiguous().view(-1, C, D, H)
            ).mean()

        return loss


class FSCLoss3D(torch.nn.Module):
    def __init__(self, size, delta=1, alpha=None, drop_DC=False, device="cuda"):
        super(FSCLoss3D, self).__init__()
        self.device = device
        self.size = size
        self.drop_DC = drop_DC
        self.alpha = alpha
        self.shells, self.freq = get_shell_masks_3d(size=(size, size, size), delta=delta, device=device)

    def forward(self, img_ref, img_pred):

        fsc = fourier_shell_correlation(img_ref, img_pred, self.shells, self.freq, self.drop_DC, self.alpha)
        loss = 1 - (fsc ** 2).mean()  # squaring FSC -> phase agnostic
        return loss


class FSCLoss3DF(torch.nn.Module):
    def __init__(self, delta=1, alpha=None, drop_DC=False, device="cuda"):
        super(FSCLoss3DF, self).__init__()
        self.device = device
        self.delta = delta
        self.alpha = alpha
        self.drop_DC = drop_DC

    def forward(self, img_ref, img_pred):
        shells, freq = get_shell_masks_3d(size=img_ref.shape[2:], delta=self.delta, device=self.device)
        fsc = fourier_shell_correlation(img_ref, img_pred, shells, freq, self.drop_DC, self.alpha)
        loss = 1 - (fsc ** 2).mean()  # squaring FSC -> phase agnostic
        return loss

class CSCLoss(nn.Module):

    def __init__(self, model_id, eval_mode=True, verbose=True, feat_dist_func='L1', compare_input=True, device='cuda', **kwargs):

        """
        Cross-Scale Consistency Loss (CSCLoss)
        :param model_id: ID of the pre-trained model to use for feature extraction (should be a model trained on the same data/task)
        :param eval_mode: Whether to set the model to eval mode (recommended for feature extraction)
        :param verbose: Whether to print verbose information about the setup
        :param feat_dist_func: Distance function to use for comparing intermediate features (options: 'L1', 'L2', 'FSC')
        :param compare_input: Whether to also compare the input images (in addition to intermediate features and final output)
        :param device: Device to run the loss computation on (e.g., 'cuda' or 'cpu')
        :param kwargs: Additional keyword arguments for specific feature distance functions (e.g., size_hr for FSC loss)
        """

        super(CSCLoss, self).__init__()

        if feat_dist_func == "L2":
            self.feat_dist_func = nn.MSELoss()
        elif feat_dist_func == "L1":
            self.feat_dist_func = nn.L1Loss()
        elif feat_dist_func == "FSC":
            self.feat_dist_func = FSCLoss3DF(delta=1,
                                             alpha=2.0,
                                             drop_DC=False,
                                             device=device)  # more flexible

        self.output_dist_func = nn.L1Loss()  # L1 loss for final output distance

        self.compare_input = compare_input

        opt_path = load_options_from_experiment_id(model_id, root_dir="", file_type="yaml")
        opt_csc = OmegaConf.load(opt_path)
        opt_csc['dist'] = False  # ensure distributed is False for loss computation

        from models.select_model import define_Model
        self.net = define_Model(opt_csc, mode='test')
        self.net.load(model_id, mode='test')  # load model
        self.model = self.net.get_bare_model(self.net.netG)

        self.L = len(self.model.output_names)  # number of layers to compare

        if (eval_mode):
            self.eval()
            self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        if (verbose):
            print(f'Setting up CSC loss with features distance function: {feat_dist_func}')
            print(f'Using Degradation architecture {self.net.__class__.__name__} with ID: {model_id}')

    def normalize_tensor(self, in_feat, eps=1e-10):
        norm_factor = torch.sqrt(torch.sum(in_feat ** 2, dim=1, keepdim=True))
        return in_feat / (norm_factor + eps)

    def forward(self, in0, in1, normalize=False):
        if normalize:  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1

        outs0, outs1 = self.model(in0, ret_features=True), self.model(in1, ret_features=True)
        feats0, feats1, dists = {}, {}, {}

        # Compute distance of intermediate features + final output
        for kk in range(self.L - 1):
            # feats0[kk], feats1[kk] = self.normalize_tensor(outs0[kk]), self.normalize_tensor(outs1[kk])
            # feats0[kk], feats1[kk] = outs0[kk], outs1[kk]
            # dists[kk] = self.feat_dist_func(feats0[kk], feats1[kk])
            dists[kk] = self.feat_dist_func(outs0[kk], outs1[kk])

        # Compute distance of final output
        dists[self.L - 1] = self.output_dist_func(outs0[self.L - 1], outs1[self.L - 1])

        if self.compare_input:
            # Compute distance of input images (optional)
            dists[self.L] = self.feat_dist_func(in0, in1)

        loss = 0
        for l in range(len(dists)):
            loss += dists[l]

        return loss



if __name__ == "__main__":

    print("Done")

