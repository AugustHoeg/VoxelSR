import torch
import torch.nn as nn
from torch.cuda.amp import custom_fwd, custom_bwd
from torchvision.models import vgg19
import torch.nn.functional as F

import config

from kornia.filters import spatial_gradient3d

from utils.utils_3D_image import laplacian3D, gdx, gdy, gdz
#from utils import laplacian3Dlaplacian3D

from structure_tensor_utils.structure_tensor_functions import compute_S_matrix_V3, log_euclidean_metric_torch, get_gaussian_kernel, normalize_S, volumetric_gaussian_blur_V2
#from structure_tensor.structure_tensor_utils import compute_S_matrix_V3, log_euclidean_metric_torch, get_gaussian_kernel, normalize_S, volumetric_gaussian_blur_V2
import torch.cuda.amp


def compute_discriminator_loss(prop_real, prop_fake):
    dis_loss_fake = F.binary_cross_entropy_with_logits(prop_fake, torch.zeros_like(prop_fake))
    dis_loss_real = F.binary_cross_entropy_with_logits(prop_real, torch.ones_like(prop_real) - 0.1 * torch.ones_like(prop_real))
    # Formulate Discriminator loss: Max log(D(I_HR)) + log(1 - D(G(I_LR)))
    #dis_loss_fake = loss_fn_dict["BCE_Logistic"](prop_fake, torch.zeros_like(prop_fake))
    #dis_loss_real = loss_fn_dict["BCE_Logistic"](prop_real, torch.ones_like(prop_real) - 0.1 * torch.ones_like(prop_real))  # Added one-side label smoothing
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

# Old generator loss function
# def compute_generator_loss(real_hi_res=None, fake_hi_res=None, loss_fn_dict=None, loss_val_dict=None, prop_fake=None, device="cpu"):
#
#     # Formulate Generator loss: Min log(I_HR - G(I_LR)) <-> Max log(D(G(I_LR)))
#     #adv_loss = config.LOSS_WEIGHTS["ADV"] * loss_fn_dict["BCE_Logistic_Loss"](prop_fake, torch.ones_like(prop_fake))
#
#     aux_loss = torch.tensor(0.0).to(device)
#     #print(type(aux_loss))
#     gen_loss = torch.tensor(0.0).to(device)
#     #print(type(gen_loss))
#     for key, value in loss_val_dict.items():
#         if value > 0:
#             if key == "ADV":  # Use only adversarial loss if model is a GAN and ADV > 0
#                 if prop_fake is not None:
#                     aux_loss = value * loss_fn_dict["BCE_Logistic"](prop_fake,  torch.ones_like(prop_fake))
#             else:
#                 aux_loss = value*loss_fn_dict[key](real_hi_res, fake_hi_res)
#             #wandb.log({key: aux_loss.item() / n_samples})
#             gen_loss += aux_loss
#
#     return gen_loss



def compute_generator_loss(real_hi_res=None, fake_hi_res=None, loss_fn_dict=None, loss_val_dict=None, prop_real=None, prop_fake=None, model="plain", device="cuda"):

    # Formulate Generator loss: Min log(I_HR - G(I_LR)) <-> Max log(D(G(I_LR)))
    #adv_loss = config.LOSS_WEIGHTS["ADV"] * loss_fn_dict["BCE_Logistic_Loss"](prop_fake, torch.ones_like(prop_fake))

    gen_loss = torch.tensor(0.0).to(device)
    aux_loss = torch.tensor(0.0).to(device)

    if model == "wgan-gp":
        if prop_fake is not None:
            gen_loss += -(5e-3 * torch.mean(prop_fake.reshape(-1)))
        #loss_val_dict['ADV'] = 0  # Remove ADV if model is wgan-gp

    elif model == "ragan":
        if prop_fake is not None:
            gen_loss += loss_val_dict['ADV'] * (
                    loss_fn_dict["BCE_Logistic"](prop_real - torch.mean(prop_fake, dim=0), torch.zeros_like(prop_real)) +
                    loss_fn_dict["BCE_Logistic"](prop_fake - torch.mean(prop_real, dim=0), torch.ones_like(prop_fake))) / 2
        #loss_val_dict['ADV'] = 0

    elif model == "gan":
        if prop_fake is not None:
            gen_loss += loss_val_dict['ADV'] * loss_fn_dict["BCE_Logistic"](prop_fake, torch.ones_like(prop_fake))
        #loss_val_dict['ADV'] = 0
    else:
        loss_val_dict['ADV'] = 0  # Remove ADV if model is not gan

    #aux_loss = torch.tensor(0.0).to(device)
    for key, value in loss_val_dict.items():
        if value > 0 and key != 'ADV':
            aux_loss += value*loss_fn_dict[key](real_hi_res, fake_hi_res)
            #wandb.log({key: aux_loss.item() / n_samples})

    gen_loss += aux_loss

    return gen_loss


def construct_S_from_dV(dV, rho):
    if dV.dtype == torch.float64:
        rho = torch.tensor(rho).double()
    elif dV.dtype == torch.float32:
        rho = torch.tensor(rho).float()
    k = get_gaussian_kernel(rho, also_dg=False, radius=None)

    if dV.device.type == "cuda":
        k = k.to(config.DEVICE)

    dVx = volumetric_gaussian_blur_V2(dV[:, 0].unsqueeze(0), k, separate=False)
    dVy = volumetric_gaussian_blur_V2(dV[:, 1].unsqueeze(0), k, separate=False)
    dVz = volumetric_gaussian_blur_V2(dV[:, 2].unsqueeze(0), k, separate=False)

    dV[:, 0] = dVx
    dV[:, 1] = dVy
    dV[:, 2] = dVz

    Sxx = dV[:, 0] * dV[:, 0]
    Syy = dV[:, 1] * dV[:, 1]
    Szz = dV[:, 2] * dV[:, 2]
    Sxy = dV[:, 0] * dV[:, 1]
    Sxz = dV[:, 0] * dV[:, 2]
    Syz = dV[:, 1] * dV[:, 2]

    S = torch.cat((Sxx, Syy, Szz, Sxy, Sxz, Syz), dim=0)
    return S


class StructureLoss3D(nn.Module):
    def __init__(self, sigma, rho, use_kornia=False):
        super().__init__()

        self.sigma = sigma
        self.rho = rho

        # Use MSE loss as error metric for structure tensors
        self.loss_func = nn.MSELoss()
        #self.loss_val = 0

        self.use_kornia = use_kornia

    def forward(self, real_hi_res, fake_hi_res):
        loss = torch.tensor([0]).to(config.DEVICE)
        for real_patch, fake_patch in zip(real_hi_res, fake_hi_res):
        #for i in range(real_hi_res.shape[0]):
            #real_patch = real_hi_res[i, 0, :, :, :].double()
            #fake_patch = fake_hi_res[i, 0, :, :, :].double() # Ensure that the fake patch is float32
            #S_real_ele = compute_S_matrix_V2(real_patch.squeeze().float(), 0.5, 2.0, False, 'valid')
            #S_fake_ele = compute_S_matrix_V2(fake_patch.squeeze().float(), 0.5, 2.0, False, 'valid')

            if self.use_kornia:
                dV_real = spatial_gradient3d(real_patch.unsqueeze(0).float(), mode='diff', order=1)
                dV_fake = spatial_gradient3d(fake_patch.unsqueeze(0).float(), mode='diff', order=1)

                S_real_ele = construct_S_from_dV(dV_real[0], self.rho)
                S_fake_ele = construct_S_from_dV(dV_fake[0], self.rho)
            else:
                S_real_ele = compute_S_matrix_V3(real_patch.double(), sigma=self.sigma, rho=self.rho, sep=True, padding="valid")
                S_fake_ele = compute_S_matrix_V3(fake_patch.double(), sigma=self.sigma, rho=self.rho, sep=True, padding="valid")
            #print("Number of nans in S_real_ele", torch.sum(torch.isnan(S_real_ele)))
            #print("Number of nans in S_fake_ele", torch.sum(torch.isnan(S_fake_ele)))

            #S_real_ele = normalize_S(S_real_ele, tensor=True)
            #S_fake_ele = normalize_S(S_fake_ele, tensor=True)

            dist, dist_points = log_euclidean_metric_torch(S_real_ele, S_fake_ele)
            loss = loss + dist

        #b, c, h, w, d = fake_hi_res.size()
        #loss = loss / (b*c*h*w*d)
        return loss


class TextureLoss3D(nn.Module):
    def __init__(self, layer_idx=35, device="cpu"):
        super().__init__()
        self.layer_5_4 = layer_idx
        self.device = device

        self.vgg_model = vgg19(pretrained=True).features[:self.layer_5_4].eval().to(device)
        self.loss_func = nn.MSELoss()

        for param in self.vgg_model.parameters():
            param.requires_grad = False

    def forward(self, real_hi_res, fake_hi_res):
        #loss = torch.zeros(1).to(DCSRN_config.DEVICE)
        for i in range(real_hi_res.shape[0]):
            real_patch = real_hi_res[i, :, :, :, :]
            fake_patch = fake_hi_res[i, :, :, :, :]
            for _ in range(3):
                # Since VGG accepts only RGB images, patches are tiled to (BATCH_SIZE, 3, H, W, D)
                phi_real = self.vgg_model(torch.tile(real_patch, (3, 1, 1, 1)).permute(1, 0, 2, 3))
                phi_fake = self.vgg_model(torch.tile(fake_patch, (3, 1, 1, 1)).permute(1, 0, 2, 3))
                real_patch = real_patch.permute(0, 3, 1, 2)
                fake_patch = fake_patch.permute(0, 3, 1, 2)

                # For the real and fake 3D patch do:
                # pass 3D patch through VGG network in coronal direction
                # rotate 3D patch and pass through axial direction
                # rotate 3D patch and pass through sagittal direction
                # Subtract the feature vectors of real and fake using MSELoss
                # Finally, sum over features losses for each direction.

        # Perhaps we need 1 more permute here to go back to the same patch orientation.
        return 0

class TotalVariationLoss3D(nn.Module):
    def __init__(self, mode):
        super().__init__()
        #self.vgg_model = vgg19(pretrained=True).features[:self.layer_5_4].eval().to(SRGAN_config.DEVICE)
        #self.loss_func = nn.MSELoss()
        self.mode = mode

        #for param in self.vgg_model.parameters():
        #    param.requires_grad = False

    def forward(self, fake_hi_res):
        #features_real = self.vgg_model(real_hi_res)
        #features_fake = self.vgg_model(fake_hi_res)
        b, c, h, w, d = fake_hi_res.size()
        if self.mode == "sum_of_squares":
            tv_x = torch.pow(fake_hi_res[:, :, 1:, :, :] - fake_hi_res[:, :, :-1, :, :], 2).sum()
            tv_y = torch.pow(fake_hi_res[:, :, :, 1:, :] - fake_hi_res[:, :, :, :-1, :], 2).sum()
            tv_z = torch.pow(fake_hi_res[:, :, :, :, 1:] - fake_hi_res[:, :, :, :, :-1], 2).sum()
            return (tv_x + tv_y + tv_z)/(b*c*h*w*d)
        elif self.mode == "L2":
            tv_x = torch.pow(fake_hi_res[:, :, 1:, :, :] - fake_hi_res[:, :, :-1, :, :], 2)
            tv_y = torch.pow(fake_hi_res[:, :, :, 1:, :] - fake_hi_res[:, :, :, :-1, :], 2)
            tv_z = torch.pow(fake_hi_res[:, :, :, :, 1:] - fake_hi_res[:, :, :, :, :-1], 2)
            return torch.sum(torch.sqrt(tv_x.flatten() + tv_y.flatten() + tv_z.flatten()))/(b*c*h*w*d)
        #diff_x = self.loss_func(fake_hi_res[:, :, 1:, :, :], fake_hi_res[:, :, :-1, :, :])
        #diff_y = self.loss_func(fake_hi_res[:, :, :, 1:, :], fake_hi_res[:, :, :, :-1, :])
        #diff_z = self.loss_func(fake_hi_res[:, :, :, :, 1:], fake_hi_res[:, :, :, :, :-1])

        #return torch.sqrt_(diff_x + diff_y + diff_z)

class VGGLoss(nn.Module):
    def __init__(self, layer_idx=35, device="cpu"):
        super().__init__()
        self.layer_5_4 = layer_idx
        #self.vgg_model = vgg19(weights="DEFAULT").features[:self.layer_5_4].eval().to(SRGAN_config.DEVICE)
        self.vgg_model = vgg19(weights="DEFAULT").features[:self.layer_5_4].eval().to(device)
        self.loss = nn.MSELoss()

        for param in self.vgg_model.parameters():
            param.requires_grad = False

    def forward(self, real_hi_res, fake_hi_res):
        features_real = self.vgg_model(real_hi_res)
        features_fake = self.vgg_model(fake_hi_res)
        return self.loss(features_real, features_fake)

class VGGLoss3D(nn.Module):
    def __init__(self, num_parts=4, layer_idx=36, loss_func=nn.MSELoss(), device="cpu"):
        super().__init__()
        self.num_parts = num_parts
        self.layer_5_4 = layer_idx
        self.device = device
        #vgg_model_grayscale = nn.Sequential(nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #                                    *vgg19(pretrained=True).features[1:self.layer_5_4]).eval().to(SRGAN_config.DEVICE)
        ##self.first_vgg_layer = nn.Sequential(nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        ##vgg_module_list = nn.ModuleList(vgg19(pretrained=True).features[:self.layer_5_4])

        ##self.first_vgg_layer.add_module("hej", nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        ##self.vgg_model_gray = self.first_vgg_layer.add_module(vgg19(pretrained=True).features[:self.layer_5_4])

        #self.vgg_model = vgg19(pretrained=True).features[:self.layer_5_4].eval().to(DCSRN_config.DEVICE)
        #self.vgg_model = vgg19('DEFAULT').features[:self.layer_5_4].eval().to(DCSRN_config.DEVICE)
        self.vgg_model = vgg19('DEFAULT').features[:self.layer_5_4].eval().to(device)
        self.loss_func = loss_func

        for param in self.vgg_model.parameters():
            param.requires_grad = False

    def forward(self, real_hi_res, fake_hi_res):

        img_size = real_hi_res.size(2)
        vgg_batch_size = img_size // self.num_parts

        loss = torch.zeros(1).to(self.device)
        for i in range(real_hi_res.shape[0]):
            real_patch = real_hi_res[i, :, :, :, :]
            fake_patch = fake_hi_res[i, :, :, :, :]
            for _ in range(3):
                for j in range(self.num_parts):
                    start_idx = j * vgg_batch_size
                    end_idx = (j + 1) * vgg_batch_size if j != (self.num_parts - 1) else img_size  # Ensure the last part includes all remaining elements
                    # Since VGG accepts only RGB images, patches are tiled to (BATCH_SIZE, 3, H, W, D)
                    batch1 = torch.tile(real_patch[:, start_idx:end_idx, :, :], (3, 1, 1, 1)).permute(1, 0, 2, 3)
                    batch2 = torch.tile(fake_patch[:, start_idx:end_idx, :, :], (3, 1, 1, 1)).permute(1, 0, 2, 3)
                    loss = loss + self.loss_func(self.vgg_model(batch1), self.vgg_model(batch2))
                    #loss = loss + self.loss_func(self.vgg_model(torch.tile(real_patch, (3, 1, 1, 1)).permute(1, 0, 2, 3)),
                    #                             self.vgg_model(torch.tile(fake_patch, (3, 1, 1, 1)).permute(1, 0, 2, 3)))

                real_patch = real_patch.permute(0, 3, 1, 2)
                fake_patch = fake_patch.permute(0, 3, 1, 2)

                # For the real and fake 3D patch do:
                # pass 3D patch through VGG network in coronal direction
                # rotate 3D patch and pass through axial direction
                # rotate 3D patch and pass through sagittal direction
                # Subtract the feature vectors of real and fake using MSELoss
                # Finally, sum over features losses for each direction.

        # Perhaps we need 1 more permute here to go back to the same patch orientation.
        return loss.squeeze()


class GradientLoss3D(nn.Module):
    def __init__(self, kernel, order, loss_func, sigma=None):
        super().__init__()
        self.kernel = kernel
        self.order = order
        self.loss_func = loss_func
        self.sigma = sigma

        #for param in self.vgg_model.parameters():
        #    param.requires_grad = False

    def forward(self, real_hi_res, fake_hi_res):

        if self.sigma is None:
            return self.loss_func(spatial_gradient3d(real_hi_res, mode=self.kernel, order=self.order),
                                  spatial_gradient3d(fake_hi_res, mode=self.kernel, order=self.order))
        elif self.sigma > 0:
            g, gd, gdd = get_gaussian_kernel(self.sigma, also_ddg=True, radius=None)
            if (real_hi_res.device.type == "cuda") or (fake_hi_res.device.type == "cuda"):
                g = g.to(config.DEVICE)
                gd = gd.to(config.DEVICE)
                gdd = gdd.to(config.DEVICE)

            loss = torch.zeros(1).to(config.DEVICE)
            dev = gd
            for i in range(self.order):
                if i == 1:
                    dev = gdd
                for real_patch, fake_patch in zip(real_hi_res, fake_hi_res):
                    Ix_real = gdx(real_patch.unsqueeze(0), g, dev, prepend_one=False, padding='valid')
                    Ix_fake = gdx(fake_patch.unsqueeze(0), g, dev, prepend_one=False, padding='valid')
                    loss = loss + self.loss_func(Ix_real, Ix_fake)

                    Iy_real = gdy(real_patch.unsqueeze(0), g, dev, prepend_one=False, padding='valid')
                    Iy_fake = gdy(fake_patch.unsqueeze(0), g, dev, prepend_one=False, padding='valid')
                    loss = loss + self.loss_func(Iy_real, Iy_fake)

                    Iz_real = gdz(real_patch.unsqueeze(0), g, dev, prepend_one=False, padding='valid')
                    Iz_fake = gdz(fake_patch.unsqueeze(0), g, dev, prepend_one=False, padding='valid')
                    loss = loss + self.loss_func(Iz_real, Iz_fake)

            return loss[0]

        else:
            print("Gradient loss Error!")
            return 0
        #features_real = self.vgg_model(real_hi_res)
        #features_fake = self.vgg_model(fake_hi_res)
        #return self.loss(features_real, features_fake)


class LaplacianLoss3D(nn.Module):
    def __init__(self, sigma, padding, loss_func):
        super().__init__()
        self.sigma = sigma
        self.padding = padding
        self.loss_func = loss_func

        # for param in self.vgg_model.parameters():
        #    param.requires_grad = False

    def forward(self, real_hi_res, fake_hi_res):

        g, _, gdd = get_gaussian_kernel(self.sigma, also_ddg=True, radius=None)

        if (real_hi_res.device.type == "cuda") or (fake_hi_res.device.type == "cuda"):
            g = g.to(config.DEVICE)
            gdd = gdd.to(config.DEVICE)

        loss = torch.tensor([0]).to(config.DEVICE)
        for real_patch, fake_patch in zip(real_hi_res, fake_hi_res):

            L_real = laplacian3D(real_patch, g, gdd, prepend_one=False, padding=self.padding)
            L_fake = laplacian3D(fake_patch, g, gdd, prepend_one=False, padding=self.padding)

            loss = loss + self.loss_func(L_real, L_fake)

        return loss

        # features_real = self.vgg_model(real_hi_res)
        # features_fake = self.vgg_model(fake_hi_res)
        # return self.loss(features_real, features_fake)


def bce_loss(y_real, y_pred):
    """
    Simple binary cross entropy loss
    :param y_real: target value (label)
    :param y_pred: predicted value
    :return: loss
    """
    return torch.mean(y_pred - y_real*y_pred + torch.log(1 + torch.exp(-y_pred)))

if __name__ == "__main__":


    vgg = vgg19(pretrained=True)

    print("VGG19 Features", vgg.features)

    layer_5_4 = 36 #  4th convolution (after activation) before 5th max pooling layer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgg_model = vgg.features[:layer_5_4].eval().to(device)  # Eval() ensures we dont update the weights

    for param in vgg_model.parameters():
        param.requires_grad = False


    print("Done")

