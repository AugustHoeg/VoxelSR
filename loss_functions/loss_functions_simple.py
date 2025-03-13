import torch
import torch.nn.functional as F
import torch.cuda.amp


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

if __name__ == "__main__":

    print("Done")

