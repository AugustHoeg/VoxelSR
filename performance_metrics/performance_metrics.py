
import kornia as korn
import torch
import torch.nn as nn
import torchio as tio
from monai.metrics.regression import SSIMMetric, PSNRMetric, RMSEMetric

def calculate_metric(img_H, img_E, border=0, metric_fn=None):
    metric = 0
    metric_slice_list = []

    if len(img_H.shape) > 4:  # 3D volume
        B, C, H, W, D = img_H.shape
        for slice_idx in range(D):
            H_slice = img_H[..., slice_idx].float().squeeze().clamp(min=0.0, max=1.0)
            E_slice = img_E[..., slice_idx].float().squeeze().clamp(min=0.0, max=1.0)
            metric_slice_val = metric_fn(E_slice.cpu().numpy(), H_slice.cpu().numpy(), border=border)
            metric_slice_list.append(metric_slice_val)
            metric += metric_slice_val
        metric /= D  # Average over slices

    else:  # 2D image
        H_slice = img_H.float().squeeze().clamp(min=0.0, max=1.0)
        E_slice = img_E.float().squeeze().clamp(min=0.0, max=1.0)
        metric = metric_fn(E_slice.cpu().numpy(), H_slice.cpu().numpy(), border=border)

    return metric


def compute_performance_metrics(real_hi_res, fake_hi_res, metric_fn_dict, metric_val_dict, rescale_images=False):

    num_patches = len(real_hi_res)

    # Rescale images if needed
    if rescale_images:
        rescale = tio.transforms.RescaleIntensity((0.0, 1.0))
        img1 = torch.zeros_like(real_hi_res)
        img2 = torch.zeros_like(fake_hi_res)
        for patch_idx in range(num_patches):
             img1[patch_idx] = rescale(real_hi_res[patch_idx].cpu())
             img2[patch_idx] = rescale(fake_hi_res[patch_idx].cpu())
    else:
        img1 = real_hi_res
        img2 = fake_hi_res

    # Calculate metrics for each patch
    for patch_hr, patch_lr in zip(img1, img2):
        for key in metric_fn_dict:
            metric_val_dict[key] += calculate_metric(patch_hr, patch_lr, border=0, metric_fn=metric_fn_dict[key]) / num_patches

    return metric_val_dict


class PSNR_3D(nn.Module):
    def __init__(self, border=1):
        super().__init__()

        self.border = border
        self.metric_func = PSNRMetric(max_val=1.0, reduction="mean", get_not_nans=False)

    def forward(self, img_true, img_false):

        result = torch.mean(self.metric_func(img_true.clamp(min=0.0, max=1.0), img_false.clamp(min=0.0, max=1.0)))  # mean over patches in batch

        return result.item()


class SSIM_3D(nn.Module):
    def __init__(self, border=1, dims=3, win_size=11):
        super().__init__()

        self.border = border
        self.metric_func = SSIMMetric(dims, data_range=1.0, kernel_type="gaussian",
                                      win_size=win_size, kernel_sigma=1.5, k1=0.01, k2=0.03,
                                      reduction="mean", get_not_nans=False)

    def forward(self, img_true, img_false):

        result = torch.mean(self.metric_func(img_true, img_false))  # mean over patches in batch
        return result.item()

class NRMSE_3D(nn.Module):
    def __init__(self, border=1, normalization='euclidean'):
        super().__init__()

        self.border = border
        self.normalization = normalization
        self.metric_func = RMSEMetric(reduction='mean', get_not_nans=False)


    def forward(self, img_true, img_false):

        if self.normalization == 'euclidean':
            denom = torch.sqrt(torch.mean((img_true * img_true), dim=[1,2,3,4]))
        elif self.normalization == 'min-max':
            denom = torch.max(img_true, dim=[1,2,3,4]) - torch.min(img_true, dim=[1,2,3,4])
        elif self.normalization == 'mean':
            denom = torch.mean(img_true, dim=[1,2,3,4])
        else:
            raise ValueError("Unsupported norm_type")

        result = torch.mean(self.metric_func(img_true, img_false)/denom)  # mean over patches in batch

        return result.item()

def performance_metrics(real_hi_res, fake_hi_res):
    mse_func = nn.MSELoss()
    rescale = tio.transforms.RescaleIntensity((0.0, 1.0))
    psnr_total = 0
    ssim_total = 0
    L = 1.0  # Maximum value after scaling images between 0.0 and 1.0
    for real_patch, fake_patch in zip(real_hi_res, fake_hi_res):
        real_patch_rescaled = rescale(real_patch.cpu()).unsqueeze(0)
        fake_patch_rescaled = rescale(fake_patch.cpu()).unsqueeze(0)
        mse = mse_func(real_patch_rescaled, fake_patch_rescaled)
        psnr = 10*torch.log10((L**2)/mse)
        psnr_total += psnr

        ssim = torch.mean(korn.metrics.ssim3d(fake_patch_rescaled, real_patch_rescaled, window_size=11, max_val=L))
        ssim_total += ssim

    return psnr_total, ssim_total

if __name__ == "__main__":

    print("Done")