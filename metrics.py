# NTIRE standard metrics
import lpips
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio
from torch import Tensor, no_grad, mean

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def calculate_lpips(img0: Tensor, img1: Tensor, modelName = 'alex'):
    lpips_alex = lpips.LPIPS(net=modelName).to(device)
    # NOTE: LPIPS expects image normalized to [-1, 1]
    img0 = 2 * img0 - 1.0
    img1 = 2 * img1 - 1.0

    with no_grad():
        distance = lpips_alex(img0, img1)

    # if max(distance.shape) > 1:
    #     return mean(distance).item()
    return distance.item()


def calculate_psnr(img0: Tensor, img1: Tensor):
    img0 = img0.detach().cpu().numpy()
    img1 = img1.detach().cpu().numpy()
    mse = np.mean((img0.astype(np.float32) - img1.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    max_val = np.max(img0)
    return 20 * np.log10(max_val / np.sqrt(mse))

# def calculate_psnr(img0: Tensor, img1: Tensor):
#     target_psnr = img0[0].detach().cpu().numpy().transpose(1, 2, 0) * 65535.
#     target_psnr = target_psnr.astype(np.uint16)
#
#     out_psnr = img1[0].detach().cpu().numpy().transpose(1, 2, 0) * 65535.
#     out_psnr = out_psnr.astype(np.uint16)
#     return peak_signal_noise_ratio(target_psnr, out_psnr)

def calculate_ssim(img0: Tensor, img1: Tensor):
    bs, c, h, w = img0.shape
    size = min(h,w)
    img0 = img0.detach().cpu().numpy()
    img1 = img1.detach().cpu().numpy()
    val_list = []
    for i,(p0, p1) in enumerate(zip(img0, img1)):
        p0 = p0.transpose(1, 2, 0)[:size,:size,]
        p1 = p1.transpose(1, 2, 0)[:size,:size,]
        max_ = max(p0.max(),p1.max())
        min_ = max(p0.min(), p1.min())
        val_list.append(ssim(p0, p1, channel_axis=2,data_range=max_ - min_))
    return sum(val_list)/(len(val_list)+1e-5)






