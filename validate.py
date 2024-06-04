import os
import numpy as np
import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
import time
from piq.ssim import ssim as calculate_ssim
from torch import no_grad
from models.EBokehNet.ln_modules import DeBokehLn, getEBokehNet
from models.SBTNet.model import SBTNet
from lpips import LPIPS
from dataset import BokehDataset
from metrics import calculate_lpips, calculate_psnr, calculate_ssim
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torch.nn.functional as F
import torchvision.transforms as transforms
# 确定训练参数袭击, 确定对齐每层特征大小
# 改进transformer
# 测试训练
# 搭建可视化eval, 随机选取3张图片进行效果
to_pil = ToPILImage()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
datapath=r"./data/test"
test_dataset = BokehDataset(datapath, transform=None, test=True)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 3, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x

def eval(model, outputpath=r"./output/picture/", show_n_groups=4, load_weights=False, pth_path=None):
    assert model != None
    assert outputpath != None
    to_pil = transforms.ToPILImage()
    if load_weights and pth_path is not None:
        model.load_state_dict(torch.load('pth_path'))
    model.to(device).eval()
    dataloader = DataLoader(test_dataset, batch_size=show_n_groups, shuffle=True)
    criterion = nn.MSELoss()

    log_file = os.path.join(outputpath, r"logging/eval_log.txt")
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    with open(log_file, 'w') as f:
        f.write('Loss\tDuration\tPSNR\tSSIM\tLPIPS\n')
    start_time = time.time()

    fig, axes = plt.subplots(show_n_groups, 3, figsize=(15, 5 * show_n_groups))
    fig.suptitle('Model Outputs')

    val_loss_list = []
    PSNRs, SSIMs, LPIPSs = [], [], []
    val_start_time = time.time()

    for idx, batch in enumerate(dataloader):
        target = batch["target"].to(device)
        source = batch["source"].to(device)
        with no_grad():
            output = model(source)
            loss = criterion(output, target)
            val_loss_list.append(loss.item())
            PSNRs.append(calculate_psnr(target, output))
            SSIMs.append(calculate_ssim(target, output))
            LPIPSs.append(calculate_lpips(target[0], output[0]))

        for j, (t, s, o) in enumerate(zip(target, source, output)):
            image_source_path = os.path.join(outputpath, f"image_{j}_source.png")
            image_target_path = os.path.join(outputpath, f"image_{j}_target.png")
            image_output_path = os.path.join(outputpath, f"image_{j}_output.png")

            s = to_pil(s)
            s.save(image_source_path)
            t = to_pil(t)
            t.save(image_target_path)
            o = to_pil(o)
            o.save(image_output_path)

            axes[j,0].imshow(t)
            axes[j,0].axis('off')
            axes[j,0].set_title(f"image_{j}_target.png")

            axes[j, 1].imshow(o)
            axes[j, 1].axis('off')
            axes[j, 1].set_title(f"image_{j}_output.png")

            axes[j, 2].imshow(s)
            axes[j, 2].axis('off')
            axes[j, 2].set_title(f"image_{j}_source.png")

        break


    val_duration = time.time() - val_start_time
    val_loss = sum(val_loss_list)/len(val_loss_list)

    #Val Log
    with open(log_file, 'a') as f:
        f.write("Val Log:" + f'{val_loss}\t{val_duration}\t{sum(PSNRs) / len(PSNRs)}\t{sum(SSIMs) / len(SSIMs)}\t '
            f'{sum(LPIPSs) / len(LPIPSs)}\n')

    print(f"Val loss: {val_loss:0.03f}, used time: {val_duration}.")
    print(f"Average PSNR: {sum(PSNRs) / len(PSNRs)}")
    print(f"Average SSIM: {sum(SSIMs) / len(SSIMs)}")
    print(f"Average LPIPS: {sum(LPIPSs) / len(LPIPSs)}")
    total_duration = time.time() - start_time
    print(f'Total duration: {total_duration}\n')
    plt.show()

sbtnet = SBTNet()
sbtnet_path = r"./tract/SBTNet/eval/"
#eval(sbtnet, sbtnet_path)
ebokehnet = getEBokehNet()
ebokehnet_path = r"./tract/EBokehNet/eval/"
eval(ebokehnet, ebokehnet_path)

