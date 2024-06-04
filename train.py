import os
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
import time
from piq.ssim import ssim as calculate_ssim
from torch import no_grad
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda import synchronize
import tqdm
from models.EBokehNet.utils.eval_utils import save_tensor_img, setup_timings, setup_metrics, sanity_checks
from models.EBokehNet.ln_modules import   getEBokehNet
from models.SBTNet.model import SBTNet
from lpips import LPIPS
from dataset import BokehDataset
from metrics import calculate_lpips, calculate_psnr, calculate_ssim
from torchsummary import summary
from torchvision import transforms
# 确定训练参数袭击, 确定对齐每层特征大小
# 改进transformer
# 测试训练
# 搭建可视化eval, 随机选取3张图片进行效果


to_pil = ToPILImage()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
datapath=r"./data/test"



#test_dataset = BokehDataset(datapath, transform=None, test=True)
def adjust_learning_rate(optimizer, epoch, warmup_epochs, base_lr):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def train(model, outputpath, n_epoch=1, warmup_epochs=-10, transforms=None,
          learning_rate=1e-4, use_scheduler=False, device="cpu", load_weights=False, pth_path=None):
    assert model != None
    assert outputpath != None

    if load_weights and pth_path is not None:
        model.load_state_dict(torch.load('pth_path'))
    model.to(device)

    train_amount = 100
    train_dataset = BokehDataset(datapath, transform=transforms, train=True, samples_train=train_amount)
    val_dataset = BokehDataset(datapath, transform=transforms, validation=True, samples_train=train_amount)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    data_amount = len(train_dataloader)

    best_loss = float('inf')
    log_file = os.path.join(outputpath, r"logging/training_log.txt")
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    with open(log_file, 'w') as f:
        f.write('Epoch\tLoss\tDuration\tPSNR\tSSIM\tLPIPS\n')
    weights_path = os.path.join(outputpath, r"weights/")
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)

    learning_rate = learning_rate
    optimizer = Adam(model.parameters(), lr=learning_rate)
    if use_scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epoch, eta_min=5e-5)
    criterion = nn.MSELoss()
    start_time = time.time()
    train_loss_list = []
    n_epoch = n_epoch
    for epoch in range(n_epoch):
        if epoch < warmup_epochs:
            adjust_learning_rate(optimizer, epoch, warmup_epochs, learning_rate)
        model.train()
        epoch_loss = 0.0
        epoch_start_time = time.time()
        for idx, batch in enumerate(train_dataloader):
            target = batch["target"].to(device)
            optimizer.zero_grad()
            output, _ = model(batch)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_duration = time.time() - epoch_start_time
        epoch_loss = epoch_loss / data_amount
        train_loss_list.append(epoch_loss)

        if (epoch+1) % 1 == 0:
            model.eval()
            val_loss_list = []
            PSNRs, SSIMs, LPIPSs = [], [], []
            val_start_time = time.time()
            for idx, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
                target = batch["target"].to(device)
                with no_grad():
                    output = model(batch)
                    loss = criterion(output, target)
                    val_loss_list.append(loss.item())
                    PSNRs.append(calculate_psnr(target, output))
                    SSIMs.append(calculate_ssim(target, output))
                    LPIPSs.append(calculate_lpips(target[0], output[0]))
            val_duration = time.time() - val_start_time
            val_loss = sum(val_loss_list)/len(val_loss_list)
            # Train Log
            # with open(log_file, 'a') as f:
            #     f.write("Train Log:"+
            #         f'{epoch}\t{epoch_loss}\t{epoch_duration}\t{sum(PSNRs) / len(PSNRs)}\t{sum(SSIMs) / len(SSIMs)}\t '
            #         f'{sum(LPIPSs) / len(LPIPSs)}\n')
            # Val Log
            with open(log_file, 'a') as f:
                f.write("Val Log:"+
                    f'{epoch}\t{val_loss}\t{epoch_duration}\t{sum(PSNRs) / len(PSNRs)}\t{sum(SSIMs) / len(SSIMs)}\t '
                    f'{sum(LPIPSs) / len(LPIPSs)}\n')

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), os.path.join( weights_path,r'best_model.pth'))

            print(f"Epoch {epoch} completed, val loss: {val_loss:0.03f}, used time: {val_duration}.")
            print(f"Average PSNR: {sum(PSNRs) / len(PSNRs)}")
            print(f"Average SSIM: {sum(SSIMs) / len(SSIMs)}")
            print(f"Average LPIPS: {sum(LPIPSs) / len(LPIPSs)}")
        if use_scheduler:
            if epoch >= warmup_epochs:
                scheduler.step()
    total_duration = time.time() - start_time
    total_train_loss = sum(train_loss_list) / len(train_loss_list)

    with open(log_file, 'a') as f:
        f.write(f'Total_train_loss: {total_train_loss}\tTotal duration: {total_duration}\n')

    torch.save(model.state_dict(), os.path.join( weights_path, r'final_model.pth'))



fake_batch ={'source': torch.randn((1,3,1440, 1920)),
             'target': torch.randn((1,3,1440, 1920)),
             'alpha': torch.randn((1,3,1440, 1920)),
             'src_lens': ['Sony50mmf16.0BS'],
             'tgt_lens': ['Sony50mmf1.8BS'],
             'disparity': torch.tensor([0.4500]),
             'src_lens_type': torch.tensor([1.]),
             'tgt_lens_type': torch.tensor([1.]),
             'src_F': torch.tensor([16.]),
             'tgt_F': torch.tensor([1.8000]),
             'image_id': [('00027',)],
             'resolution': [[torch.tensor([3]), torch.tensor([1440]), torch.tensor([1920])]]}


def get_params_amount(model, input_list):
    # for idx, (name, module) in enumerate(model.named_modules()):
    #     summary(module, input_list[idx])
    total_param_amount = 0.0
    for name, param in model.named_parameters():
        total_param_amount += param.size()

data_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # transforms.RandomHorizontalFlip(p=0.3),
    # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
])

sbtnet = SBTNet()
sbtnet_path = r"./tract/SBTNet/"
train(model=sbtnet, outputpath=sbtnet_path,transforms=None)

# ebokehnet = getEBokehNet()
# summary(ebokehnet, (3,1440, 1920))
# ebokehnet_path = r"./tract/EBokehNet/"
# #train(model=ebokehnet,outputpath=ebokehnet_path, transforms=data_transform )


