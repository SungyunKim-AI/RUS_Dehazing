# User warnings ignore
import warnings

from PIL.Image import preinit
warnings.filterwarnings("ignore")

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import argparse
import numpy as np
import random
from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from dpt.models import DPTDepthModel
from dpt.discriminator import Discriminator
from dpt.Air_UNet import Air_UNet

from dataset import NYU_Dataset, RESIDE_Dataset
from torch.utils.data import DataLoader

from Module_Airlight.Airlight_Module import get_Airlight
from Module_Metrics.metrics import get_ssim_batch, get_psnr_batch
from util import misc, save_log, utils
from discriminator_val import validation

import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.utils import make_grid



def get_args():
    parser = argparse.ArgumentParser()
    # dataset parameters
    parser.add_argument('--dataset', required=False, default='NYU',  help='dataset name')
    parser.add_argument('--dataRoot', type=str, default='',  help='data file path')
    parser.add_argument('--norm', type=bool, default=True,  help='Image Normalize flag')
    
    # learning parameters
    parser.add_argument('--seed', type=int, default=101, help='Random Seed')
    parser.add_argument('--batchSize', type=int, default=2, help='test dataloader input batch size')
    parser.add_argument('--imageSize_W', type=int, default=256, help='the width of the resized input image to network')
    parser.add_argument('--imageSize_H', type=int, default=256, help='the height of the resized input image to network')
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # model parameters
    parser.add_argument('--preTrainedModel', type=str, default='weights/dpt_hybrid_nyu-2ce69ec7.pt', help='pretrained DPT path')
    parser.add_argument('--backbone', type=str, default="vitb_rn50_384", help='DPT backbone')
    parser.add_argument('--preTrainedAtpModel', type=str, default="weights/Air_UNet_epoch_19.pt", help='pretrained Airlight Model')
    
    # train_one_epoch parameters
    parser.add_argument('--verbose', type=bool, default=True, help='print log')
    parser.add_argument('--betaStep', type=float, default=0.005, help='beta step')
    parser.add_argument('--stepLimit', type=int, default=250, help='Multi step limit')
    parser.add_argument('--eps', type=float, default=1e-12, help='Epsilon value for non zero calculating')
    parser.add_argument('--epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--val_step', type=int, default=1, help='validation step')
    parser.add_argument('--save_path', type=str, default="weights", help='Discriminator model save path')
    parser.add_argument('--wandb_log', action='store_true', default=False, help='WandB logging flag')
    
    # Discrminator hyperparam
    parser.add_argument('--save_log', type=bool, default=True, help='log save flag')
    parser.add_argument('--saveORshow', type=str, default='save',  help='results show or save')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate for optimizers')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 hyperparam for Adam optimizers')

    return parser.parse_args()


def train_oen_epoch(opt, epoch, depth_model, air_model, netD, optimizerD, criterion, dataloader):
    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.
    
    netD.train()
    errD_epoch = []
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Train")):
        optimizerD.zero_grad()
        
        # Data Init
        if opt.dataset == 'NYU':
            hazy_image, clear_image, GT_airlight, GT_depth, input_name = batch
        elif opt.dataset == 'RESIDE_beta':
            hazy_image, clear_image, GT_airlight, input_name = batch
        
        clear_image = clear_image.to(opt.device)
        # airlight = get_Airlight(hazy_image).to(opt.device)
        # airlight = GT_airlight.to(opt.device)
        init_hazy = hazy_image.clone().to(opt.device)
        cur_hazy = hazy_image.clone().to(opt.device)
        
        with torch.no_grad():
            airlight = air_model(init_hazy)
        
        # Multi-Step Depth Estimation and Dehazing
        beta = opt.betaStep
        
        best_psnr, best_ssim = np.zeros(opt.batchSize), np.zeros(opt.batchSize)
        psnr_preds = torch.Tensor(opt.batchSize, 3, opt.imageSize_H, opt.imageSize_W).to(opt.device)
        ssim_preds = torch.Tensor(opt.batchSize, 3, opt.imageSize_H, opt.imageSize_W).to(opt.device)
        last_pred = torch.Tensor()
        
        errD_fake_list, errD_real_list = [], []
        stop_flag_psnr, stop_flag_ssim = [], []
        
        for step in range(1, opt.stepLimit + 1):
            # Depth Estimation
            with torch.no_grad():
                cur_hazy = cur_hazy.to(opt.device)
                _, cur_depth = depth_model.forward(cur_hazy)
            cur_depth = cur_depth.unsqueeze(1)
            
            # Transmission Map
            trans = torch.exp(cur_depth * -beta)
            trans = torch.add(trans, opt.eps)
            
            # Dehazing
            prediction = (init_hazy - airlight) / trans + airlight
            prediction = torch.clamp(prediction, -1, 1)
            
            # Calculate Metrics            
            psnr = get_psnr_batch(prediction, clear_image).detach().cpu().numpy()
            ssim = get_ssim_batch(prediction, clear_image).detach().cpu().numpy()
            prediction = prediction.detach().cpu()
            
            for i in range(opt.batchSize):
                if i not in stop_flag_psnr:
                    if best_psnr[i] <= psnr[i]:
                        best_psnr[i] = psnr[i]
                    else:
                        psnr_preds[i] = prediction[i].clone()
                        stop_flag_psnr.append(i)
                
                if i not in stop_flag_ssim:      
                    if best_ssim[i] <= ssim[i]:
                        best_ssim[i] = ssim[i]
                    else:
                        ssim_preds[i] = prediction[i].clone()
                        stop_flag_ssim.append(i)
                
                if (i not in stop_flag_psnr) and (i not in stop_flag_ssim):
                    pred = prediction[i].clone().unsqueeze(0)
                    last_pred = torch.cat((last_pred, pred), 0)
                    
            if (len(stop_flag_psnr) == opt.batchSize) and (len(stop_flag_ssim) == opt.batchSize):
                # hazy_grid = make_grid(utils.denormalize(hazy_image.detach().cpu()))
                # clear_gird = make_grid(utils.denormalize(clear_image.detach().cpu()))
                # psnr_gird = make_grid(utils.denormalize(psnr_preds.detach().cpu()))
                # ssim_gird = make_grid(utils.denormalize(ssim_preds.detach().cpu()))
                # images = torch.cat((hazy_grid, clear_gird, psnr_gird, ssim_gird), 1)
                # show(images)
                
                real_image = torch.cat((clear_image, psnr_preds, ssim_preds))
                label = torch.full((real_image.shape[0],), real_label, dtype=torch.float, device=opt.device)
                output = netD(real_image.to(opt.device)).view(-1)
                errD_real = criterion(output, label)
                errD_real = errD_real * (last_pred.shape[0] / (real_image.shape[0] * 2))
                errD_real.backward()
                errD_real_list.append(errD_real.item())
                
                # fake_image = torch.Tensor()
                # pick_num = real_image.shape[0]
                # if pick_num <= len(last_pred):
                #     for idx in random.sample(range(0, len(last_pred)), pick_num):
                #         fake_image = torch.cat((fake_image, last_pred[idx]))
                # else:
                #     pick_num /= 2
                #     for idx in random.sample(range(0, len(last_pred)), pick_num):
                #         fake_image = torch.cat((fake_image, last_pred[idx]))

                label = torch.full((last_pred.shape[0], ), fake_label, dtype=torch.float, device=opt.device)
                output = netD(last_pred.to(opt.device)).view(-1)
                errD_fake = criterion(output, label)
                errD_fake.backward()
                errD_fake_list.append(errD_fake.item())
                
                optimizerD.step()

                if opt.wandb_log:
                    wandb.log({
                        'errD_fake' : errD_fake_list[-1],
                        'errD_real' : errD_real_list[-1],
                        'errD' : errD_fake_list[-1] + errD_real_list[-1],
                        'epoch' : epoch,
                    })
                
                break   # Stop Multi Step
            else:
                cur_hazy = prediction.clone()
                beta += opt.betaStep    # Set Next Step
        
        
        errD_fake = np.mean(np.array(errD_fake_list))
        errD_real = np.mean(np.array(errD_real_list))
        errD_epoch.append(errD_fake + errD_real)

        if opt.wandb_log:
            wandb.log({
                'errD' : errD_epoch[-1],
                'epoch' : epoch,
            })
        
        if opt.verbose:    
            print(f'\nlast_psnr = {best_psnr}')
            print(f'last_ssim = {best_ssim}')
            print(f'errD_fake = {errD_fake}')
            print(f'errD_real = {errD_real}')
            print(f"errD      = {errD_epoch[-1]}")
        
        if (batch_idx != 0) and (batch_idx % 100 == 0):
            torch.save({
                'epoch': epoch,
                'model_state_dict': netD.state_dict(),
                'optimizer_state_dict': optimizerD.state_dict()
                }, f"{opt.save_path}/Discriminator_epoch_{epoch:02d}_{batch_idx}.pt")
    
    return np.mean(np.array(errD_epoch))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()
    

if __name__ == '__main__':
    opt = get_args()
    
    # opt.seed = random.randint(1, 10000)
    # random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    print("=========| Option |=========\n", opt)
    print()
    
    depth_model = DPTDepthModel(
        path = opt.preTrainedModel,
        scale=0.00030, shift=0.1378, invert=True,
        backbone=opt.backbone,
        non_negative=True,
        enable_attention_hooks=False,
    )
    depth_model = depth_model.to(memory_format=torch.channels_last)
    depth_model.to(opt.device)
    depth_model.eval()
    
    air_model = Air_UNet(input_nc=3, output_nc=3, nf=8).to(opt.device)
    air_model.load_state_dict(torch.load(opt.preTrainedAtpModel)['model_state_dict'])
    air_model.eval()
    
    netD = Discriminator().to(opt.device)
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    # netD.apply(weights_init)
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))    
    criterion = nn.BCELoss()
    
    opt.dataRoot = 'C:/Users/IIPL/Desktop/data/NYU'
    # opt.dataRoot = 'D:/data/NYU'
    # opt.dataRoot = 'C:/Users/IIPL/Desktop/data/RESIDE_beta/'
    train_set = NYU_Dataset.NYU_Dataset(opt.dataRoot + '/train', [opt.imageSize_W, opt.imageSize_H], printName=False, returnName=True, norm=opt.norm)
    # train_set = RESIDE_Dataset.RESIDE_Beta_Dataset(opt.dataRoot,  [opt.imageSize_W, opt.imageSize_H], split='train', printName=False, returnName=True, norm=opt.norm )
    train_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize,
                             num_workers=2, drop_last=True, shuffle=True)
    
    val_set = NYU_Dataset.NYU_Dataset(opt.dataRoot + '/val', [opt.imageSize_W, opt.imageSize_H], printName=False, returnName=True, norm=opt.norm)
    val_loader = DataLoader(dataset=val_set, batch_size=opt.batchSize,
                             num_workers=2, drop_last=True, shuffle=True)
    
    if opt.wandb_log:
        wandb.init(project="Discriminator", entity="rus", name='DWT_GAN', config=opt)
    
    for epoch in range(1, opt.epochs+1):
        loss = train_oen_epoch(opt, epoch, depth_model, air_model, netD, optimizerD, criterion, train_loader)
        
        if epoch % opt.val_step == 0:
            validation(opt, epoch, depth_model, air_model, netD, val_loader)
            torch.save({
                'epoch': epoch,
                'model_state_dict': netD.state_dict(),
                'optimizer_state_dict': optimizerD.state_dict(),
                'loss': loss}, f"{opt.save_path}/Discriminator_epoch_{epoch:02d}.pt")