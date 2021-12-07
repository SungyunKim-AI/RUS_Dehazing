# User warnings ignore
import warnings
warnings.filterwarnings("ignore")

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import argparse
import numpy as np
import random
from tqdm import tqdm

import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from dpt.models import DPTDepthModel
from dpt.discriminator import Discriminator

from dataset import NYU_Dataset
from torch.utils.data import DataLoader

from Module_Airlight.Airlight_Module import get_Airlight
from Module_Metrics.metrics import get_ssim_batch, get_psnr_batch
from util import misc, save_log, utils
from discriminator_val import validation



def get_args():
    parser = argparse.ArgumentParser()
    # dataset parameters
    parser.add_argument('--dataset', required=False, default='NYU_dataset',  help='dataset name')
    parser.add_argument('--dataRoot', type=str, default='',  help='data file path')
    parser.add_argument('--norm', type=bool, default=True,  help='Image Normalize flag')
    
    # learning parameters
    parser.add_argument('--seed', type=int, default=101, help='Random Seed')
    parser.add_argument('--batchSize', type=int, default=16, help='test dataloader input batch size')
    parser.add_argument('--imageSize_W', type=int, default=640, help='the width of the resized input image to network')
    parser.add_argument('--imageSize_H', type=int, default=480, help='the height of the resized input image to network')
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # model parameters
    parser.add_argument('--preTrainedModel', type=str, default='weights/dpt_hybrid_nyu-2ce69ec7.pt', help='pretrained DPT path')
    parser.add_argument('--backbone', type=str, default="vitb_rn50_384", help='DPT backbone')
    
    # train_one_epoch parameters
    parser.add_argument('--verbose', type=bool, default=True, help='print log')
    parser.add_argument('--betaStep', type=float, default=0.005, help='beta step')
    parser.add_argument('--stepLimit', type=int, default=250, help='Multi step limit')
    parser.add_argument('--eps', type=float, default=1e-12, help='Epsilon value for non zero calculating')
    parser.add_argument('--epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--val_step', type=int, default=1, help='validation step')
    parser.add_argument('--save_path', type=str, default="weights", help='Discriminator model save path')
    
    # Discrminator hyperparam
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate for optimizers')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 hyperparam for Adam optimizers')

    return parser.parse_args()


def train_oen_epoch(opt, model, netD, dataloader):
    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.
    
    netD.train()
    errD = []
    for batch in tqdm(dataloader, desc="Train"):
        netD.zero_grad()
        
        # Data Init
        hazy_image, clear_image, GT_airlight, GT_depth, input_name = batch
        
        clear_image = clear_image.to(opt.device)
        airlight = get_Airlight(hazy_image).to(opt.device)
        
        # for i in range(opt.batchSize):
        #     print()
        #     print('pred : ', torch.mean(airlight[i]))
        #     print('GT   : ', torch.mean(GT_airlight[i]))
        
        # exit()
        
        # Multi-Step Depth Estimation and Dehazing
        beta = opt.betaStep
        best_psnr = torch.zeros((opt.batchSize), dtype=torch.float32)
        best_ssim = torch.zeros((opt.batchSize), dtype=torch.float32)
        psnr_preds = torch.Tensor(opt.batchSize, 3, opt.imageSize_H, opt.imageSize_W).to(opt.device)
        ssim_preds = torch.Tensor(opt.batchSize, 3, opt.imageSize_H, opt.imageSize_W).to(opt.device)
        cur_hazy = hazy_image.clone().to(opt.device)
        errD_fake_list, errD_real_list = [], []
        
        for step in range(1, opt.stepLimit + 1):
            # Depth Estimation
            with torch.no_grad():
                cur_hazy = cur_hazy.to(opt.device)
                _, cur_depth = model.forward(cur_hazy)
            cur_depth = cur_depth.unsqueeze(1)
            
            # Transmission Map
            trans = torch.exp(cur_depth * -beta)
            trans = torch.add(trans, opt.eps)
            
            # Dehazing
            prediction = (cur_hazy - airlight) / trans + airlight
            prediction = torch.clamp(prediction, -1, 1)
            
            # Calculate Metrics            
            psnr = get_psnr_batch(prediction, clear_image).detach().cpu()
            ssim = get_ssim_batch(prediction, clear_image).detach().cpu()
            last_preds = torch.Tensor().to(opt.device)
            temp_hazy = torch.Tensor().to(opt.device)
            temp_air = torch.Tensor().to(opt.device)
            temp_clear = torch.Tensor().to(opt.device)
            
            for i in range(prediction.shape[0]):
                
                if best_psnr[i] < psnr[i]:
                    best_psnr[i] = psnr[i]
                    last_preds = torch.cat((last_preds, prediction[i].clone().unsqueeze(0)))
                    temp_hazy = torch.cat((temp_hazy, cur_hazy[i].clone().unsqueeze(0)))
                    temp_air = torch.cat((temp_air, airlight[i].clone().unsqueeze(0)))
                    temp_clear = torch.cat((temp_clear, clear_image[i].clone().unsqueeze(0)))
                else:
                    psnr_preds[i] = prediction[i].clone()
                    
                
                if best_ssim[i] < ssim[i]:
                    best_ssim[i] = ssim[i]
                else:
                    ssim_preds[i] = prediction[i].clone()
            
            
            if temp_hazy.shape[0] == 0:
                for real_image in [clear_image, psnr_preds, ssim_preds]:
                    label = torch.full((real_image.shape[0],), real_label, dtype=torch.float, device=opt.device)
                    output = netD(real_image).view(-1)
                    errD_real = criterion(output, label)
                    errD_real.backward()
                    errD_real_list.append(errD_real.item())
                optimizerD.step()
                
                break   # Stop Multi Step
            else:
                if step % 2 == 0:
                    label = torch.full((last_preds.shape[0],), fake_label, dtype=torch.float, device=opt.device)
                    output = netD(last_preds).view(-1)
                    errD_fake = criterion(output, label)
                    errD_fake.backward()
                    errD_fake_list.append(errD_fake.item())
                    optimizerD.step()
                
                clear_image = temp_clear
                cur_hazy = temp_hazy
                airlight = temp_air
                beta += opt.betaStep    # Set Next Step
        
        
        errD_fake = np.mean(np.array(errD_fake_list))
        errD_real = np.mean(np.array(errD_real_list))
        errD.append(errD_fake + errD_real)
        
        if opt.verbose:    
            print(f'\nlast_psnr = {best_psnr}')
            print(f'last_ssim = {best_ssim}')
            print(f"errD      = {errD[-1]}")
    
    return np.mean(np.array(errD))



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
            

if __name__ == '__main__':
    opt = get_args()
    
    # opt.seed = random.randint(1, 10000)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    print("=========| Option |=========\n", opt)
    print()
    
    model = DPTDepthModel(
        path = opt.preTrainedModel,
        scale=0.00030, shift=0.1378, invert=True,
        backbone=opt.backbone,
        non_negative=True,
        enable_attention_hooks=False,
    )
    model = model.to(memory_format=torch.channels_last)
    model.to(opt.device)
    model.eval()
    
    netD = Discriminator().to(opt.device)
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))    
     
    # opt.dataRoot = 'C:/Users/IIPL/Desktop/data/NYU'
    opt.dataRoot = 'D:/data/NYU'
    train_set = NYU_Dataset.NYU_Dataset(opt.dataRoot + '/train', [opt.imageSize_W, opt.imageSize_H], printName=False, returnName=True, norm=opt.norm)
    train_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize,
                             num_workers=2, drop_last=False, shuffle=True)
    
    val_set = NYU_Dataset.NYU_Dataset(opt.dataRoot + '/val', [opt.imageSize_W, opt.imageSize_H], printName=False, returnName=True, norm=opt.norm)
    val_loader = DataLoader(dataset=val_set, batch_size=opt.batchSize,
                             num_workers=2, drop_last=False, shuffle=True)
    
    for epoch in range(1, opt.epochs+1):
        loss = train_oen_epoch(opt, model, netD, train_loader)
        
        if epoch % opt.val_step == 0:
            # validation(opt, model, netD, val_loader)
            torch.save({
                'epoch': epoch,
                'model_state_dict': netD.state_dict(),
                'optimizer_state_dict': optimizerD.state_dict(),
                'loss': loss}, f"{opt.save_path}/Discriminator_epoch_{epoch:02d}.pt")