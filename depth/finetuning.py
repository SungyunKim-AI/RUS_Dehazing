from turtle import clear
import cv2

import torch.nn as nn
import argparse
import torch.optim as optim
import torch
import wandb
import os
import numpy as np

from tqdm import tqdm
from models.depth_models import DPTDepthModel
from dataset import *
from torch.utils.data import DataLoader
from utils.util import compute_errors
from utils import util
from utils.metrics import get_ssim, get_psnr

def get_args():
    parser = argparse.ArgumentParser()
    
    # model parameters : RESIDE-V0-OTS
    parser.add_argument('--dataset', required=False, default='RESIDE',  help='dataset name')
    parser.add_argument('--dataRoot', type=str, default='D:/data/RESIDE_V0_outdoor',  help='data file path')
    parser.add_argument('--scale', type=float, default=0.000150,  help='depth scale')
    parser.add_argument('--shift', type=float, default= 0.1378,  help='depth shift')
    parser.add_argument('--preTrainedModel', type=str, default='weights/depth_weights/dpt_hybrid_kitti-cb926ef4.pt', help='pretrained DPT path')
    parser.add_argument('--backbone', type=str, default="vitb_rn50_384", help='DPT backbone')

    # learning parameters
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning Rage')
    parser.add_argument('--batchSize_train', type=int, default=12, help='train dataloader input batch size')
    parser.add_argument('--imageSize_W', type=int, default=256, help='the width of the resized input image to network')
    parser.add_argument('--imageSize_H', type=int, default=256, help='the height of the resized input image to network')
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--save_path', default='weights/depth_weights', help='folder to model checkpoints')
    parser.add_argument('--log_wandb', action='store_true',  help='wandb flag')

    return parser.parse_args()

def train(model, train_loader, optim, device, loss_fun, log_wandb, epoch, global_iter):
    
    loss_sum = 0
    for batch in tqdm(train_loader):
        optim.zero_grad()
        # hazy_input, clear_input, depth_input, airlight_input, beta_input, filename
        hazy_images, _, depth_images, _, _, _ = batch
        
        hazy_images = hazy_images.to(device)
        depth_images = depth_images.to(device)
        
        depth_pred = model.forward(hazy_images)
        global_iter +=1
        
        loss = loss_fun(depth_pred, depth_images)
        loss_sum += loss.item()
        if epoch!=0:
            loss.backward()
            optim.step()
    if log_wandb:
        wandb.log({
            "train loss":loss_sum/len(train_loader),
            "iters": global_iter,
            "epoch":epoch
        })
    
def evaluate(model, val_loader, device, log_wandb, epoch):
    # hazy_input, clear_input, depth_input, airlight_input, beta_input, filename
    model.eval()
    psnr_sum = 0

    for batch in val_loader:
        hazy_images, clear_images, _, gt_airlight, gt_beta, _, = batch
        loss = 0
        with torch.no_grad():
            hazy_images = hazy_images.to(device)
            depth_pred = model(hazy_images)
            hazy_images = util.denormalize(hazy_images,opt.norm)
            clear_images = util.denormalize(clear_images.to(device),opt.norm)
            trans = torch.exp(depth_pred*gt_beta.item()*-1)
            gt_airlight = util.air_denorm(opt.dataset, opt.norm, gt_airlight)[0][0]
            prediction = (hazy_images - gt_airlight) / (trans + 1e-12) + gt_airlight
            #loss = loss_fun(prediction, clear_images)
            psnr = get_psnr(prediction.detach().cpu(), clear_images.detach().cpu())
            psnr_sum += psnr
    if log_wandb:
        wandb.log({
            "val psnr":psnr_sum/len(val_loader),
            "epoch":epoch
        })
        
        # cv2.imshow("haze", hazy_images[0].detach().cpu().numpy().transpose(1,2,0))
        # cv2.imshow("prediction", prediction[0].detach().cpu().numpy().transpose(1,2,0))
        # cv2.waitKey(0)
    
    
        
def run(model, train_loader, val_loader, optim, device, log_wandb):
    #loss_fun = nn.L1Loss().to(device)
    loss_fun = nn.MSELoss().to(device)
    global_iter = 0;   
    evaluate(model, val_loader, device, log_wandb, 0)
    train(model, train_loader, optim, device, loss_fun, log_wandb, 0, global_iter)
    
    for epoch in range(1, 100):
        train(model, train_loader, optim, device, loss_fun, log_wandb, epoch, global_iter)
        
        evaluate(model, val_loader, device, log_wandb, epoch)
        
        weight_path = f'{opt.save_path}/{os.path.basename(opt.preTrainedModel)[:-3]}_{opt.dataset}_{epoch:03}.pt'  #path for storing the weights of genertaor
        torch.save(model.state_dict(), weight_path)
        print(weight_path, "was saved!")
        
if __name__ == '__main__':
    
    opt = get_args()
    
    opt.log_wandb = True
    opt.norm = True
    opt.verbose = True
    
    config_defaults = {
        'model_name' : 'DPT_finetuning',
        'init_lr' : opt.lr,
        'dataset' : opt.dataset,
        'batch_size' : opt.batchSize_train,
        'image_size' : [opt.imageSize_W, opt.imageSize_H]
    }
    
    if opt.log_wandb:
        wandb.init(config=config_defaults, project='Dehazing', entity='rus')
        wandb.run.name = config_defaults['model_name']
        
    model = DPTDepthModel(
        path = opt.preTrainedModel,
        scale = opt.scale,
        shift = opt.shift,
        invert = True,
        non_negative = True,
        enable_attention_hooks=False,
    )
    
    model = model.to(memory_format = torch.channels_last)
    model.to(opt.device)
    
    dataset_args = dict(img_size=[opt.imageSize_W, opt.imageSize_H], norm=opt.norm)
    if opt.dataset == 'RESIDE':
        train_set = RESIDE_Dataset(opt.dataRoot + '/train', **dataset_args)
        val_set = RESIDE_Dataset(opt.dataRoot + '/val', **dataset_args)
        
    
    loader_args = dict(num_workers=4, drop_last=False, shuffle=True)
    train_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize_train, **loader_args)
    val_loader = DataLoader(dataset=val_set, batch_size=1, drop_last=False, shuffle=False)

    optimizer = optim.Adam(model.parameters(), opt.lr, betas = (0.9, 0.999), eps=1e-08)
    run(model, train_loader, val_loader, optimizer, opt.device, opt.log_wandb)    