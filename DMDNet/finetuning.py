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

def get_args():
    parser = argparse.ArgumentParser()
    
    # model parameters : RESIDE-V0-OTS
    parser.add_argument('--dataset', required=False, default='RESIDE',  help='dataset name')
    parser.add_argument('--dataRoot', type=str, default='D:/data/RESIDE_V0_outdoor',  help='data file path')
    parser.add_argument('--scale', type=float, default=0.000150,  help='depth scale')
    parser.add_argument('--shift', type=float, default= 0.1378,  help='depth shift')
    parser.add_argument('--preTrainedModel', type=str, default='weights/depth_weights/dpt_hybrid_nyu-2ce69ec7.pt', help='pretrained DPT path')
    parser.add_argument('--backbone', type=str, default="vitb_rn50_384", help='DPT backbone')

    # learning parameters
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning Rage')
    parser.add_argument('--batchSize_train', type=int, default=4, help='train dataloader input batch size')
    parser.add_argument('--imageSize_W', type=int, default=256, help='the width of the resized input image to network')
    parser.add_argument('--imageSize_H', type=int, default=256, help='the height of the resized input image to network')
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--save_path', default='weights/depth_weights', help='folder to model checkpoints')
    parser.add_argument('--log_wandb', action='store_true',  help='wandb flag')

    return parser.parse_args()

def train(model, train_loader, optim, device, loss_fun, log_wandb, epoch):
    
    loss_sum = 0
    for batch in tqdm(train_loader):
        global_iter = 0
        optim.zero_grad()
        # hazy_input, clear_input, depth_input, airlight_input, beta_input, filename
        hazy_images, _, depth_images, _, _, _ = batch
        
        hazy_images = hazy_images.to(device)
        depth_images = depth_images.to(device)
        
        depth_pred = model.forward(hazy_images)
        global_iter +=1
        
        loss = loss_fun(depth_pred, depth_images)
        if epoch!=0:
            loss.backward()
            optim.step()
        loss_sum += loss.item()
    if log_wandb:
        wandb.log({
            "loss":loss_sum / global_iter,
            "epoch":epoch
        })
    
def evaluate(model, input, device, log_wandb, epoch):
    # hazy_input, clear_input, depth_input, airlight_input, beta_input, filename
    model.eval()
    hazy_image, _, depth_image, _, _, _, = input
    with torch.no_grad():
        depth_image = torch.Tensor(depth_image).to(device)
        hazy_image = torch.Tensor(hazy_image).to(device).unsqueeze(0)
        depth_pred = model.forward(hazy_image)[0]
    
    depth_set = torch.cat([depth_pred, depth_image],dim=2)
    cv2.imwrite(f'eval_{epoch:03}.jpg',(depth_set/10*255).detach().cpu().numpy().astype(np.uint8).transpose(1,2,0))
    
    
        
def run(model, train_loader, optim, device, log_wandb):
    #loss_fun = nn.L1Loss().to(device)
    loss_fun = nn.MSELoss().to(device)
    evaluate(model, train_loader.dataset[0], device, log_wandb, 0)
    train(model, train_loader, optim, device, loss_fun, log_wandb, 0)
    
    for epoch in range(1, 100):
        train(model, train_loader, optim, device, loss_fun, log_wandb, epoch)
        
        evaluate(model, train_loader.dataset[0], device, log_wandb, epoch)
        
        weight_path = f'{opt.save_path}/{os.path.basename(opt.preTrainedModel)[:-4]}_{opt.dataset}_{epoch:03}.pt'  #path for storing the weights of genertaor
        torch.save(model.state_dict(), weight_path)
        print(weight_path, "was saved!")
        
if __name__ == '__main__':
    
    opt = get_args()
    
    opt.log_wandb = False
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
        
    
    loader_args = dict(num_workers=4, drop_last=False, shuffle=False)
    train_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize_train, **loader_args)

    optimizer = optim.Adam(model.parameters(), opt.lr, betas = (0.9, 0.999), eps=1e-08)
    run(model, train_loader, optimizer, opt.device, opt.log_wandb)    