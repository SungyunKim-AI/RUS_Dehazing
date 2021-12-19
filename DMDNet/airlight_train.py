import argparse
import random
import wandb
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from models import UNet
from dataset import NYU_Dataset, RESIDE_Beta_Dataset

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet')
    
    parser.add_argument('--dataset', required=False, default='RESIDE_beta',  help='dataset name')
    parser.add_argument('--dataRoot', type=str, default='',  help='data file path')
    parser.add_argument('--norm', type=bool, default=True,  help='Image Normalize flag')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    
    # learning parameters
    parser.add_argument('--seed', type=int, default=101, help='Random Seed')
    parser.add_argument('--batchSize', type=int, default=32, help='test dataloader input batch size')
    parser.add_argument('--imageSize_W', type=int, default=256, help='the width of the resized input image to network')
    parser.add_argument('--imageSize_H', type=int, default=256, help='the height of the resized input image to network')
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # train_one_epoch parameters
    parser.add_argument('--verbose', type=bool, default=True, help='print log')
    parser.add_argument('--epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--val_step', type=int, default=1, help='validation step')
    parser.add_argument('--save_path', type=str, default="weights", help='Airlight Estimation model save path')
    parser.add_argument('--wandb_log', action='store_true', default=True, help='WandB logging flag')
    
    # hyperparam
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizers')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 hyperparam for Adam optimizers')

    return parser.parse_args()

def train_one_epoch(opt, dataloader, net, optimizer, grad_scaler, criterion, epoch, iters):
    net.train()
    epoch_loss = []
    
    with tqdm(dataloader, desc=f'Epoch {epoch}/{opt.epochs}') as pbar:
        for batch in pbar:
            iters += 1
            
            # Data Init
            if opt.dataset == 'NYU':
                hazy_images, clear_images, GT_airlights, GT_depths, input_names = batch
            elif opt.dataset == 'RESIDE_beta':
                hazy_images, clear_images, GT_airlights, input_names = batch

            hazy_images = hazy_images.to(opt.device)
            GT_airlights = GT_airlights.to(opt.device, dtype=torch.float)
            
            with torch.cuda.amp.autocast(enabled=opt.amp):
                air_pred = net(hazy_images)
                loss = criterion(air_pred, GT_airlights)

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            
            epoch_loss.append(loss.item())
            
            if opt.wandb_log:
                wandb.log({
                    'train loss': loss.item(),
                    'iters': iters,
                    'epoch': epoch
                })
            pbar.set_postfix(**{'loss (batch)': loss.item()})
    
    epoch_loss = np.array(epoch_loss).mean()
    return  epoch_loss, iters


def validation(opt, dataloader, net, criterion, epoch):
    net.eval()
    val_score = []

    for batch in tqdm(dataloader, desc='Validate', leave=False):
        # Data Init
        if opt.dataset == 'NYU':
            hazy_images, clear_images, GT_airlights, GT_depths, input_names = batch
        elif opt.dataset == 'RESIDE_beta':
            hazy_images, clear_images, GT_airlights, input_names = batch
        
        hazy_images = hazy_images.to(opt.device)
        GT_airlights = GT_airlights.to(opt.device, dtype=torch.float)

        with torch.no_grad():
            air_preds = net(hazy_images)
            loss = criterion(GT_airlights, air_preds)

        val_score.append(loss.item())
        
    val_score = np.array(val_score).mean()
    print(f'Validation score: {val_score}')
    
    if opt.wandb_log:
        temp_hazy = hazy_images[0].detach().cpu().numpy().transpose(1,2,0)
        temp_hazy = np.rint((temp_hazy * 0.5 - 0.5) * 255.0).astype(np.uint8)
        temp_hazy = Image.fromarray(temp_hazy)
        
        temp_air = GT_airlights[0].detach().cpu().numpy().transpose(1,2,0)
        temp_air = np.repeat(np.rint((temp_air * 0.5 - 0.5) * 255.0).astype(np.uint8), 3, axis=2)
        temp_air = Image.fromarray(temp_air)
        
        temp_pred = air_preds[0].detach().cpu().numpy().transpose(1,2,0)
        temp_pred = np.repeat(np.rint((temp_pred * 0.5 - 0.5) * 255.0).astype(np.uint8), 3, axis=2)
        temp_pred = Image.fromarray(temp_pred)
        
        wandb.log({
            'validation score': val_score,
            'hazy_images': wandb.Image(temp_hazy),
            'airlight': {
                'true': wandb.Image(temp_air),
                'pred': wandb.Image(temp_pred)
            },
            'image_name' : input_names[0],
            'epoch' : epoch, 
        })
    
    return val_score



if __name__ == '__main__':
    opt = get_args()

    # opt.seed = random.randint(1, 10000)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    print("=========| Option |=========\n", opt)
    print()
    
    net = UNet(in_channels=3, out_channels=1, bilinear=True)
    net.to(device=opt.device)
    
    
    # opt.dataRoot = 'C:/Users/IIPL/Desktop/data/NYU'
    # opt.dataRoot = 'D:/data/NYU'
    opt.dataRoot = 'D:/data/RESIDE_beta'
    dataset_args = dict(img_size=[opt.imageSize_W, opt.imageSize_H], printName=False, returnName=True, norm=opt.norm)
    if opt.dataset == 'NYU':
        train_set = NYU_Dataset(opt.dataRoot + '/train', **dataset_args)
        val_set   = NYU_Dataset(opt.dataRoot + '/val', **dataset_args)
    elif opt.dataset == 'RESIDE_beta':
        train_set = RESIDE_Beta_Dataset(opt.dataRoot + '/train', **dataset_args)
        val_set   = RESIDE_Beta_Dataset(opt.dataRoot + '/val',   **dataset_args)
    
    loader_args = dict(batch_size=opt.batchSize, num_workers=2, drop_last=False, shuffle=True)
    train_loader = DataLoader(dataset=train_set, **loader_args)
    val_loader = DataLoader(dataset=val_set, **loader_args)
    
    if opt.wandb_log:
        wandb.init(project="Airlight", entity="rus", name='original_UNet_RESIDE', config=opt)
    
    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=opt.lr, weight_decay=1e-8, momentum=0.9)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=opt.amp)
    criterion = nn.L1Loss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    
    iters = 0
    for epoch in range(1, opt.epochs+1):
        epoch_loss, iters = train_one_epoch(opt, train_loader, net, optimizer, grad_scaler, criterion, epoch, iters)
        scheduler.step(epoch_loss)
        
        if epoch % opt.val_step == 0:
            val_score = validation(opt, val_loader, net, criterion, epoch)
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, f"{opt.save_path}/orginal_UNet_epoch_{epoch:02d}.pt")
    
