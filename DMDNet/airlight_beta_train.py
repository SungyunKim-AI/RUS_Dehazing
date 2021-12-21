import argparse
import random
import wandb
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from models.air_beta_models import UNet
from dataset import NYU_Dataset, RESIDE_Beta_Dataset

def get_args():
    # opt.dataRoot = 'C:/Users/IIPL/Desktop/data/NYU'
    # opt.dataRoot = 'D:/data/NYU'
    # opt.dataRoot = 'D:/data/RESIDE_beta'
    parser = argparse.ArgumentParser(description='Train the Airlight & Beta Train')
    parser.add_argument('--dataset', required=False, default='RESIDE_beta',  help='dataset name')
    parser.add_argument('--dataRoot', type=str, default='D:/data/RESIDE_beta',  help='data file path')
    
    # learning parameters
    parser.add_argument('--seed', type=int, default=101, help='Random Seed')
    parser.add_argument('--batchSize', type=int, default=8, help='test dataloader input batch size')
    parser.add_argument('--imageSize_W', type=int, default=256, help='the width of the resized input image to network')
    parser.add_argument('--imageSize_H', type=int, default=256, help='the height of the resized input image to network')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizers')
    parser.add_argument('--epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--val_step', type=int, default=1, help='validation step')
    parser.add_argument('--norm', type=bool, default=True,  help='Image Normalize flag')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # train_one_epoch parameters
    parser.add_argument('--verbose', type=bool, default=True, help='print log')
    parser.add_argument('--save_path', type=str, default="weights/air_beta_weights", help='Airlight Estimation model save path')
    parser.add_argument('--wandb_log', action='store_true', default=True, help='WandB logging flag')
    

    return parser.parse_args()

def train_one_epoch(opt, dataloader, net, optimizer, grad_scaler, criterion, epoch, iters):
    net.train()
    epoch_loss = []
    
    with tqdm(dataloader, desc=f'Epoch {epoch}/{opt.epochs}') as pbar:
        for batch in pbar:
            iters += 1
            
            # Data Init
            hazy_images, clear_images, GT_depths, GT_airs_, GT_betas_, input_names = batch
            hazy_images = hazy_images.to(opt.device)
            
            GT_airs = torch.Tensor()
            GT_betas = torch.Tensor()
            for i in range(opt.batchSize):
                GT_airs = torch.cat((GT_airs, torch.full((1,1,250,250), GT_airs_[i].item())))
                GT_betas = torch.cat((GT_betas, torch.full((1,1,250,250), GT_betas_[i].item())))
            GT_airs = GT_airs.to(opt.device)
            GT_betas = GT_betas.to(opt.device)
            
            with torch.cuda.amp.autocast(enabled=opt.amp):
                pred_airs, pred_betas = net(hazy_images)
                
                air_loss = criterion["air_loss"](pred_airs, GT_airs)
                beta_loss = criterion["beta_loss"](pred_betas, GT_betas)
                var_loss = (pred_airs.std(dim=(2,3)) + pred_betas.std(dim=(2,3))).mean()
            
            loss = air_loss + beta_loss + 0.0001 * var_loss
            
            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            
            epoch_loss.append(loss.item())
            
            if opt.wandb_log:
                wandb.log({
                    'air loss': air_loss.item(),
                    'beta loss': beta_loss.item(),
                    'var loss': var_loss.item(),
                    'loss': loss.item(),
                    'iters': iters,
                    'epoch': epoch
                })
            pbar.set_postfix(**{'loss (batch)': loss.item()})
    
    epoch_loss = np.array(epoch_loss).mean()
    return  epoch_loss, iters


def validation(opt, dataloader, net, criterion, epoch):
    net.eval()

    for batch in tqdm(dataloader, desc='Validate', leave=False):
        # Data Init
        hazy_images, clear_images, GT_depths, GT_airs_, GT_betas_, input_names = batch
        hazy_images = hazy_images.to(opt.device)
        
        GT_airs = torch.Tensor()
        GT_betas = torch.Tensor()
        for i in range(opt.batchSize):
            GT_airs = torch.cat((GT_airs, torch.full((1,1,250,250), GT_airs_[i].item())))
            GT_betas = torch.cat((GT_betas, torch.full((1,1,250,250), GT_betas_[i].item())))
        GT_airs = GT_airs.to(opt.device)
        GT_betas = GT_betas.to(opt.device)
        
        with torch.no_grad():
            pred_airs, pred_betas = net(hazy_images)                
            air_loss = criterion["air_loss"](pred_airs, GT_airs)
            beta_loss = criterion["beta_loss"](pred_betas, GT_betas)
            var_loss = (pred_airs.std(dim=(2,3)) + pred_betas.std(dim=(2,3))).mean()
            
        loss = air_loss + beta_loss + var_loss
    
    if opt.wandb_log:
        wandb.log({
            'val air loss': air_loss.item(),
            'val beta loss': beta_loss.item(),
            'val var loss': var_loss.item(),
            'val loss': loss.item(),
            'iters': iters,
            'epoch': epoch
        })



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
    
    dataset_args = dict(img_size=[opt.imageSize_W, opt.imageSize_H], norm=opt.norm)
    if opt.dataset == 'NYU':
        train_set = NYU_Dataset(opt.dataRoot + '/train', **dataset_args)
        val_set   = NYU_Dataset(opt.dataRoot + '/val', **dataset_args)
    elif opt.dataset == 'RESIDE_beta':
        train_set = RESIDE_Beta_Dataset(opt.dataRoot + '/train', **dataset_args)
        val_set   = RESIDE_Beta_Dataset(opt.dataRoot + '/val',   **dataset_args)
    
    loader_args = dict(batch_size=opt.batchSize, num_workers=2, drop_last=True, shuffle=True)
    train_loader = DataLoader(dataset=train_set, **loader_args)
    val_loader = DataLoader(dataset=val_set, **loader_args)
    
    if opt.wandb_log:
        wandb.init(project="Air_Beta", entity="rus", name='UNet_RESIDE_1D', config=opt)
    
    optimizer = optim.RMSprop(net.parameters(), lr=opt.lr, weight_decay=1e-8, momentum=0.9)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=opt.amp)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    
    criterion = {"dehazing_loss": nn.L1Loss().to(opt.device), 
                 "beta_loss": nn.L1Loss().to(opt.device), 
                 "air_loss": nn.L1Loss().to(opt.device)}
    
    iters = 0
    for epoch in range(1, opt.epochs+1):
        epoch_loss, iters = train_one_epoch(opt, train_loader, net, optimizer, grad_scaler, criterion, epoch, iters)
        scheduler.step(epoch_loss)
        
        if epoch % opt.val_step == 0:
            validation(opt, val_loader, net, criterion, epoch)
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, f"{opt.save_path}/air_beta_{epoch:02d}.pt")
    
