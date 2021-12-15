import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import random
from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from models import Air_UNet, UNet
from Module_Airlight.Airlight_Module import get_Airlight

from dataset import NYU_Dataset, RESIDE_Dataset
from torch.utils.data import DataLoader

def get_args():
    parser = argparse.ArgumentParser()
    # dataset parameters
    parser.add_argument('--dataset', required=False, default='NYU',  help='dataset name')
    parser.add_argument('--dataRoot', type=str, default='',  help='data file path')
    parser.add_argument('--norm', type=bool, default=True,  help='Image Normalize flag')
    
    # learning parameters
    parser.add_argument('--seed', type=int, default=101, help='Random Seed')
    parser.add_argument('--batchSize', type=int, default=128, help='test dataloader input batch size')
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
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

    return parser.parse_args()

def train_one_epoch(opt, epoch, net, criterion, optimizer, grad_scaler, dataloader):
    net.train()
    epoch_loss = 0
    with tqdm(dataloader, desc=f'Epoch {epoch}/{opt.epochs}') as pbar:
        for batch in pbar:
            # Data Init
            if opt.dataset == 'NYU':
                hazy_images, clear_images, GT_airlights, GT_depths, input_names = batch
            elif opt.dataset == 'RESIDE_beta':
                hazy_images, clear_images, GT_airlights, input_names = batch
            
            hazy_images  = hazy_images.to(opt.device)
            GT_airlights = GT_airlights.float().to(opt.device)
            air_pred = net(hazy_images)
            loss = criterion(air_pred, GT_airlights)
            
            loss.backward()
            
            # with torch.cuda.amp.autocast(enabled=opt.amp):
            #     air_pred = net(hazy_images)
            #     loss = criterion(air_pred, GT_airlights)

            # optimizer.zero_grad(set_to_none=True)
            # grad_scaler.scale(loss).backward()
            # grad_scaler.step(optimizer)
            # grad_scaler.update()

            epoch_loss += loss.item()
            if opt.wandb_log:
                wandb.log({
                    'train loss': loss.item(),
                    'epoch': epoch
                })
            pbar.set_postfix(**{'loss (batch)': loss.item()})
    return epoch_loss

def validation(opt, epoch, net, criterion, dataloader):
    net.eval()
    val_score = []
    for batch in tqdm(dataloader, desc='Validation round', unit='batch', leave=False):
        # Data Init
        if opt.dataset == 'NYU':
            hazy_images, clear_images, GT_airlights, GT_depths, input_names = batch
        elif opt.dataset == 'RESIDE_beta':
            hazy_images, clear_images, GT_airlights, input_names = batch
        
        hazy_images = hazy_images.to(opt.device) 
        GT_airlights = GT_airlights.to(opt.device)
        # GT_airlights = GT_airlights.float().to(opt.device)

        with torch.no_grad():
            air_preds = net(hazy_images)
            loss = criterion(GT_airlights, air_preds)
            
            # air_val = torch.Tensor().to(opt.device)
            # for i in range(opt.batchSize):
            #     air_val = torch.cat((air_val, torch.min(air_preds[i]).unsqueeze(0)), 0)
                
            # print('\n', air_val.shape, GT_airlights.shape)
            # loss = criterion(GT_airlights, air_val)
        
        val_score.append(loss.item())
        
    val_score = np.mean(np.array(val_score))
    print(f'Validation score: {val_score}')
    
    if opt.wandb_log:
        # log_air = air_preds[0].cpu().clone()
        # log_air = torch.round(((log_air * 0.5) + 0.5) * 255)
        
        wandb.log({
            'validation score': val_score,
            # 'hazy_images': wandb.Image(hazy_images[0].cpu()),
            # 'airlight': {
            #     'true': wandb.Image(log_GT_air),
            #     'pred': wandb.Image(log_air)
            # },
            'epoch': epoch
        })
    
    return val_score

def validation_air_module(opt, criterion, dataloader):
    val_score = []
    for batch in tqdm(dataloader, desc='Validation round', unit='batch', leave=False):
        # Data Init
        if opt.dataset == 'NYU':
            hazy_images, clear_images, GT_airlights, GT_depths, input_names = batch
        elif opt.dataset == 'RESIDE_beta':
            hazy_images, clear_images, GT_airlights, input_names = batch
        
        airlights = get_Airlight(hazy_images)
        loss = criterion(GT_airlights, airlights)
        val_score.append(loss.item())
    
    val_score = np.mean(np.array(val_score))
    print(f'Validation score: {val_score}')
    
    if opt.wandb_log:
        log_GT_air = GT_airlights[0].clone()
        log_GT_air = torch.round(((log_GT_air * 0.5) + 0.5) * 255)
        
        log_air = airlights[0].clone()
        log_air = torch.round(((log_air * 0.5) + 0.5) * 255)
        
        wandb.log({
            'validation score': val_score,
            'hazy_images': wandb.Image(hazy_images[0].cpu()),
            'airlight': {
                'true': wandb.Image(log_GT_air),
                'pred': wandb.Image(log_air)
            }
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
    
    net = Air_UNet(input_nc=3, output_nc=3, nf=8).to(opt.device)
    # optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)) 
    optimizer = optim.RMSprop(net.parameters(), lr=opt.lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=opt.amp)
    criterion = nn.L1Loss()
       
    
    opt.dataRoot = 'C:/Users/IIPL/Desktop/data/NYU'
    # opt.dataRoot = 'D:/data/NYU'
    # opt.dataRoot = 'C:/Users/IIPL/Desktop/data/RESIDE_beta/'
    train_set = NYU_Dataset.NYU_Dataset(opt.dataRoot + '/train', [opt.imageSize_W, opt.imageSize_H], printName=False, returnName=True, norm=opt.norm)
    # train_set = RESIDE_Dataset.RESIDE_Beta_Dataset(opt.dataRoot,  [opt.imageSize_W, opt.imageSize_H], split='train', printName=False, returnName=True, norm=opt.norm )
    train_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize,
                             num_workers=2, drop_last=False, shuffle=True)
    
    val_set = NYU_Dataset.NYU_Dataset(opt.dataRoot + '/val', [opt.imageSize_W, opt.imageSize_H], printName=False, returnName=True, norm=opt.norm)
    val_loader = DataLoader(dataset=val_set, batch_size=opt.batchSize,
                             num_workers=2, drop_last=True, shuffle=True)
    
    opt.wandb_log = False
    if opt.wandb_log:
        wandb.init(project="Airlight", entity="rus", name='Air_UNet_pooling', config=opt)
    
    # validation_air_module(opt, criterion, val_loader)
    
    # for epoch in range(1, opt.epochs+1):
    #     epoch_loss = train_one_epoch(opt, epoch, net, criterion, optimizer, grad_scaler, train_loader)
        
    #     if epoch % opt.val_step == 0:
    #         val_score = validation(opt, epoch, net, criterion, val_loader)
    #         scheduler.step(val_score)
    #         torch.save({
    #             'epoch': epoch,
    #             'model_state_dict': net.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict()}, 
    #                    f"{opt.save_path}/Air_UNet_pool_epoch_{epoch:02d}.pt")
    
    epoch = 19
    checkpoint = torch.load(f'weights/Air_UNet_epoch_{epoch}.pt')
    net.load_state_dict(checkpoint['model_state_dict'])
    validation(opt, epoch, net, criterion, val_loader)