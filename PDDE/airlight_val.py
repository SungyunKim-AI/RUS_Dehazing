from os.path import basename
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models.air_models import UNet
from dataset import NYU_Dataset, RESIDE_Beta_Dataset

def get_args():
    # opt.dataRoot = 'C:/Users/IIPL/Desktop/data/NYU'
    # opt.dataRoot = 'D:/data/NYU'
    # opt.dataRoot = 'D:/data/RESIDE_beta'
    parser = argparse.ArgumentParser(description='Train the UNet')
    parser.add_argument('--dataset', required=False, default='NYU',  help='dataset name')
    parser.add_argument('--dataRoot', type=str, default='D:/data/NYU',  help='data file path')
    
    # learning parameters
    parser.add_argument('--seed', type=int, default=101, help='Random Seed')
    parser.add_argument('--batchSize', type=int, default=4, help='test dataloader input batch size')
    parser.add_argument('--imageSize_W', type=int, default=256, help='the width of the resized input image to network')
    parser.add_argument('--imageSize_H', type=int, default=256, help='the height of the resized input image to network')
    parser.add_argument('--norm', type=bool, default=True,  help='Image Normalize flag')
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # train_one_epoch parameters
    parser.add_argument('--verbose', type=bool, default=True, help='print log')
    parser.add_argument('--model_path', type=str, default="weights/air_weights", help='Airlight Estimation model save path')
    

    return parser.parse_args()

def validation(opt, dataloader, net, criterion):
    net.eval()
    val_score = []
    
    if opt.dataset == 'NYU':
        air_list = np.array([0.8, 0.9, 1.0])
    else:
        air_list = np.array([0.8, 0.85, 0.9, 0.95, 1.0])
        

    for batch in dataloader:
        # Data Init
        hazy_images, _, _, GT_air, _, input_name = batch
        hazy_images = hazy_images.to(opt.device)
        GT_air = GT_air.to(opt.device, dtype=torch.float)

        with torch.no_grad():
            pred_air = net(hazy_images)
            # loss = criterion(GT_air, pred_air)
            
        # pred_air = F.adaptive_avg_pool2d(pred_air, output_size=1).squeeze()
        
        GT_air = GT_air.detach().cpu().squeeze().numpy()
        pred_air = pred_air.detach().cpu().squeeze().numpy()
        if len(pred_air.shape) == 3:
            for i in range(opt.batchSize):
                min_air = np.min(pred_air[i])
                avg_air = np.mean(pred_air[i])
                print(f"GT airlight : {GT_air[i]:.4f} / min airlight : {min_air:.4f} / avg airlight : {avg_air:.4f}")
        else:
            GT_air = (GT_air * air_list.std()) + air_list.mean()
            GT_air = np.clip(GT_air, 0, 1)
            print(pred_air)
            pred_air = (pred_air * air_list.std()) + air_list.mean()
            pred_air = np.clip(pred_air, 0, 1)
            print(pred_air)
            err = np.abs(GT_air-pred_air)
            for i in range(opt.batchSize):
                file_name = basename(input_name[i])
                print(f"{file_name} => GT : {GT_air[i]:.4f} / pred : {pred_air[i]:.4f} / err : {err[i]:.4f}")

                val_score.append(err)
    
    return np.array(val_score).mean()


if __name__ == '__main__':
    opt = get_args()

    # opt.seed = random.randint(1, 10000)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    print("=========| Option |=========\n", opt)
    print()
    
    net = UNet([opt.imageSize_W, opt.imageSize_H], in_channels=3, out_channels=1, bilinear=True)
    net.to(device=opt.device)
    
    checkpoint = torch.load(opt.model_path + '/Air_UNet_NYU_1D.pt')
    net.load_state_dict(checkpoint['model_state_dict'])
    
    dataset_args = dict(img_size=[opt.imageSize_W, opt.imageSize_H], norm=opt.norm)
    if opt.dataset == 'NYU':
        val_set   = NYU_Dataset(opt.dataRoot + '/val', **dataset_args)
    elif opt.dataset == 'RESIDE_beta':
        val_set   = RESIDE_Beta_Dataset(opt.dataRoot + '/val',   **dataset_args)
    
    loader_args = dict(batch_size=opt.batchSize, num_workers=2, drop_last=False, shuffle=False)
    val_loader = DataLoader(dataset=val_set, **loader_args)
    criterion = nn.MSELoss()
    
    val_score = validation(opt, val_loader, net, criterion)
    print(f'Validation score: {val_score}')
    