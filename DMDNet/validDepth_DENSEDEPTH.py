import argparse
import cv2
import numpy as np
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import torch
from dataset import *
from utils.entropy_module import Entropy_Module
from utils import util
from utils.metrics import get_ssim, get_psnr
import os
import csv
from densedepth import *
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--betaStep', type=float, default=0.01, help='beta step')
    parser.add_argument('--norm', action='store_true',  help='Image Normalize flag')
    # NYU
    parser.add_argument('--dataset', required=False, default='KITTI',  help='dataset name')
    parser.add_argument('--dataRoot', type=str, default='D:/data/KITTI',  help='data file path')
    return parser.parse_args()

def print_score(score):
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = score
    print(f'{abs_rel:.2f} {sq_rel:.2f} {rmse:.2f} {rmse_log:.2f} | {a1:.2f} {a2:.2f} {a3:.2f}')

def run(opt, model, loader):
    up_module = torch.nn.Upsample(scale_factor=(2,2)).to('cuda')
    
    output_folder = 'output_DenseDenpth_depth_' + opt.dataset
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for batch in tqdm(loader):
        # hazy_input, clear_input, GT_depth, GT_airlight, GT_beta, haze
        hazy_images, clear_images, depth_images, gt_airlight, gt_beta, input_names = batch
        # print(input_names)
        with torch.no_grad():
            hazy_images = hazy_images.to('cuda')
            clear_images = clear_images.to('cuda')
            #depth_images = 1/(depth_images.to('cuda'))
            cur_hazy = hazy_images.to('cuda')
            init_depth = 1/up_module(model(cur_hazy))*20
            depth_images = 1/up_module(model.forward(clear_images))*20
            
        output_name = output_folder + '/' + input_names[0] + '.csv'
        if not os.path.exists(f'{output_folder}/{input_names[0][:-4]}'):
            os.makedirs(f'{output_folder}/{input_names[0][:-4]}')
        f = open(output_name,'w', newline='')
        wr = csv.writer(f)
        
        cur_depth = None
        airlight = util.air_denorm(opt.dataset, opt.norm, gt_airlight).item()
        # print('airlight = ', airlight)
        gt_beta = gt_beta.item()
        # print('beta = ',gt_beta)
        
        init_score = util.compute_errors(init_depth[0].detach().cpu().numpy(), depth_images[0].detach().cpu().numpy())
        
        steps = int((gt_beta+0.1) / opt.betaStep)
        dehaze = None
        for step in range(0,steps):
            # if step == int(gt_beta/opt.betaStep)-1:
            #     print('gt_step')
            with torch.no_grad():
                cur_depth = 1/up_module(model(cur_hazy))*20
            cur_hazy = util.denormalize(cur_hazy,opt.norm)
            trans = torch.exp(cur_depth*opt.betaStep*-1)
            prediction = (cur_hazy - airlight) / (trans + 1e-12) + airlight
            
            prediction = torch.clamp(prediction.float(),0,1)
            cur_hazy = util.normalize(prediction[0].detach().cpu().numpy().transpose(1,2,0).astype(np.float32),opt.norm).unsqueeze(0).to('cuda')
            
            ratio = np.median(depth_images[0].detach().cpu().numpy()) / np.median(cur_depth[0].detach().cpu().numpy())
            
            multi_score = util.compute_errors(cur_depth[0].detach().cpu().numpy() * ratio, depth_images[0].detach().cpu().numpy())
            wr.writerow([step]+multi_score)
            
            # if step == int(gt_beta/opt.betaStep):
            #     dehaze = cur_hazy.clone()
        
        
            haze_set = torch.cat([util.denormalize(hazy_images, opt.norm)[0]*255, util.denormalize(cur_hazy, opt.norm)[0]*255, util.denormalize(clear_images, opt.norm)[0]*255], dim=1)
            depth_set = torch.cat([init_depth[0]*255/torch.max(init_depth[0]), cur_depth[0]*255/torch.max(cur_depth[0]), depth_images[0]*255/torch.max(depth_images[0])],dim=1)
            
            depth_set = depth_set.repeat(3,1,1)
            save_set = torch.cat([haze_set, depth_set], dim=2)
            cv2.imwrite(f'{output_folder}/{input_names[0][:-4]}/{step:03}.jpg', cv2.cvtColor(save_set.detach().cpu().numpy().astype(np.uint8).transpose(1,2,0), cv2.COLOR_RGB2BGR))
           
            
            
            # cv2.imshow('depth', depth_set.detach().cpu().numpy().astype(np.uint8).transpose(1,2,0))
            # cv2.imshow('dehaze', cv2.cvtColor(haze_set.detach().cpu().numpy().astype(np.uint8).transpose(1,2,0),cv2.COLOR_RGB2BGR))
            # cv2.waitKey(0)        
        #init_psnr = get_psnr(init_depth[0].detach().cpu().numpy(), depth_images[0].detach().cpu().numpy())
        #multi_psnr = get_psnr(cur_depth[0].detach().cpu().numpy(), depth_images[0].detach().cpu().numpy())
        # print(init_psnr, multi_psnr)

        f.close()
    

if __name__ == '__main__':
    opt = get_args()
    opt.norm = False
    
    # init model
    model = DenseDepth()
    model.eval()
    model.to('cuda')
    
    
    # init dataset
    if opt.dataset=='KITTI':
        weight = torch.load('densedepth/densedepth_kitti.pt')
        model.load_state_dict(weight)
        width = 1216
        height = 352
        val_set   = KITTI_Dataset('D:/data/KITTI' + '/val',  img_size=[width,height], norm=False)
    elif opt.dataset == 'NYU':
        weight = torch.load('densedepth/densedepth_nyu.pt')
        model.load_state_dict(weight)
        width = 640
        height = 480
    
    loader_args = dict(batch_size=1, num_workers=1, drop_last=False, shuffle=True)
    val_loader = DataLoader(dataset=val_set, **loader_args)

    run(opt, model, val_loader)
    
    