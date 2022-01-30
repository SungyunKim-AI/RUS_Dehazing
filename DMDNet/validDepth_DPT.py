import argparse
import cv2
import numpy as np
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import torch
from models.depth_models import DPTDepthModel
from dataset import *
from utils.entropy_module import Entropy_Module
from utils.airlight_module import Airlight_Module
from utils import util
from utils.metrics import get_ssim, get_psnr
import os
import csv

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--betaStep', type=float, default=0.01, help='beta step')
    parser.add_argument('--norm', action='store_true',  help='Image Normalize flag')
    # NYU
    # parser.add_argument('--dataset', required=False, default='NYU',  help='dataset name')
    # parser.add_argument('--dataRoot', type=str, default='D:/data/NYU_crop/val_for_depth',  help='data file path')
    
    # KITTI
    parser.add_argument('--dataset', required=False, default='KITTI',  help='dataset name')
    parser.add_argument('--dataRoot', type=str, default='D:/data/KITTI_eigen_benchmark/val',  help='data file path')
    
    
    
    return parser.parse_args()

def print_score(score):
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = score
    print(f'{abs_rel:.2f} {sq_rel:.2f} {rmse:.2f} {rmse_log:.2f} | {a1:.2f} {a2:.2f} {a3:.2f}')

def run(opt, model, loader, airlight_module, entropy_module):
    model.eval()
    
    output_folder = 'output_DPT_depth_' + opt.dataset
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if opt.dataset == 'KITTI':
        dim = 1
    if opt.dataset == 'NYU':
        dim = 2
    
    for batch in tqdm(loader):
        # hazy_input, clear_input, GT_depth, GT_airlight, GT_beta, haze
        hazy_images, clear_images, depth_images, gt_airlight, gt_beta, input_names = batch
        with torch.no_grad():
            hazy_images = hazy_images.to('cuda')
            clear_images = clear_images.to('cuda')
            depth_images = depth_images.to('cuda')
            cur_hazy = hazy_images.to('cuda')
            init_depth = model.forward(cur_hazy)
            #depth_images = model.forward(clear_images)

        output_name = output_folder + '/' + input_names[0] + '.csv'
        if not os.path.exists(f'{output_folder}/{input_names[0][:-4]}'):
            os.makedirs(f'{output_folder}/{input_names[0][:-4]}')
        f = open(output_name,'w', newline='')
        wr = csv.writer(f)
        
        cur_depth = None

        airlight = airlight_module.get_airlight(cur_hazy, opt.norm)
        airlight = util.air_denorm(opt.dataset, opt.norm, airlight)

        # airlight = util.air_denorm(opt.dataset, opt.norm, airlight).item()
        # airlight = util.air_denorm(opt.dataset, opt.norm, gt_airlight).item()

        steps = int((gt_beta+0.1) / opt.betaStep)
        for step in range(0,steps):
            with torch.no_grad():
                cur_depth = model.forward(cur_hazy)
            cur_hazy = util.denormalize(cur_hazy,opt.norm)
            trans = torch.exp(cur_depth/5*opt.betaStep*-1)
            prediction = (cur_hazy - airlight) / (trans + 1e-12) + airlight
            prediction = torch.clamp(prediction.float(),0,1)
             
            entropy, _, _ = entropy_module.get_cur(cur_hazy[0].detach().cpu().numpy().transpose(1,2,0))
            haze_set = torch.cat([util.denormalize(hazy_images, opt.norm)[0]*255, cur_hazy[0]*255, util.denormalize(clear_images, opt.norm)[0]*255], dim=1)
            
            ratio = np.median(depth_images[0].detach().cpu().numpy()) / np.median(cur_depth[0].detach().cpu().numpy())
            multi_score = util.compute_errors(cur_depth[0].detach().cpu().numpy() * ratio, depth_images[0].detach().cpu().numpy())
            wr.writerow([step]+multi_score+[entropy])
        
            depth_set = torch.cat([init_depth[0]*255/torch.max(init_depth[0]), cur_depth[0]*255/torch.max(cur_depth[0]), depth_images[0]*255/torch.max(depth_images[0])],dim=dim)
            depth_set = depth_set.repeat(3,1,1)
            save_set = torch.cat([haze_set, depth_set], dim=2)
            cv2.imwrite(f'{output_folder}/{input_names[0][:-4]}/{step:03}.jpg', cv2.cvtColor(save_set.detach().cpu().numpy().astype(np.uint8).transpose(1,2,0), cv2.COLOR_RGB2BGR))
            
            
            cv2.imshow('depth', cv2.resize(depth_set.detach().cpu().numpy().astype(np.uint8).transpose(1,2,0),(500,500)))
            cv2.imshow('dehaze', cv2.resize(cv2.cvtColor(haze_set.detach().cpu().numpy().astype(np.uint8).transpose(1,2,0),cv2.COLOR_RGB2BGR),(500,500)))
            cv2.waitKey(1)        

            cur_hazy = util.normalize(prediction[0].detach().cpu().numpy().transpose(1,2,0).astype(np.float32),opt.norm).unsqueeze(0).to('cuda')
        f.close()
    

if __name__ == '__main__':
    opt = get_args()
    opt.norm = True
    
    if opt.dataset == 'NYU':
        model = DPTDepthModel(
            path = 'weights/depth_weights/dpt_hybrid_nyu-2ce69ec7.pt',
            scale = 0.000305,
            shift = 0.1378,
            invert = True,
            backbone = 'vitb_rn50_384',
            non_negative=True,
            enable_attention_hooks=False
        ).to(memory_format=torch.channels_last)
        model.to('cuda')
    elif opt.dataset == 'KITTI':
        model = DPTDepthModel(
            path='weights/depth_weights/dpt_hybrid_kitti-cb926ef4.pt',
            scale=0.00006016,
            shift=0.00579,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        model.to('cuda')
    
    if opt.dataset == 'NYU':
        dataset = NYU_Dataset(opt.dataRoot, img_size=[640,480], norm=opt.norm)
    if opt.dataset == 'KITTI':
        dataset = KITTI_Dataset(opt.dataRoot, img_size=[1216,352], norm=opt.norm)
    loader = DataLoader(dataset, batch_size=1, num_workers=1, drop_last=False, shuffle=True)
    

    airlight_module = Airlight_Module()
    entropy_module = Entropy_Module()
    
    run(opt, model, loader, airlight_module, entropy_module)
    
    