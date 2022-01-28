import argparse
import cv2
import numpy as np
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import torch
from models.depth_models import DPTDepthModel
from dataset import *
from utils.entropy_module import Entropy_Module
from utils import util
from utils.metrics import get_ssim, get_psnr
import os
import csv
import monodepth.networks as networks
from monodepth.layers import disp_to_depth


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

def run(opt, encoder, decoder, loader):
    
    output_folder = 'output_Monodepth_depth_' + opt.dataset
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for batch in tqdm(loader):
        # hazy_input, clear_input, GT_depth, GT_airlight, GT_beta, haze
        hazy_images, clear_images, depth_images, gt_airlight, gt_beta, input_names = batch
        # print(input_names)
        with torch.no_grad():
            hazy_images = hazy_images.to('cuda')
            clear_images = clear_images.to('cuda')
            depth_images = 1/(depth_images.to('cuda'))
            cur_hazy = hazy_images.to('cuda')
            init_depth = decoder(encoder(cur_hazy))[("disp", 0)]
            _, init_depth = disp_to_depth(init_depth, 0.1, 100)
            
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
            with torch.no_grad():
                cur_depth = decoder(encoder(cur_hazy))[("disp", 0)]
                _, cur_depth = disp_to_depth(cur_depth, 0.1, 100)
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
    
    # init encoder
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load('monodepth/models/mono+stereo_1024x320/encoder.pth')
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to('cuda')
    encoder.eval()
    
    # init decoder
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))
    loaded_dict = torch.load('monodepth/models/mono+stereo_1024x320/depth.pth', map_location='cuda')
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to('cuda')
    depth_decoder.eval()
    
    # init dataset
    val_set   = KITTI_Dataset('D:/data/KITTI' + '/val',  img_size=[feed_width,feed_height], norm=False)
    loader_args = dict(batch_size=1, num_workers=1, drop_last=False, shuffle=True)
    val_loader = DataLoader(dataset=val_set, **loader_args)

    run(opt, encoder, depth_decoder, val_loader)
    
    