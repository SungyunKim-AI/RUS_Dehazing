import argparse
import cv2
import numpy as np
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import torch
from KITTI_Dataset import *
from utils import util
import os
import csv
import monodepth.networks as networks
from monodepth.layers import disp_to_depth
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--betaStep', type=float, default=0.005, help='beta step')
    parser.add_argument('--norm', action='store_true',  help='Image Normalize flag')
    parser.add_argument('--dataset', required=False, default='KITTI',  help='dataset name')
    parser.add_argument('--dataRoot', type=str, default='D:/data/KITTI',  help='data file path')
    parser.add_argument('--dehazedModel', type=str, default='FFA',  help='dehazing model name')
    return parser.parse_args()

def print_score(score):
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = score
    print(f'{abs_rel:.2f} {sq_rel:.2f} {rmse:.2f} {rmse_log:.2f} | {a1:.2f} {a2:.2f} {a3:.2f}')

def run(opt, encoder, decoder, loader, improve_best_list=None):
    
    output_folder = 'D:/data/output_depth/dehazed_Monodepth_' + opt.dataset
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_name = f'{output_folder}/dehazed_Monodepth_{opt.dataset}.csv'
    f = open(output_name,'w', newline='')
    
    for batch in tqdm(loader):
        # haze_input, clear_input, GT_depth, GT_airlight, GT_beta, haze
        haze_images, clear_images, dehazed_images, gt_depth, _, _, input_names = batch
        
        if input_names[0] != '2011-10-03-drive-0047-sync-0000000096_1.0_0.06':
            continue
        
        # Improve best
        input_name = os.path.basename(input_names[0])
        if improve_best_list is not None:
            if input_name not in improve_best_list:
                continue
        
        haze_images = haze_images.to('cuda')
        clear_images = clear_images.to('cuda')
        dehazed_images = dehazed_images.to('cuda')
        
        with torch.no_grad():
            # set gt depth
            gt_depth_median = torch.median(gt_depth)
            _, gt_depth = disp_to_depth(decoder(encoder(clear_images))[("disp", 0)], 0.1, 100)
            init_ratio = gt_depth_median / torch.median(gt_depth).item()
            gt_depth *= init_ratio
            
            # depth estimation for haze images
            _, haze_depth = disp_to_depth(decoder(encoder(haze_images))[("disp", 0)], 0.1, 100)
            haze_depth *= init_ratio
            
            # depth estimation for dehazed images
            dehazed_depth = decoder(encoder(dehazed_images))[("disp", 0)]
            _, dehazed_depth = disp_to_depth(dehazed_depth, 0.1, 100)
            dehazed_depth *= init_ratio
            
        # compute errors
        ratio = np.median(gt_depth[0].detach().cpu().numpy()) / np.median(dehazed_depth[0].detach().cpu().numpy())
        haze_multi_score = util.compute_errors(haze_depth[0].detach().cpu().numpy() * ratio, gt_depth[0].detach().cpu().numpy())
        dehazed_multi_score = util.compute_errors(dehazed_depth[0].detach().cpu().numpy() * ratio, gt_depth[0].detach().cpu().numpy())
        
        wr = csv.writer(f)
        wr.writerow([input_name] + haze_multi_score + dehazed_multi_score)

        # #visualize haze and depth
        haze_viz = (util.denormalize(haze_images, opt.norm)[0].detach().cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)
        dehazed_viz = (dehazed_images[0].detach().cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)
        init_clear_viz = (util.denormalize(clear_images, opt.norm)[0].detach().cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)
        haze_set= cv2.cvtColor(np.concatenate([haze_viz, dehazed_viz, init_clear_viz], axis = 0), cv2.COLOR_RGB2BGR)

        haze_depth_viz = util.visualize_depth_inverse(haze_depth[0])
        dehazed_depth_viz = util.visualize_depth_inverse(dehazed_depth[0])
        gt_depth_viz = util.visualize_depth_inverse(gt_depth[0])
        depth_set = np.concatenate([haze_depth_viz, dehazed_depth_viz, gt_depth_viz],axis=0)

        save_set = np.concatenate([haze_set, depth_set], axis=1)
        cv2.imwrite(f'{output_folder}/{input_name}.jpg', save_set)
        
        print([input_name] + haze_multi_score + dehazed_multi_score)
        cv2.imshow('depth', cv2.resize(save_set,(2000,1000)))
        cv2.waitKey(0)
        cv2.waitKey(0)

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
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
    loaded_dict = torch.load('monodepth/models/mono+stereo_1024x320/depth.pth', map_location='cuda')
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to('cuda')
    depth_decoder.eval()
    
    # init dataset
    val_set   = KITTI_Dataset_dehazed(opt.dataRoot + '/val',  img_size=[feed_width,feed_height], model_name=opt.dehazedModel, norm=opt.norm)
    loader_args = dict(batch_size=1, num_workers=1, drop_last=False, shuffle=False)
    val_loader = DataLoader(dataset=val_set, **loader_args)

    # Improve best
    df = pd.read_csv('D:/data/output_depth/_statistics/KITTI_statistic.csv')
    improve_best_list = df['name'].tolist()
    
    run(opt, encoder, depth_decoder, val_loader, improve_best_list=None)
    
    