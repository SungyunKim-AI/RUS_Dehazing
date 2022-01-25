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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--betaStep', type=float, default=0.05, help='beta step')
    parser.add_argument('--norm', action='store_true',  help='Image Normalize flag')
    # NYU
    parser.add_argument('--dataset', required=False, default='NYU',  help='dataset name')
    parser.add_argument('--dataRoot', type=str, default='D:/data/NYU_crop/val',  help='data file path')
    return parser.parse_args()

def run(opt, model, loader, metrics_module):
    model.eval()
    
    for batch in loader:
        # hazy_input, clear_input, GT_depth, GT_airlight, GT_beta, haze
        hazy_images, _, depth_images, gt_airlight, gt_beta, _ = batch
        with torch.no_grad():
            hazy_images = hazy_images.to('cuda')
            depth_images = depth_images.to('cuda')
            cur_hazy = hazy_images.to('cuda')
            init_depth = model.forward(cur_hazy)
        cur_depth = None
        airlight = util.air_denorm(opt.dataset, opt.norm, gt_airlight).to('cuda')
        print('airlight = ', airlight)
        gt_beta = gt_beta.item()
        print('beta = ',gt_beta)
        
        steps = int(gt_beta / opt.betaStep)
            
        for step in tqdm(range(0,steps+1)):
            with torch.no_grad():
                cur_depth = model.forward(cur_hazy)
            cur_hazy = util.denormalize(cur_hazy,opt.norm)
            trans = torch.exp(cur_depth*opt.betaStep*-1)
            prediction = (cur_hazy - airlight) / (trans + 1e-12) + airlight
            
            cur_hazy = util.normalize(prediction[0].detach().cpu().numpy().transpose(1,2,0).astype(np.float32),opt.norm).unsqueeze(0).to('cuda')   
        
        init_psnr = get_psnr(init_depth[0].detach().cpu().numpy(), depth_images[0].detach().cpu().numpy())
        multi_psnr = get_psnr(cur_depth[0].detach().cpu().numpy(), depth_images[0].detach().cpu().numpy())
        print(init_psnr, multi_psnr)
        
        
        
        depth_set = torch.cat([init_depth[0]/10*255, cur_depth[0]/10*255, depth_images[0]/10*255],dim=2)
        
        cv2.imshow("haze", cv2.cvtColor(util.denormalize(hazy_images, opt.norm)[0].detach().cpu().numpy().transpose(1,2,0),cv2.COLOR_RGB2BGR))
        cv2.imshow('depth', depth_set.detach().cpu().numpy().astype(np.uint8).transpose(1,2,0))
        cv2.waitKey(0)
    

if __name__ == '__main__':
    opt = get_args()
    opt.norm = True
    
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
    
    if opt.dataset == 'NYU':
        dataset = NYU_Dataset(opt.dataRoot, img_size=[256,256], norm=opt.norm)
    loader = DataLoader(dataset, batch_size=1, num_workers=1, drop_last=False, shuffle=True)
    metrics_module = Entropy_Module()
    run(opt, model, loader, metrics_module)
    
    