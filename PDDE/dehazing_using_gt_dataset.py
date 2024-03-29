# User warnings ignore
import warnings
warnings.filterwarnings("ignore")

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import argparse
import csv
import random
from tqdm import tqdm

import torch
from models.depth_models import DPTDepthModel
from models.air_models import UNet

from dataset import *
from torch.utils.data import DataLoader

from utils.metrics import get_ssim, get_psnr
from utils import util
from utils.util import compute_errors
from utils.entropy_module import Entropy_Module
from glob import glob
from utils.io import *


def get_args():
    parser = argparse.ArgumentParser()
    # dataset parameters
    # NYU
    # parser.add_argument('--dataset', required=False, default='NYU',  help='dataset name')
    # parser.add_argument('--dataRoot', type=str, default='D:/data/NYU_crop',  help='data file path')
    # parser.add_argument('--scale', type=float, default=0.000305,  help='depth scale')
    # parser.add_argument('--shift', type=float, default= 0.1378,  help='depth shift')
    # parser.add_argument('--preTrainedModel', type=str, default='weights/depth_weights/dpt_hybrid_nyu-2ce69ec7_nyu_haze_002.pt', help='pretrained DPT path')
    # parser.add_argument('--preTrainedAirModel', type=str, default='weights/air_weights/Air_UNet_NYU_1D.pt', help='pretrained Air path')
    
    # RESIDE
    parser.add_argument('--dataset', required=False, default='RESIDE',  help='dataset name')
    parser.add_argument('--dataRoot', type=str, default='D:/data/RESIDE',  help='data file path')
    parser.add_argument('--scale', type=float, default=0.000150,  help='depth scale')
    parser.add_argument('--shift', type=float, default= 0.1378,  help='depth shift')
    parser.add_argument('--preTrainedModel', type=str, default='weights/depth_weights/dpt_hybrid_nyu-2ce69ec_RESIDE_011.pt', help='pretrained DPT path')
    parser.add_argument('--preTrainedAirModel', type=str, default='weights/air_weights/Air_UNet_RESIDE_1D.pt', help='pretrained Air path')
    
    # learning parameters
    parser.add_argument('--seed', type=int, default=101, help='Random Seed')
    parser.add_argument('--norm', action='store_true',  help='Image Normalize flag')
    parser.add_argument('--imageSize_W', type=int, default=256, help='the width of the resized input image to network')
    parser.add_argument('--imageSize_H', type=int, default=256, help='the height of the resized input image to network')
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # model parameters
    parser.add_argument('--backbone', type=str, default="vitb_rn50_384", help='DPT backbone')
    
    # run parameters
    parser.add_argument('--betaStep', type=float, default=0.01, help='beta step')
    parser.add_argument('--stepLimit', type=int, default=100, help='Multi step limit')
    parser.add_argument('--eps', type=float, default=1e-12, help='Epsilon value for non zero calculating')
    return parser.parse_args()
    


def run(opt, model, airlight_model, metrics_module, loader):
    model.eval()
    airlight_model.eval()

    output_folder = 'output_' + opt.dataset
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pbar = tqdm(loader)
    for batch in pbar:
        hazy_images, clear_images, _, gt_airlight, gt_betas, input_names = batch
        gt_beta = gt_betas[0]
        
        last_mean_entropy = 0
        last_max_entropy = 0
        last_min_entropy = 0
            
        output_name = output_folder + '/' + input_names[0][:-4] + '.csv'
        f = open(output_name,'w', newline='')
        wr = csv.writer(f)
        
        with torch.no_grad():
            cur_hazy = hazy_images.to(opt.device)
            #airlight = airlight_model.forward(cur_hazy)
        #airlight = util.air_denorm(opt.dataset,opt.norm,airlight)
        airlight = util.air_denorm(opt.dataset,opt.norm, gt_airlight)[0].item()
        hazy_image = util.denormalize(hazy_images, opt.norm)[0].detach().cpu().numpy().transpose(1,2,0)
        clear_image = util.denormalize(clear_images, opt.norm)[0].detach().cpu().numpy().transpose(1,2,0)
        
        step_limit = int((gt_beta+0.1)/opt.betaStep)
        for step in range(0, step_limit):
            with torch.no_grad():
                cur_depth = model.forward(cur_hazy)
            cur_hazy = util.denormalize(cur_hazy,opt.norm)
            trans = torch.exp(cur_depth*opt.betaStep*-1)
            prediction = (cur_hazy - airlight) / (trans + opt.eps) + airlight
            if step==0:
                prediction = cur_hazy

            cur_mean_entropy, cur_max_entropy, cur_min_entropy = metrics_module.get_cur(prediction[0].detach().cpu().numpy().transpose(1,2,0))
            cur_psnr = get_psnr(prediction[0].detach().cpu().numpy(),util.denormalize(clear_images,opt.norm)[0].detach().cpu().numpy())
            cur_ssim = get_ssim(prediction[0].detach().cpu().numpy(),util.denormalize(clear_images,opt.norm)[0].detach().cpu().numpy()).item()
            
            # if cur_mean_entropy<last_mean_entropy and (best_mean_entropy_image is None) and step!=1:
            #     print("^^^^^^^^^^^^^^^^^^^^^^^^ best mean_entropy")
            #     best_mean_entropy_image = cv2.cvtColor(cur_hazy[0].detach().cpu().numpy().transpose(1,2,0),cv2.COLOR_RGB2BGR)
            
            # if cur_max_entropy<last_max_entropy and (best_max_entropy_image is None) and step!=1:
            #     print("^^^^^^^^^^^^^^^^^^^^^^^^ best max_entropy")
            #     best_max_entropy_image = cv2.cvtColor(cur_hazy[0].detach().cpu().numpy().transpose(1,2,0),cv2.COLOR_RGB2BGR)
            
            # if cur_min_entropy<last_min_entropy and (best_min_entropy_image is None) and step!=1:
            #     print("^^^^^^^^^^^^^^^^^^^^^^^^ best min_entropy")
            #     best_min_entropy_image = cv2.cvtColor(cur_hazy[0].detach().cpu().numpy().transpose(1,2,0),cv2.COLOR_RGB2BGR)
            
            wr.writerow([step, cur_mean_entropy, cur_max_entropy, cur_min_entropy, cur_psnr, cur_ssim])
            last_mean_entropy = cur_mean_entropy
            last_max_entropy = cur_max_entropy
            last_min_entropy = cur_min_entropy
            
            cur_hazy = util.normalize(prediction[0].detach().cpu().numpy().transpose(1,2,0),opt.norm).unsqueeze(0).to(opt.device)
        # if best_mean_entropy_image is not None:
        #     cv2.imshow("best_mean", best_mean_entropy_image)
        # if best_max_entropy_image is not None:
        #     cv2.imshow("best_max", best_max_entropy_image)
        # if best_min_entropy_image is not None:
        #     cv2.imshow("best_min", best_min_entropy_image)
        # cv2.waitKey(0) 
        
        f.close()
           


if __name__ == '__main__':
    opt = get_args()
    opt.norm = True
    opt.verbose = True
    
    #opt.seed = random.randint(1, 10000)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    print("=========| Option |=========\n", opt)
    
    
    model = DPTDepthModel(
        path = opt.preTrainedModel,
        scale=opt.scale, shift=opt.shift, invert=True,
        backbone=opt.backbone,
        non_negative=True,
        enable_attention_hooks=False,
    )
    model = model.to(memory_format=torch.channels_last)
    model.to(opt.device)
    
        
    airlight_model = UNet([opt.imageSize_W, opt.imageSize_H], in_channels=3, out_channels=1, bilinear=True)
    checkpoint = torch.load(opt.preTrainedAirModel)
    airlight_model.load_state_dict(checkpoint['model_state_dict'])
    airlight_model.to(opt.device)

    dataset_args = dict(img_size=[opt.imageSize_W, opt.imageSize_H], norm=opt.norm)
    if opt.dataset == 'NYU':
        val_set   = NYU_Dataset(opt.dataRoot + '/val', **dataset_args)
    elif opt.dataset == 'RESIDE':
        #val_set   = RESIDE_Dataset(opt.dataRoot + '/val',   **dataset_args)
        val_set   = RESIDE_Dataset('D:/data/RESIDE_V0_outdoor' + '/val',   **dataset_args)

    loader_args = dict(batch_size=1, num_workers=1, drop_last=False, shuffle=True)
    val_loader = DataLoader(dataset=val_set, **loader_args)
    metrics_module = Entropy_Module()
    
    run(opt, model, airlight_model, metrics_module, val_loader)