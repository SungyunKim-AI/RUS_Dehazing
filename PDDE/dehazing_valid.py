# User warnings ignore
import warnings

from torch._C import wait

warnings.filterwarnings("ignore")

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import argparse
import cv2
import numpy as np
import random
from tqdm import tqdm

import torch
from models.depth_models import DPTDepthModel
from models.air_models import UNet

from dataset import *
from torch.utils.data import DataLoader

from utils.metrics import get_ssim, get_psnr
from utils import misc, save_log, util
from utils.util import compute_errors
from utils.entropy_module import Entropy_Module
from glob import glob
from utils.io import *


def get_args():
    parser = argparse.ArgumentParser()
    # dataset parameters
    # NYU
    parser.add_argument('--dataset', required=False, default='NYU',  help='dataset name')
    parser.add_argument('--scale', type=float, default=0.000305,  help='depth scale')
    parser.add_argument('--shift', type=float, default= 0.1378,  help='depth shift')
    parser.add_argument('--preTrainedModel', type=str, default='weights/depth_weights/dpt_hybrid_nyu-2ce69ec7_nyu_haze_002.pt', help='pretrained DPT path')
    parser.add_argument('--preTrainedAirModel', type=str, default='weights/air_weights/Air_UNet_NYU_1D.pt', help='pretrained Air path')
    
    # RESIDE
    # parser.add_argument('--dataset', required=False, default='RESIDE_beta',  help='dataset name')
    # parser.add_argument('--scale', type=float, default=0.000150,  help='depth scale')
    # parser.add_argument('--shift', type=float, default= 0.1378,  help='depth shift')
    # parser.add_argument('--preTrainedModel', type=str, default='weights/depth_weights/dpt_hybrid_nyu-2ce69ec7_reside_haze_002.pt', help='pretrained DPT path')
    # parser.add_argument('--preTrainedAirModel', type=str, default='weights/air_weights/Air_UNet_RESIDE_1D.pt', help='pretrained Air path')
    
    # learning parameters
    parser.add_argument('--seed', type=int, default=101, help='Random Seed')
    parser.add_argument('--norm', action='store_true',  help='Image Normalize flag')
    parser.add_argument('--imageSize_W', type=int, default=256, help='the width of the resized input image to network')
    parser.add_argument('--imageSize_H', type=int, default=256, help='the height of the resized input image to network')
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # model parameters
    parser.add_argument('--backbone', type=str, default="vitb_rn50_384", help='DPT backbone')
    
    # run parameters
    parser.add_argument('--betaStep', type=float, default=0.005, help='beta step')
    parser.add_argument('--stepLimit', type=int, default=250, help='Multi step limit')
    parser.add_argument('--eps', type=float, default=1e-12, help='Epsilon value for non zero calculating')
    return parser.parse_args()
    


def run(opt, model, airlight_model, imgs, transform):
    model.eval()
    airlight_model.eval()

    for img in imgs:
        hazy, clear = load_item(img['hazy'], img['clear'],transform)
        haze_name = os.path.basename(img['hazy'])
        print(haze_name)

        hazy_images = torch.Tensor(hazy).unsqueeze(0).to(opt.device)
        clear_images = torch.Tensor(clear).unsqueeze(0).to(opt.device)
        cur_hazy = hazy_images
        last_psnr = 0
        with torch.no_grad():
            airlight = airlight_model.forward(cur_hazy)
        airlight = util.air_denorm(opt.dataset,opt.norm,airlight)
        print(airlight)

        for step in range(1, opt.stepLimit+1):
            with torch.no_grad():
                cur_depth = model.forward(cur_hazy)
            cur_hazy = util.denormalize(cur_hazy,opt.norm)

            trans = torch.exp(cur_depth*opt.betaStep*-1)
            prediction = (cur_hazy - airlight) / (trans + opt.eps) + airlight
            folder_name = f'output/{haze_name[:-4]}'
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)
            cv2.imwrite(f"{folder_name}/{haze_name[:-4]}_{step*opt.betaStep:1.3f}.jpg",cv2.cvtColor(prediction[0].detach().cpu().numpy().transpose(1,2,0),cv2.COLOR_RGB2BGR)*255)
            cv2.waitKey(0)
            cur_psnr = get_psnr(prediction[0].detach().cpu().numpy(),util.denormalize(clear_images,opt.norm)[0].detach().cpu().numpy())
            print(cur_psnr)
            
            if cur_psnr<last_psnr:
                break
            last_psnr = cur_psnr
            cur_hazy = util.normalize(prediction[0].detach().cpu().numpy().transpose(1,2,0),opt.norm).unsqueeze(0).to(opt.device)
            


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

    hazy_imgs = glob('input/hazy/*.*')

    imgs=[]
    for i,_ in enumerate(hazy_imgs):
        token = os.path.basename(hazy_imgs[i]).split('_')
        imgs.append({'hazy':hazy_imgs[i],'clear':'input/clear/'+token[0]+'.'+token[-1].split('.')[-1]})

    
    transform = make_transform([opt.imageSize_W, opt.imageSize_H], norm=opt.norm)

    run(opt, model, airlight_model, imgs, transform)