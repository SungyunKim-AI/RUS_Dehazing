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
from utils.io import *


def get_args():
    parser = argparse.ArgumentParser()
    # RESIDE
    parser.add_argument('--dataset', required=False, default='RESIDE',  help='dataset name')
    parser.add_argument('--dataRoot', type=str, default='D:/data/RESIDE_V0_outdoor',  help='data file path')
    parser.add_argument('--scale', type=float, default=0.000150,  help='depth scale')
    parser.add_argument('--shift', type=float, default= 0.1378,  help='depth shift')
    # parser.add_argument('--preTrainedModel', type=str, default='weights/depth_weights/dpt_hybrid_nyu-2ce69ec_RESIDE_017_RESIDE_003_RESIDE_004.pt', help='pretrained DPT path')
    parser.add_argument('--preTrainedModel', type=str, default='weights/depth_weights/dpt_hybrid_kitti-cb926ef4_RESIDE_046.pt', help='pretrained DPT path')
    parser.add_argument('--preTrainedAirModel', type=str, default='weights/air_weights/Air_UNet_RESIDE_V0_epoch_16.pt', help='pretrained Air path')
    
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
    parser.add_argument('--stepLimit', type=int, default=50, help='Multi step limit')
    parser.add_argument('--eps', type=float, default=1e-12, help='Epsilon value for non zero calculating')
    return parser.parse_args()
    


def run(opt, model, airlight_model, metrics_module, loader):
    model.eval()
    airlight_model.eval()

    output_folder = 'output4_' + opt.dataset
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    f = open(output_folder + '/SOTS_Ours.csv','w', newline='')
    wr = csv.writer(f)

    pbar = tqdm(loader)
    for batch in pbar:
        hazy_images, clear_images, _, _, gt_betas, input_names = batch
        input_name = input_names[0][:-4]
        gt_beta = gt_betas[0]
        clear_image = util.denormalize(clear_images,opt.norm)[0].detach().cpu().numpy()
        
        with torch.no_grad():
            cur_hazy = hazy_images.to(opt.device)
            airlight = airlight_model.forward(cur_hazy)
            init_depth = model.forward(cur_hazy)
        airlight = util.air_denorm(opt.dataset,opt.norm,airlight)
        
        cur_depth = None
        sum_depth = torch.zeros_like(init_depth).to('cuda')

        entropy_max, ent_flag, ent_limit = 0, 0, 20
        for step in range(0, opt.stepLimit):
            with torch.no_grad():
                cur_depth = model.forward(cur_hazy)
            
            # New Depth Method
            diff_depth = cur_depth*step - sum_depth
            cur_hazy = util.denormalize(cur_hazy,opt.norm)
            trans = torch.exp((diff_depth+cur_depth)*opt.betaStep*-1)
            sum_depth = cur_depth * (step+1)
            
            # Prev Depth Method
            # cur_hazy = util.denormalize(cur_hazy,opt.norm)
            # trans = torch.exp(cur_depth*opt.betaStep*-1)
            
            prediction = (cur_hazy - airlight) / (trans + opt.eps) + airlight
            prediction = torch.clamp(prediction, 0, 1)

            entropy, _, _ = metrics_module.get_cur(cur_hazy[0].detach().cpu().numpy().transpose(1,2,0))
            if entropy_max < entropy:
                entropy_max = entropy
                ent_flag = 0
                best_dehazed = cur_hazy
            else:
                if ent_flag >= ent_limit:
                    break
                else:
                    ent_flag += 1
            
            cur_hazy = util.normalize(prediction[0].detach().cpu().numpy().transpose(1,2,0),opt.norm).unsqueeze(0).to(opt.device)
            
        dehaze_image = best_dehazed[0].detach().cpu().numpy()
        
        psnr = get_psnr(dehaze_image, clear_image)
        ssim = get_ssim(dehaze_image, clear_image).item()
        
        wr.writerow([input_name, gt_beta.item(), psnr, ssim, entropy])
        
        dehaze_image = (dehaze_image*255).astype(np.uint8).transpose(1,2,0)
        cv2.imwrite(f'{output_folder}/{input_name}.jpg', cv2.cvtColor(dehaze_image, cv2.COLOR_RGB2BGR))

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
        val_set   = NYU_Dataset(opt.dataRoot + '/train', **dataset_args)
    elif opt.dataset == 'RESIDE':
        val_set   = RESIDE_Dataset(opt.dataRoot + '/val',   **dataset_args)

    loader_args = dict(batch_size=1, num_workers=1, drop_last=False, shuffle=True)
    val_loader = DataLoader(dataset=val_set, **loader_args)
    metrics_module = Entropy_Module()
    
    run(opt, model, airlight_model, metrics_module, val_loader)