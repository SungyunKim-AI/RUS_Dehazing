# User warnings ignore
import warnings

from numpy.lib.function_base import diff
warnings.filterwarnings("ignore")

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import argparse
import cv2
import numpy as np
import random
from tqdm import tqdm

import torch
from dpt.models import DPTDepthModel

from dataset import NYU_Dataset
from torch.utils.data import DataLoader

from Module_Metrics.metrics import get_ssim, get_psnr
from util import misc, save_log, utils



def get_args():
    parser = argparse.ArgumentParser()
    # dataset parameters
    parser.add_argument('--dataset', required=False, default='NYU_dataset',  help='dataset name')
    parser.add_argument('--dataRoot', type=str, default='D:/data/Dense_Haze/train',  help='data file path')
    parser.add_argument('--norm', type=bool, default=True,  help='Image Normalize flag')
    
    # learning parameters
    parser.add_argument('--seed', type=int, default=101, help='Random Seed')
    parser.add_argument('--batchSize_val', type=int, default=1, help='test dataloader input batch size')
    parser.add_argument('--imageSize_W', type=int, default=640, help='the width of the resized input image to network')
    parser.add_argument('--imageSize_H', type=int, default=480, help='the height of the resized input image to network')
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # model parameters
    parser.add_argument('--preTrainedModel', type=str, default='weights/dpt_hybrid-midas-501f0c75.pt', help='pretrained DPT path')
    parser.add_argument('--backbone', type=str, default="vitb_rn50_384", help='DPT backbone')
    
    # test_stop_when_threshold parameters
    parser.add_argument('--save_log', type=bool, default=True, help='log save flag')
    parser.add_argument('--saveORshow', type=str, default='save',  help='results show or save')
    parser.add_argument('--verbose', type=bool, default=True, help='print log')
    parser.add_argument('--betaStep', type=float, default=0.005, help='beta step')
    parser.add_argument('--stepLimit', type=int, default=250, help='Multi step limit')
    parser.add_argument('--eps', type=float, default=1e-12, help='Epsilon value for non zero calculating')
    
    return parser.parse_args()
    
def tensor2numpy(x):
    return x.clone().detach().cpu().numpy()

def test_stop_when_threshold(opt, model, test_loader):
    
    model.eval()
    
    pbar = tqdm(test_loader)
    for batch in pbar:
        images_dict = {}        # results images dictionary
        csv_log = []            # result log save to csv
        
        # Data Init
        hazy_image, clear_image, airlight, GT_depth, input_name \
            = batch[0], batch[1], batch[3].numpy(), batch[2].numpy(), batch[4][0]
        pbar.set_description(os.path.basename(input_name))

        
        # Depth Estimation
        with torch.no_grad():
            hazy_image = hazy_image.to(opt.device)
            clear_image = clear_image.to(opt.device)
            _, init_depth = model.forward(hazy_image)
            _, clear_depth = model.forward(clear_image)
        

        images_dict['init_hazy'] = tensor2numpy(hazy_image)[0]      # 3x480x640
        images_dict['clear'] = tensor2numpy(clear_image)[0]         # 3x480x640
        images_dict['init_depth'] = tensor2numpy(init_depth)        # 1x480x640
        images_dict['clear_depth'] = tensor2numpy(clear_depth)      # 1x480x640
        images_dict['airlight'] = airlight.transpose(0, 2, 1)       # 1x480x640
        images_dict['GT_depth'] = GT_depth.transpose(0, 2, 1)
        
        
        # Multi-Step Depth Estimation and Dehazing
        beta = opt.betaStep
        cur_hazy = hazy_image
        best_psnr, best_ssim = 0.0, 0.0
        for step in range(1, opt.stepLimit + 1):
            # Depth Estimation
            with torch.no_grad():
                cur_hazy = cur_hazy.to(opt.device)
                _, cur_depth = model.forward(cur_hazy)
            
            cur_depth = tensor2numpy(cur_depth)[0]
            
            # Transmission Map
            trans = np.exp(cur_depth * beta * -1)
            
            # Dehazing
            prediction = (images_dict['init_hazy'] - images_dict['airlight']) / (trans + opt.eps) + images_dict['airlight']
            prediction = np.clip(prediction, -1, 1)
            
            # Calculate Metrics
            psnr = get_psnr(prediction, images_dict['clear'])
            ssim = get_ssim(prediction, images_dict['clear']).item()
            
            
            if best_psnr < psnr:
                best_psnr = psnr
                
                if best_ssim < ssim:
                    best_ssim = ssim
                
                if opt.save_log:
                    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 \
                        = utils.compute_errors(images_dict['GT_depth'], cur_depth)
                    csv_log.append([step, beta, best_psnr, best_ssim, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3])
            else:
                # Final Depth Estimation
                with torch.no_grad():
                    dehazed = cur_hazy.to(opt.device)
                    _, final_depth = model.forward(dehazed)

                images_dict['final_depth'] = tensor2numpy(final_depth)[0]
                images_dict['psnr_best_prediction'] = tensor2numpy(cur_hazy)[0]

                if opt.save_log:
                    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 \
                        = utils.compute_errors(images_dict['GT_depth'], images_dict['final_depth'])
                    csv_log.append([step, beta, best_psnr, best_ssim, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3])

                if opt.verbose:
                    gt_beta = utils.get_GT_beta(input_name)
                    print(f'last_step  = {step}')
                    print(f'last_beta  = {beta}({gt_beta})')
                    print(f'last_psnr  = {best_psnr}')
                    print(f'last_ssim  = {best_ssim}')
                break
            
            # Set Next Step
            beta += opt.betaStep
            cur_hazy = torch.Tensor(prediction).unsqueeze(0)

                
        if opt.save_log:
            save_log.write_csv_depth_err(opt.dataRoot, input_name, csv_log)
        
                
        if opt.saveORshow != '':
            misc.depth_results_saveORshow(opt.dataRoot, input_name, 
                                          [opt.imageSize_W, opt.imageSize_H],
                                          images_dict, opt.saveORshow)
            

if __name__ == '__main__':
    opt = get_args()
    
    # opt.seed = random.randint(1, 10000)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    print("=========| Option |=========\n", opt)
    
    model = DPTDepthModel(
        path = opt.preTrainedModel,
        scale=0.00030, shift=0.1378, invert=True,
        backbone=opt.backbone,
        non_negative=True,
        enable_attention_hooks=False,
    )
    
    model = model.to(memory_format=torch.channels_last)
    
    model.to(opt.device)
    
    # opt.dataRoot = 'C:/Users/IIPL/Desktop/data/RESIDE_beta/train'
    # opt.dataRoot = 'D:/data/RESIDE_beta_sample/train'
    opt.dataRoot = 'D:/data/NYU/'
    dataset_test = NYU_Dataset.NYU_Dataset(opt.dataRoot, [opt.imageSize_W, opt.imageSize_H], printName=False, returnName=True, norm=opt.norm)
    loader_test = DataLoader(dataset=dataset_test, batch_size=opt.batchSize_val,
                             num_workers=1, drop_last=False, shuffle=False)
    
    test_stop_when_threshold(opt, model, loader_test)