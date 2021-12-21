# User warnings ignore
from genericpath import exists
import warnings

from numpy.lib.function_base import diff
from numpy.lib.type_check import imag

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


def get_args():
    parser = argparse.ArgumentParser()
    # dataset parameters
    parser.add_argument('--dataset', required=False, default='RESIDE_beta',  help='dataset name')
    parser.add_argument('--dataRoot', type=str, default='D:/data/RESIDE_beta/',  help='data file path')
    parser.add_argument('--norm', action='store_true',  help='Image Normalize flag')
    
    # learning parameters
    parser.add_argument('--seed', type=int, default=101, help='Random Seed')
    parser.add_argument('--imageSize_W', type=int, default=256, help='the width of the resized input image to network')
    parser.add_argument('--imageSize_H', type=int, default=256, help='the height of the resized input image to network')
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # model parameters
    parser.add_argument('--preTrainedModel', type=str, default='weights/dpt_hybrid_nyu-2ce69ec7.pt', help='pretrained DPT path')
    parser.add_argument('--backbone', type=str, default="vitb_rn50_384", help='DPT backbone')
    
    # test_stop_when_threshold parameters
    parser.add_argument('--save_log', action='store_true', help='log save flag')
    parser.add_argument('--saveORshow', type=str, default='show',  help='results show or save')
    parser.add_argument('--verbose', action='store_true', help='print log')
    parser.add_argument('--betaStep', type=float, default=0.005, help='beta step')
    parser.add_argument('--stepLimit', type=int, default=250, help='Multi step limit')
    parser.add_argument('--metricsThreshold', type=float, default=0, help='Metrics threshold: Entropy(0.00), NIQUE(??)')
    parser.add_argument('--eps', type=float, default=1e-12, help='Epsilon value for non zero calculating')
    
    return parser.parse_args()
    


def test_stop_when_threshold(opt, model, airlight_model, val_loader, metrics_module):
    model.eval()
    airlight_model.eval()

    pbar = tqdm(val_loader)
    psnr_psnr_list=[]
    metric_psnr_list=[]
    for batch in pbar:
        images_dict = {}
        best_psnr_dict={}
        best_ssim_dict={}
        best_metrics_dict={}
        
        # Data Init
        hazy_images, clear_images , depth_images, gt_airlights, gt_betas, input_name = batch
        pbar.set_description(os.path.basename(input_name[0]))
        csv_log = []
        
        # Depth Estimation
        with torch.no_grad():
            hazy_images = hazy_images.to(opt.device)
            clear_images = clear_images.to(opt.device)
            depth_images = depth_images.to(opt.device)
            init_depth = model.forward(hazy_images)
            clear_depth = model.forward(clear_images)
            init_airlight = airlight_model.forward(hazy_images)

        init_hazy = hazy_images.detach().cpu()
        images_dict['init_hazy'] = np.clip(util.denormalize(init_hazy, norm=opt.norm).numpy()[0].transpose(1,2,0),0,1)
        
        gt_depth = depth_images[0].detach().cpu().numpy().transpose(1,2,0)
        images_dict['gt_depth'] = gt_depth

        gt_airlight = gt_airlights[0].detach().cpu()
        gt_airlight = util.air_denorm(opt.dataset, opt.norm, gt_airlight)
        images_dict['gt_airlight'] = np.full_like(gt_depth, gt_airlight)

        gt_beta = gt_betas[0].item()

        init_depth = init_depth[0].detach().cpu().numpy().transpose(1,2,0)
        images_dict['init_depth'] = init_depth
                
        init_airlight = init_airlight[0].detach().cpu()
        init_airlight = util.air_denorm(opt.dataset, opt.norm, init_airlight)
        images_dict['init_airlight'] = np.full_like(gt_depth, init_airlight)
        print(init_airlight)
        
        clear = clear_images.clone().detach().cpu()
        images_dict['clear'] = np.clip(util.denormalize(clear, norm=opt.norm)[0].numpy().transpose(1,2,0),0,1)
        
        clear_depth = clear_depth[0].detach().cpu().numpy().transpose(1,2,0)
        images_dict['clear_depth'] = clear_depth
        
        # Multi-Step Depth Estimation and Dehazing
        metrics_module.reset(images_dict['init_hazy'])
        beta = opt.betaStep
        beta_step = opt.betaStep
        cur_hazy, last_depth = init_hazy, images_dict['init_depth']
        best_psnr_dict['psnr'], best_ssim_dict['ssim'], best_metrics_dict['metrics'] = 0.0, 0.0, 0.0
        for step in range(1, opt.stepLimit + 1):
            # Depth Estimation
            with torch.no_grad():
                cur_hazy = cur_hazy.to(opt.device)
                cur_depth = model.forward(cur_hazy)
            
            cur_hazy = util.denormalize(cur_hazy, norm=opt.norm)[0].detach().cpu().numpy().transpose(1,2,0)
            
            cur_depth = cur_depth[0].detach().cpu().numpy().transpose(1,2,0)
            #cur_depth = np.minimum(cur_depth, last_depth)
            
            # Transmission Map
            trans = np.exp(cur_depth * beta * -1)
            
            # Dehazing
            prediction = (images_dict['init_hazy'] -  images_dict['init_airlight']) / (trans + opt.eps) +  images_dict['init_airlight']
            prediction = np.clip(prediction, 0, 1)
            
            # Calculate Metrics
            diff_metrics = metrics_module.get_diff(prediction)
            psnr = get_psnr(prediction, images_dict['clear'])
            ssim = get_ssim(prediction, images_dict['clear']).item()
            
            if best_psnr_dict['psnr'] < psnr:
                images_dict['psnr_best_prediction'] = prediction
                best_psnr_dict['step'] = step
                best_psnr_dict['beta'] = beta
                best_psnr_dict['psnr'] = psnr
                best_psnr_dict['ssim'] = ssim
                best_psnr_dict['metrics'] = metrics_module.cur_value
            
            if best_ssim_dict['ssim'] < ssim:
                images_dict['ssim_best_prediction'] = prediction
                best_ssim_dict['step'] = step
                best_ssim_dict['beta'] = beta
                best_ssim_dict['psnr'] = psnr
                best_ssim_dict['ssim'] = ssim
                best_ssim_dict['metrics'] = metrics_module.cur_value
                    
            if best_metrics_dict['metrics'] < metrics_module.cur_value:
                images_dict['metrics_best_prediction'] = prediction
                best_metrics_dict['step'] = step
                best_metrics_dict['beta'] = beta
                best_metrics_dict['psnr'] = psnr
                best_metrics_dict['ssim'] = ssim
                best_metrics_dict['metrics'] = metrics_module.cur_value


            # Set Next Step
            cur_hazy = torch.Tensor(prediction.transpose(2,0,1)).unsqueeze(0)
            
            # Stop Condition    
            #if diff_metrics <= opt.metricsThreshold or opt.stepLimit == step:
            # if opt.stepLimit == step:            
            if gt_beta+0.1 < beta or opt.stepLimit == step:
                if beta_step!=0:
                    beta_step = 0
                    continue
                
                trans = np.exp(images_dict['init_depth'] * best_metrics_dict['beta'] * -1)
                one_shot_prediction = (images_dict['init_hazy'] - images_dict['init_airlight']) / (trans+opt.eps) + images_dict['init_airlight']
                images_dict['one_shot_prediction'] = np.clip(one_shot_prediction, 0, 1)
                
                with torch.no_grad():
                    images_dict['psnr_best_depth'] = model.forward(torch.Tensor(images_dict['psnr_best_prediction'].transpose(2,0,1)).unsqueeze(0).to(opt.device))[0].detach().cpu().numpy().transpose(1,2,0)
                    images_dict['ssim_best_depth'] = model.forward(torch.Tensor(images_dict['ssim_best_prediction'].transpose(2,0,1)).unsqueeze(0).to(opt.device))[0].detach().cpu().numpy().transpose(1,2,0)
                    images_dict['metrics_best_depth'] = model.forward(torch.Tensor(images_dict['metrics_best_prediction'].transpose(2,0,1)).unsqueeze(0).to(opt.device))[0].detach().cpu().numpy().transpose(1,2,0)
                    images_dict['one_shot_depth'] = model.forward(torch.Tensor(images_dict['one_shot_prediction'].transpose(2,0,1)).unsqueeze(0).to(opt.device))[0].detach().cpu().numpy().transpose(1,2,0)
                    
                    best_psnr_depth_error = compute_errors(images_dict['gt_depth'], images_dict['psnr_best_depth'])
                    best_ssim_depth_error = compute_errors(images_dict['gt_depth'], images_dict['ssim_best_depth'])
                    best_metrics_depth_error = compute_errors(images_dict['gt_depth'], images_dict['metrics_best_depth'])
                    one_shot_depth_error = compute_errors(images_dict['gt_depth'], images_dict['one_shot_depth'])
                    clear_depth_error = compute_errors(images_dict['gt_depth'], clear_depth)
                    init_depth_error = compute_errors(images_dict['gt_depth'], images_dict['init_depth'])
                
                gt_trans = np.exp(images_dict['gt_depth'] * gt_beta * -1)
                gt_prediction = (images_dict['init_hazy'] - images_dict['gt_airlight']) / (gt_trans+opt.eps) + images_dict['gt_airlight']
                images_dict['gt_prediction'] = np.clip(gt_prediction, 0, 1)
            
                oneshot_psnr = get_psnr(images_dict['one_shot_prediction'], images_dict['clear'])
                oneshot_ssim = get_ssim(images_dict['one_shot_prediction'], images_dict['clear']).item()
                oneshot_metrics = metrics_module.get_cur(images_dict['one_shot_prediction'])

                psnr_psnr_list.append(best_psnr_dict['psnr'])
                metric_psnr_list.append(best_metrics_dict['psnr'])

                if opt.verbose:
                    clear_metrics = metrics_module.get_cur(images_dict['clear'])
                    print(f'        | step |  beta  |  psnr  |  ssim  | metrics | depth_error: | abs_rel | sq_rel |  rmse  | rmse_log |   a1   |   a2   |   a3   |')
                    print('-------------------------------------------------------------------------------------------------------------------------------------')
                    print(f"   init : \t\t\t\t\t\t\t   | {init_depth_error[0]:^7.4f} | {init_depth_error[1]:^5.4f} | {init_depth_error[2]:^6.4f} | {init_depth_error[3]:^8.4f} | {init_depth_error[4]:^6.4f} | {init_depth_error[5]:^6.4f} | {init_depth_error[6]:^6.4f} |")
                    print(f" 1-shot : {   1:^4} | {best_metrics_dict['beta']:^5.4f} | {oneshot_psnr:^6.3f} | {oneshot_ssim:^5.4f} | {oneshot_metrics:^7.3f} |\
              | {one_shot_depth_error[0]:^7.4f} | {one_shot_depth_error[1]:^5.4f} | {one_shot_depth_error[2]:^6.4f} | {one_shot_depth_error[3]:^8.4f} | {one_shot_depth_error[4]:^6.4f} | {one_shot_depth_error[5]:^6.4f} | {one_shot_depth_error[6]:^6.4f} |")
                    print(f"metrics : {best_metrics_dict['step']:^4} | {   best_metrics_dict['beta']:^5.4f} | {best_metrics_dict['psnr']:6.3f} | {best_metrics_dict['ssim']:^5.4f} | {best_metrics_dict['metrics']:^7.3f} |\
              | {best_metrics_depth_error[0]:^7.4f} | {best_metrics_depth_error[1]:^5.4f} | {best_metrics_depth_error[2]:^6.4f} | {best_metrics_depth_error[3]:^8.4f} | {best_metrics_depth_error[4]:^6.4f} | {best_metrics_depth_error[5]:^6.4f} | {best_metrics_depth_error[6]:^6.4f} |")
                    #print(f'   last : {step:^4} | {   beta:^5.4f} | {psnr:6.3f} | {ssim:^5.4f} | {metrics_module.last_value:^7.3f} |\
              #| {depth_error[0]:^7.4f} | {depth_error[1]:^5.4f} | {depth_error[2]:^6.4f} | {depth_error[3]:^8.4f} | {depth_error[4]:^6.4f} | {depth_error[5]:^6.4f} | {depth_error[6]:^6.4f} |')
                    print(f"   psnr : {best_psnr_dict['step']:^4} | {   best_psnr_dict['beta']:^5.4f} | {best_psnr_dict['psnr']:6.3f} | {best_psnr_dict['ssim']:^5.4f} | {best_psnr_dict['metrics']:^7.3f} |\
              | {best_psnr_depth_error[0]:^7.4f} | {best_psnr_depth_error[1]:^5.4f} | {best_psnr_depth_error[2]:^6.4f} | {best_psnr_depth_error[3]:^8.4f} | {best_psnr_depth_error[4]:^6.4f} | {best_psnr_depth_error[5]:^6.4f} | {best_psnr_depth_error[6]:^6.4f} |")
                    print(f"   ssim : {best_ssim_dict['step']:^4} | {   best_ssim_dict['beta']:^5.4f} | {best_ssim_dict['psnr']:6.3f} | {best_ssim_dict['ssim']:^5.4f} | {best_ssim_dict['metrics']:^7.3f} |\
              | {best_ssim_depth_error[0]:^7.4f} | {best_ssim_depth_error[1]:^5.4f} | {best_ssim_depth_error[2]:^6.4f} | {best_ssim_depth_error[3]:^8.4f} | {best_ssim_depth_error[4]:^6.4f} | {best_ssim_depth_error[5]:^6.4f} | {best_ssim_depth_error[6]:^6.4f} |")
                    print(f"  clear : {   0:^4} | {gt_beta:^5.4f} | {get_psnr(images_dict['clear'], images_dict['clear']):^6.3f} | {get_ssim(images_dict['clear'], images_dict['clear']):^5.4f} | {            clear_metrics:^7.3f} |\
              | {clear_depth_error[0]:^7.4f} | {clear_depth_error[1]:^5.4f} | {clear_depth_error[2]:^6.4f} | {clear_depth_error[3]:^8.4f} | {clear_depth_error[4]:^6.4f} | {clear_depth_error[5]:^6.4f} | {clear_depth_error[6]:^6.4f} |")
                break
            
            beta += beta_step
            if opt.save_log:
                csv_log.append([step, beta_step, metrics_module.cur_value, diff_metrics, psnr, ssim])
                
        if opt.save_log:
            save_log.write_csv(opt.dataRoot, opt.metrics_module, input_name[0], csv_log)
                        
        if opt.saveORshow != '':
            misc.all_results_saveORshow(opt.dataRoot, input_name[0],
                                        images_dict, opt.saveORshow)

    psnr_psnr = np.array(psnr_psnr_list)
    metric_psnr = np.array(metric_psnr_list)

    print(psnr_psnr.mean(), psnr_psnr.std())
    print(metric_psnr.mean(), metric_psnr.std())
            

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
        path = 'weights/depth_weights/dpt_hybrid_nyu-2ce69ec7_reside_haze_002.pt',
        scale=0.000150, shift=0.1378, invert=True,  #NYU scale = 0.000305, RESIDE_beta sclae = 0.000150
        backbone=opt.backbone,
        non_negative=True,
        enable_attention_hooks=False,
    )
    model = model.to(memory_format=torch.channels_last)
    model.to(opt.device)
    
        
    airlight_model = UNet([opt.imageSize_W, opt.imageSize_H], in_channels=3, out_channels=1, bilinear=True)
    checkpoint = torch.load(f'weights/air_weights/Air_UNet_RESIDE_1D.pt')
    airlight_model.load_state_dict(checkpoint['model_state_dict'])
    airlight_model.to(opt.device)


    dataset_args = dict(img_size=[opt.imageSize_W, opt.imageSize_H], norm=opt.norm)
    if opt.dataset == 'NYU':
        val_set   = NYU_Dataset(opt.dataRoot + '/val', **dataset_args)
    elif opt.dataset == 'RESIDE_beta':
        val_set   = RESIDE_Beta_Dataset(opt.dataRoot + '/val',   **dataset_args)

    loader_args = dict(batch_size=1, num_workers=1, drop_last=False, shuffle=True)
    val_loader = DataLoader(dataset=val_set, **loader_args)
    
    metrics_module = Entropy_Module()
    test_stop_when_threshold(opt, model, airlight_model, val_loader, metrics_module)