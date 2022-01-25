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

from dataset import *
from torch.utils.data import DataLoader

from Module_Airlight.Airlight_Module import Airlight_Module
from Module_Metrics.Entropy_Module import Entropy_Module
from Module_Metrics.NIQE_Module import NIQE_Module
from Module_Metrics.metrics import get_ssim, get_psnr
from util import misc, save_log, utils
from validate_NYU_depth import compute_errors


def get_args():
    parser = argparse.ArgumentParser()
    # dataset parameters
    parser.add_argument('--dataset', required=False, default='RESIDE-beta',  help='dataset name')
    parser.add_argument('--dataRoot', type=str, default='D:/data/Dense_Haze/train',  help='data file path')
    parser.add_argument('--norm', action='store_true',  help='Image Normalize flag')
    
    # learning parameters
    parser.add_argument('--seed', type=int, default=101, help='Random Seed')
    parser.add_argument('--batchSize_train', type=int, default=1, help='train dataloader input batch size')
    parser.add_argument('--batchSize_val', type=int, default=1, help='test dataloader input batch size')
    parser.add_argument('--imageSize_W', type=int, default=256, help='the width of the resized input image to network')
    parser.add_argument('--imageSize_H', type=int, default=256, help='the height of the resized input image to network')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--evalIter', type=int, default=10, help='interval for evaluating')
    parser.add_argument('--savePath', default='weights', help='folder to model checkpoints')
    parser.add_argument('--inputPath', default='input', help='input path')
    parser.add_argument('--outputPath', default='output_dehazed', help='dehazed output path')
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # model parameters
    parser.add_argument('--preTrainedModel', type=str, default='weights/dpt_hybrid_nyu-2ce69ec7.pt', help='pretrained DPT path')
    parser.add_argument('--backbone', type=str, default="vitb_rn50_384", help='DPT backbone')
    
    # test_stop_when_threshold parameters
    parser.add_argument('--save_log', action='store_true', help='log save flag')
    parser.add_argument('--saveORshow', type=str, default='show',  help='results show or save')
    parser.add_argument('--verbose', action='store_true', help='print log')
    parser.add_argument('--airlight_step_flag', action='store_true', help='flag of multi step airlight estimation')
    parser.add_argument('--betaStep', type=float, default=0.05, help='beta step')
    parser.add_argument('--stepLimit', type=int, default=250, help='Multi step limit')
    parser.add_argument('--metrics_module', type=str, default='Entropy_Module',  help='No Reference metrics method name')
    parser.add_argument('--metricsThreshold', type=float, default=0, help='Metrics threshold: Entropy(0.00), NIQUE(??)')
    parser.add_argument('--eps', type=float, default=1e-12, help='Epsilon value for non zero calculating')
    
    return parser.parse_args()
    


def test_stop_when_threshold(opt, model, test_loader, metrics_module):
    airlight_module = Airlight_Module()
    
    model.eval()
    
    pbar = tqdm(test_loader)
    for batch in pbar:
        images_dict = {}
        best_psnr_dict={}
        best_ssim_dict={}
        best_metrics_dict={}
        
        # Data Init
        hazy_images, clear_images , _, input_name = batch
        #hazy_images, clear_images, depth_images, input_name = batch
        #hazy_images, clear_images, airlight_images, depth_images,input_name = batch
        pbar.set_description(os.path.basename(input_name[0]))
        csv_log = []
        
        # Depth Estimation
        with torch.no_grad():
            hazy_images = hazy_images.to(opt.device)
            clear_images = clear_images.to(opt.device)
            _, init_depth = model.forward(hazy_images)
            _, clear_depth = model.forward(clear_images)
        

        init_depth = init_depth.clone().detach().cpu().numpy().transpose(1,2,0)
        images_dict['init_depth'] = init_depth
        
        init_hazy = hazy_images.clone().detach().cpu()
        images_dict['init_hazy'] = np.clip(utils.denormalize(init_hazy, norm=opt.norm)[0].numpy().transpose(1,2,0),0,1)
        
        clear = clear_images.clone().detach().cpu()
        images_dict['clear'] = np.clip(utils.denormalize(clear, norm=opt.norm)[0].numpy().transpose(1,2,0),0,1)
        
        clear_depth = clear_depth.detach().cpu().numpy().transpose(1,2,0)
        images_dict['clear_depth'] = clear_depth
        

        # Airlight Estimation
        images_dict['init_airlight'], _ = airlight_module.LLF(images_dict['init_hazy'])
        images_dict['clear_airlight'], _ = airlight_module.LLF(images_dict['clear'])
        
        
        # Multi-Step Depth Estimation and Dehazing
        metrics_module.reset(images_dict['init_hazy'])
        airlight = None
        beta = opt.betaStep
        beta_step = opt.betaStep
        cur_hazy, last_depth = init_hazy, images_dict['init_depth']
        best_psnr_dict['psnr'], best_ssim_dict['ssim'], best_metrics_dict['metrics'] = 0.0, 0.0, 0.0
        for step in range(1, opt.stepLimit + 1):
            # Depth Estimation
            with torch.no_grad():
                cur_hazy = cur_hazy.to(opt.device)
                _, cur_depth = model.forward(cur_hazy)
                
            print(f"beta = {beta:4.3f}")
            
            cur_hazy = utils.denormalize(cur_hazy, norm=opt.norm)[0].detach().cpu().numpy().transpose(1,2,0)
            
            cur_depth = cur_depth.detach().cpu().numpy().transpose(1,2,0)
            #cur_depth = np.minimum(cur_depth, last_depth)
            
            # Airlight Estimation
            if opt.airlight_step_flag == False:
                airlight = images_dict['init_airlight']
            else:
                airlight, _ = airlight_module.LLF(cur_hazy)
            
            # Transmission Map
            trans = np.exp(cur_depth * beta * -1)
            
            # Dehazing
            prediction = (images_dict['init_hazy'] - airlight) / (trans + opt.eps) + airlight
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
            if diff_metrics <= opt.metricsThreshold or opt.stepLimit == step:
            #if beta > 0.3:
            # if opt.stepLimit == step:            
            #if gt_beta+0.1 < beta or opt.stepLimit == step:
                trans = np.exp(images_dict['init_depth'] * best_metrics_dict['beta'] * -1)
                one_shot_prediction = (images_dict['init_hazy'] - images_dict['init_airlight']) / (trans+opt.eps) + images_dict['init_airlight']
                images_dict['one_shot_prediction'] = np.clip(one_shot_prediction, 0, 1)
            
                oneshot_psnr = get_psnr(images_dict['one_shot_prediction'], images_dict['clear'])
                oneshot_ssim = get_ssim(images_dict['one_shot_prediction'], images_dict['clear']).item()
                oneshot_metrics = metrics_module.get_cur(images_dict['one_shot_prediction'])
                
                with torch.no_grad():
                    images_dict['psnr_best_depth'] = model.forward(torch.Tensor(images_dict['psnr_best_prediction'].transpose(2,0,1)).unsqueeze(0).to(opt.device))[1].detach().cpu().numpy().transpose(1,2,0)
                    images_dict['ssim_best_depth'] = model.forward(torch.Tensor(images_dict['ssim_best_prediction'].transpose(2,0,1)).unsqueeze(0).to(opt.device))[1].detach().cpu().numpy().transpose(1,2,0)
                    images_dict['metrics_best_depth'] = model.forward(torch.Tensor(images_dict['metrics_best_prediction'].transpose(2,0,1)).unsqueeze(0).to(opt.device))[1].detach().cpu().numpy().transpose(1,2,0)
                    images_dict['one_shot_depth'] = model.forward(torch.Tensor(images_dict['one_shot_prediction'].transpose(2,0,1)).unsqueeze(0).to(opt.device))[1].detach().cpu().numpy().transpose(1,2,0)
                       
                if opt.verbose:
                    clear_metrics = metrics_module.get_cur(images_dict['clear'])
                    print(f'        | step |  beta  |  psnr  |  ssim  | metrics |')
                    print('-------------------------------------------------------------------------------------------------------------------------------------')
                    print(f" 1-shot : {   1:^4} | {best_metrics_dict['beta']:^5.4f} | {oneshot_psnr:^6.3f} | {oneshot_ssim:^5.4f} | {oneshot_metrics:^7.3f} |")
                    print(f"metrics : {best_metrics_dict['step']:^4} | {   best_metrics_dict['beta']:^5.4f} | {best_metrics_dict['psnr']:6.3f} | {best_metrics_dict['ssim']:^5.4f} | {best_metrics_dict['metrics']:^7.3f} |")
                    #print(f'   last : {step:^4} | {   beta:^5.4f} | {psnr:6.3f} | {ssim:^5.4f} | {metrics_module.last_value:^7.3f} |\
              #| {depth_error[0]:^7.4f} | {depth_error[1]:^5.4f} | {depth_error[2]:^6.4f} | {depth_error[3]:^8.4f} | {depth_error[4]:^6.4f} | {depth_error[5]:^6.4f} | {depth_error[6]:^6.4f} |')
                    print(f"   psnr : {best_psnr_dict['step']:^4} | {   best_psnr_dict['beta']:^5.4f} | {best_psnr_dict['psnr']:6.3f} | {best_psnr_dict['ssim']:^5.4f} | {best_psnr_dict['metrics']:^7.3f} |")
                    print(f"   ssim : {best_ssim_dict['step']:^4} | {   best_ssim_dict['beta']:^5.4f} | {best_ssim_dict['psnr']:6.3f} | {best_ssim_dict['ssim']:^5.4f} | {best_ssim_dict['metrics']:^7.3f} |")
                    print(f"  clear : {   0:^4} | {0:^5.4f} | {get_psnr(images_dict['clear'], images_dict['clear']):^6.3f} | {get_ssim(images_dict['clear'], images_dict['clear']):^5.4f} | {            clear_metrics:^7.3f} |")
                break
            
            beta += beta_step
            if opt.save_log:
                csv_log.append([step, beta_step, metrics_module.cur_value, diff_metrics, psnr, ssim])
            
            os.system("cls")
                
        if opt.save_log:
            save_log.write_csv(opt.dataRoot, opt.metrics_module, input_name[0], csv_log)
                        
        if opt.saveORshow != '':
            misc.all_results_saveORshow(opt.dataRoot, input_name[0], 
                                        opt.airlight_step_flag,
                                        images_dict, opt.saveORshow)
            

if __name__ == '__main__':
    opt = get_args()
    opt.norm = True
    
    # opt.seed = random.randint(1, 10000)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    print("=========| Option |=========\n", opt)
    
    model = DPTDepthModel(
        path = opt.preTrainedModel,
        scale=0.000305, shift=0.1378, invert=True,
        backbone=opt.backbone,
        non_negative=True,
        enable_attention_hooks=False,
    )
    
    model = model.to(memory_format=torch.channels_last)
    
    model.to(opt.device)
    
    # opt.dataRoot = 'C:/Users/IIPL/Desktop/data/RESIDE_beta/train'
    # opt.dataRoot = 'D:/data/RESIDE_beta_sample/train'
    #opt.dataRoot = 'D:/data/RESIDE_beta/'
    #opt.dataRoot = 'D:/data/D_Hazy_NYU/'
    opt.dataRoot = 'D:/data/Dense_Haze/'
    dataset_test =NTIRE_Dataset.Dense_Haze_Dataset(opt.dataRoot, [opt.imageSize_W, opt.imageSize_H], split='train', printName=False, returnName=True, norm=opt.norm)
    loader_test = DataLoader(dataset=dataset_test, batch_size=opt.batchSize_val,
                             num_workers=1, drop_last=False, shuffle=True)
    
    # opt.metrics_module = 'NIQE_Module'
    metrics_module = locals()[opt.metrics_module]()
    test_stop_when_threshold(opt, model, loader_test, metrics_module)