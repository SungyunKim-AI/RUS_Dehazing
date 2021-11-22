# User warnings ignore
import warnings
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

from dataset import NTIRE_Dataset, RESIDE_Dataset
from torch.utils.data import DataLoader

from Module_Airlight.Airlight_Module import Airlight_Module
from Module_Metrics.Entropy_Module import Entropy_Module
from Module_Metrics.NIQE_Module import NIQE_Module
from Module_Metrics.metrics import get_ssim, get_psnr
from util import misc, save_log, utils



def get_args():
    parser = argparse.ArgumentParser()
    # dataset parameters
    parser.add_argument('--dataset', required=False, default='RESIDE-beta',  help='dataset name')
    parser.add_argument('--dataRoot', type=str, default='D:/data/Dense_Haze/train',  help='data file path')
    parser.add_argument('--norm', type=bool, default=False,  help='Image Normalize flag')
    
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
    parser.add_argument('--preTrainedModel', type=str, default='weights/dpt_hybrid-midas-501f0c75.pt', help='pretrained DPT path')
    parser.add_argument('--backbone', type=str, default="vitb_rn50_384", help='DPT backbone')
    
    # test_stop_when_threshold parameters
    parser.add_argument('--save_log', type=bool, default=True, help='log save flag')
    parser.add_argument('--saveORshow', type=str, default='save',  help='results show or save')
    parser.add_argument('--verbose', type=bool, default=True, help='print log')
    parser.add_argument('--psnr_ssim_tracking', type=bool, default=True, help='best psnr & ssim traking')
    parser.add_argument('--one_shot_flag', type=bool, default=False, help='flag of One shot dehazing')
    parser.add_argument('--airlight_step_flag', type=bool, default=False, help='flag of multi step airlight estimation')
    parser.add_argument('--betaStep', type=float, default=0.005, help='beta step')
    parser.add_argument('--stepLimit', type=int, default=250, help='Multi step limit')
    parser.add_argument('--metrics_module', type=str, default='Entropy_Module',  help='No Reference metrics method name')
    parser.add_argument('--metricsThreshold', type=float, default=0.011, help='Metrics threshold: Entropy(0.001), NIQUE(-1.11)')
    parser.add_argument('--eps', type=float, default=1e-12, help='Epsilon value for non zero calculating')
    
    return parser.parse_args()
    


def test_stop_when_threshold(opt, model, test_loader, metrics_module):
    airlight_module = Airlight_Module()
    
    model.eval()
    
    pbar = tqdm(test_loader)
    for batch in pbar:
        images_dict = {}
        # Data Init
        if len(batch) == 3:
            hazy_images, clear_images, input_name = batch
            airlight_images = None
        else:
            hazy_images, clear_images, airlight_images, input_name = batch
        pbar.set_description(os.path.basename(input_name[0]))
        csv_log = []
        
        # Depth Estimation
        with torch.no_grad():
            hazy_images = hazy_images.to(opt.device)
            clear_images = clear_images.to(opt.device)
            _, init_depth = model.forward(hazy_images)
            _, clear_depth = model.forward(clear_images)
        
        init_depth_ = init_depth.clone().detach().cpu().numpy().transpose(1,2,0)
        images_dict['init_depth'] = utils.depth_norm(init_depth_)
        
        init_hazy = hazy_images.clone().detach().cpu()
        images_dict['init_hazy'] = utils.denormalize(init_hazy, norm=opt.norm)[0].numpy().transpose(1,2,0)
        
        clear = clear_images.clone().detach().cpu()
        images_dict['clear'] = utils.denormalize(clear, norm=opt.norm)[0].numpy().transpose(1,2,0)
        
        clear_depth = clear_depth.detach().cpu().numpy().transpose(1,2,0)
        images_dict['clear_depth'] = utils.depth_norm(clear_depth)
        

        # Airlight Estimation
        images_dict['init_airlight'], _ = airlight_module.LLF(images_dict['init_hazy'])
        images_dict['clear_airlight'], _ = airlight_module.LLF(images_dict['clear'])
        
        
        # Multi-Step Depth Estimation and Dehazing
        metrics_module.reset(images_dict['init_hazy'])
        airlight = None
        beta = opt.betaStep
        # opt.stepLimit = int(utils.get_GT_beta(input_name) / opt.betaStep) + 20
        cur_hazy, last_depth = init_hazy, images_dict['init_depth']
        best_psnr, best_ssim = 0.0, 0.0
        for step in range(1, opt.stepLimit + 1):
            # Depth Estimation
            with torch.no_grad():
                cur_hazy = cur_hazy.to(opt.device)
                _, cur_depth = model.forward(cur_hazy)
            
            cur_hazy = utils.denormalize(cur_hazy, norm=opt.norm)[0].detach().cpu().numpy().transpose(1,2,0)
            
            cur_depth = cur_depth.detach().cpu().numpy().transpose(1,2,0)
            cur_depth = utils.depth_norm(cur_depth)
            cur_depth = np.minimum(cur_depth, last_depth)
            
            # Airlight Estimation
            if opt.airlight_step_flag == False:
                airlight = images_dict['init_airlight']
            else:
                airlight, _ = airlight_module.LLF(cur_hazy)
            
            # Transmission Map
            trans = np.exp(cur_depth * opt.betaStep * -1)
            
            # Dehazing
            prediction = (cur_hazy - airlight) / (trans + opt.eps) + airlight
            prediction = np.clip(prediction, 0, 1)
            
            # Calculate Metrics
            diff_metrics = metrics_module.get_diff(prediction)
            psnr = get_psnr(prediction, images_dict['clear'])
            ssim = get_ssim(prediction, images_dict['clear']).item()
            
            if opt.psnr_ssim_tracking:
                if best_psnr < psnr:
                    images_dict['psnr_best_prediction'] = prediction
                    images_dict['psnr_best_depth'] = cur_depth
                    best_psnr = psnr
                
                if best_ssim < ssim:
                    images_dict['ssim_best_prediction'] = prediction
                    images_dict['ssim_best_depth'] = cur_depth
                    best_ssim = ssim
            
            if opt.save_log:
                csv_log.append([step, opt.betaStep, metrics_module.cur_value, diff_metrics, psnr, ssim])
            
            # Stop Condition
            if diff_metrics <= opt.metricsThreshold or opt.stepLimit == step:
            # if opt.stepLimit == step:
                images_dict['metrics_best_prediction'] = prediction
                images_dict['metrics_best_depth'] = cur_depth
            
                if opt.verbose:
                    gt_beta = utils.get_GT_beta(input_name[0])
                    clear_metrics = metrics_module.get_cur(images_dict['clear'])
                    print(f'last_step    = {step}')
                    print(f'last_beta    = {beta}({gt_beta})')
                    print(f'last_psnr    = {psnr}')
                    print(f'last_ssim    = {ssim}')
                    print(f'last_metrics = {metrics_module.last_value}')
                    print(f'clear_metrics  = {clear_metrics}')
                break
            
            # Set Next Step
            beta += opt.betaStep
            cur_hazy = torch.Tensor(prediction.transpose(2,0,1)).unsqueeze(0)
            last_depth = cur_depth.copy()
                
        if opt.save_log:
            save_log.write_csv(opt.dataRoot, opt.metrics_module, input_name[0], csv_log)
        
        # One-Shot Dehazing
        if opt.one_shot_flag:
            beta_gt = utils.get_GT_beta(input_name[0])

            trans = np.exp(init_depth_ * beta_gt * -1)
            one_shot_prediction = (images_dict['init_hazy'] - images_dict['init_airlight']) / (trans+opt.eps) + images_dict['init_airlight']
            images_dict['one_shot_prediction'] = np.clip(one_shot_prediction, 0, 1)
            
            oneshot_psnr = get_psnr(images_dict['one_shot_prediction'], images_dict['clear'])
            oneshot_ssim = get_ssim(images_dict['one_shot_prediction'], images_dict['clear']).item()
            oneshot_metrics = metrics_module.get_cur(images_dict['one_shot_prediction'])
            if opt.verbose:
                print(f'one-shot: beta = {beta}, psnr = {oneshot_psnr}, ssim={oneshot_ssim}, metrics={oneshot_metrics}')
                
        if opt.saveORshow != '':
            misc.all_results_saveORshow(opt.dataRoot, input_name[0], 
                                        opt.airlight_step_flag, opt.one_shot_flag, 
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
        scale=1, shift=0, invert=False,
        backbone=opt.backbone,
        non_negative=True,
        enable_attention_hooks=False,
    )
    
    model = model.to(memory_format=torch.channels_last)
    
    model.to(opt.device)
    
    # opt.dataRoot = 'C:/Users/IIPL/Desktop/data/RESIDE_beta/train'
    # opt.dataRoot = 'D:/data/RESIDE_beta_sample/train'
    opt.dataRoot = 'D:/data/RESIDE_beta_sample_100/'
    dataset_test = RESIDE_Dataset.RESIDE_Beta_Dataset(opt.dataRoot, [opt.imageSize_W, opt.imageSize_H], split='train', printName=False, returnName=True, norm=opt.norm)
    loader_test = DataLoader(dataset=dataset_test, batch_size=opt.batchSize_val,
                             num_workers=4, drop_last=False, shuffle=False)
    
    # opt.metrics_module = 'NIQE_Module'
    metrics_module = locals()[opt.metrics_module]()
    test_stop_when_threshold(opt, model, loader_test, metrics_module)