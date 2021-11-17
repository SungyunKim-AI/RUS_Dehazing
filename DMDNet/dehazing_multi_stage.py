# User warnings ignore
import warnings
warnings.filterwarnings("ignore")

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
from Module_Metrics.metrics import ssim, psnr
from util import misc, save_log, utils



def get_args():
    parser = argparse.ArgumentParser()
    # dataset parameters
    parser.add_argument('--dataset', required=False, default='pix2pix',  help='')
    parser.add_argument('--dataRoot', type=str, default='D:/data/Dense_Haze/train',  help='data file path')
    
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
    parser.add_argument('--one_shot', type=bool, default=True, help='flag of One shot dehazing')
    parser.add_argument('--result_show', type=bool, default=True, help='result images display flag')
    parser.add_argument('--save_log', type=bool, default=False, help='log save flag')
    parser.add_argument('--airlight_step_flag', type=bool, default=False, help='flag of multi step airlight estimation')
    parser.add_argument('--betaStep', type=float, default=0.001, help='beta step')
    parser.add_argument('--stepLimit', type=int, default=1000, help='Multi step limit')
    parser.add_argument('--metrics_module', type=str, default='Entropy_Module',  help='No Reference metrics method name')
    parser.add_argument('--metricsThreshold', type=float, default=0.001, help='Metrics different threshold')
    parser.add_argument('--eps', type=float, default=1e-8, help='Epsilon value for non zero calculating')
    
    return parser.parse_args()


def test_stop_when_threshold(opt, model, test_loader):
    airlight_module = Airlight_Module()
    
    model.eval()
    psnr_sum, ssim_sum = 0.0, 0.0
    
    for batch in tqdm(test_loader):
        # Data Init
        if len(batch) == 2:
            hazy_images, clear_images = batch
            airlight_images = None
        else:
            hazy_images, clear_images, airlight_images, input_name = batch
        csv_log = []
        
        
        # Depth Estimation
        with torch.no_grad():
            hazy_images = hazy_images.to(opt.device)
            clear_images = clear_images.to(opt.device)
            _, init_depth = model.forward(hazy_images)
            _, init_clear_depth = model.forward(clear_images)
        
        init_hazy = utils.denormalize(hazy_images)[0].detach().cpu().numpy().transpose(1,2,0)     # H x W x 3
        init_clear = utils.denormalize(clear_images)[0].detach().cpu().numpy().transpose(1,2,0)   # H x W x 3
        
        
        # Airlight Estimation
        if airlight_images == None:
            init_airlight, _ = airlight_module.LLF(init_hazy)                          # H x W x 3
        else:
            init_airlight = utils.denormalize(airlight_images)[0].numpy().transpose(1,2,0)    # H x W x 3
        clear_airlight = airlight_module.LLF(init_clear)
        
        init_depth = init_depth.detach().cpu().numpy().transpose(1,2,0)
        init_depth = utils.depth_norm(init_depth)
        
        init_clear_depth = init_clear_depth.detach().cpu().numpy().transpose(1,2,0)
        init_clear_depth = utils.depth_norm(init_clear_depth)
        
        
        # Multi-Step Depth Estimation and Dehazing
        metrics_module = locals()[opt.metrics_module]((init_hazy*255).astype(np.uint8))
        depth = init_depth.copy()
        prediction, airlight = None, None
        max_flag = True
        for i in range(1, opt.stepLimit + 1):
            step = i-1
            last_depth = depth.copy()
            
            with torch.no_grad():
                hazy_images = hazy_images.to(opt.device)
                _, depth = model.forward(hazy_images)
                            
            hazy = utils.denormalize(hazy_images)[0].detach().cpu().numpy().transpose(1,2,0)
            
            if opt.airlight_step_flag == False:
                airlight = init_airlight
            else:
                airlight, _ = airlight_module.LLF(init_hazy)
            
            depth = depth.detach().cpu().numpy().transpose(1,2,0)
            depth = utils.depth_norm(depth)
            depth = np.minimum(depth, last_depth)
            
            # Transmission Map
            trans = np.exp(depth * opt.betaStep * -1)
            
            # Dehazing
            prediction = (hazy - airlight) / (trans + opt.eps) + airlight
            prediction = np.clip(prediction, 0,1)
            hazy_images = torch.Tensor(((prediction-0.5)/0.5).transpose(2,0,1)).unsqueeze(0)
            
            # Calculate Metrics
            diff_metrics = metrics_module.get_diff((prediction*255).astype(np.uint8))
            _psnr = psnr(init_clear,prediction)
            _ssim = ssim(init_clear,prediction).item()
            
            # Stop Condition
            if (diff_metrics < opt.metricsThreshold or i == opt.stepLimit) and max_flag:
                psnr_sum += _psnr
                ssim_sum += _ssim
                print(f'last_stage   = {step}')
                print(f'last_psnr    = {_psnr}')
                print(f'last_ssim    = {_ssim}')
                print(f'last_metrics = {metrics_module.last_value}')
                max_flag = False
            
            if opt.save_log:
                csv_log.append([i,opt.betaStep, metrics_module.cur_value, diff_metrics, _psnr, _ssim])
                
        if opt.save_log:
            save_log.write_csv(input_name, csv_log)
        
        # One-Shot Dehazing
        if opt.one_shot == True:
            trans = np.exp(init_depth * (opt.betaStep*step) * -1)
            one_shot_prediction = (init_hazy-init_airlight)/(trans+opt.eps) + init_airlight
            one_shot_prediction = np.clip(one_shot_prediction,0,1)
            oneshot_psnr = psnr(init_clear,one_shot_prediction)
            oneshot_ssim = ssim(init_clear,one_shot_prediction).item()
            print(f'one-shot: beta = {opt.betaStep*step}, psnr = {oneshot_psnr}, ssim={oneshot_ssim}')
            
            clear_etp = metrics_module.get_cur((init_clear*255).astype(np.uint8))
            print(f'clear_metrics  = {clear_etp}')
        else:
            one_shot_prediction = None
        
        if opt.result_show:
            misc.multi_show([init_hazy,     prediction, init_clear, 
                        init_depth,    depth,      init_clear_depth, 
                        init_airlight, airlight,   clear_airlight, 
                        one_shot_prediction])
    
    batch_num = len(test_loader)
    print(f'mean_psnr = {psnr_sum/batch_num}, mean_ssim = {ssim_sum/batch_num}')

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
    
    opt.dataRoot = 'C:/Users/IIPL/Desktop/data/RESIDE_beta/train'
    # opt.dataRoot = '/Users/sungyoon-kim/Documents/GitHub/RUS_Dehazing/DMDNet/data_sample/RESIDE-beta/train'
    dataset_test = RESIDE_Dataset.RESIDE_Beta_Dataset(opt.dataRoot,[opt.imageSize_W, opt.imageSize_H], printName=True, returnName=True)
    loader_test = DataLoader(dataset=dataset_test, batch_size=opt.batchSize_val,
                             num_workers=0, drop_last=False, shuffle=True)
    
    opt.metrics_module = 'NIQE_Module'
    test_stop_when_threshold(opt, model, loader_test)