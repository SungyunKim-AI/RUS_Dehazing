# User warnings ignore
import warnings
warnings.filterwarnings("ignore")

import argparse
import cv2
import numpy as np
import random
from tqdm import tqdm
from skvideo.measure import niqe

import torch
from dpt.models import DPTDepthModel

from dataset.HazeDataset import *
from torch.utils.data import DataLoader

from Module_Airlight.Airlight_Module import Airlight_Module
from Module_Metrics.Entropy_Module import Entropy_Module
from Module_Metrics.metrics import ssim, psnr
from util.io import *
from util.misc import *
from util.save_log import write_csv


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=False, default='pix2pix',  help='')
    parser.add_argument('--dataRoot', type=str, default='D:/data/Dense_Haze/train',  help='data file path')
    parser.add_argument('--preTrainedModel', type=str, default='weights/dpt_hybrid-midas-501f0c75.pt', help='pretrained DPT path')
    parser.add_argument('--manualSeed', type=int, default=101, help='B2A: facade, A2B: edges2shoes')
    parser.add_argument('--backbone', type=str, default="vitb_rn50_384", help='DPT backbone')
    parser.add_argument('--batchSize_train', type=int, default=4, help='input batch size')
    parser.add_argument('--batchSize_val', type=int, default=1, help='input batch size')
    parser.add_argument('--imageSize_W', type=int, default=256, help='the width of the resized input image to network')
    parser.add_argument('--imageSize_H', type=int, default=256, help='the height of the resized input image to network')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--savePath', default='weights', help='folder to model checkpoints')
    parser.add_argument('--inputPath', default='input', help='input path')
    parser.add_argument('--outputPath', default='output_dehazed', help='dehazed output path')
    parser.add_argument('--evalIter', type=int, default=10, help='interval for evaluating')
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--eps', type=float, default=1e-8, help='Epsilon value for non zero calculating')
    
    # test_stop_when_threshold parameters
    parser.add_argument('--one_shot', type=bool, default=True, help='flag of One shot dehazing')
    parser.add_argument('--result_show', type=bool, default=False, help='result images display flag')
    parser.add_argument('--save_log', type=bool, default=False, help='log save flag')
    parser.add_argument('--airlight_step_flag', type=bool, default=False, help='flag of multi step airlight estimation')
    parser.add_argument('--betaStep', type=float, default=0.001, help='beta step')
    parser.add_argument('--stepLimit', type=int, default=1000, help='Multi step limit')
    parser.add_argument('--metrics_module', type=str, default='Entropy_Module',  help='Non Reference metrics method name')
    parser.add_argument('--metricsThreshold', type=float, default=0.001, help='Metrics different threshold')
    
    
    return parser.parse_args()

def depth_norm(depth):
    a = 0.0012
    b = 3.7
    return a * depth + b

def test_with_D_Hazy_NYU_notation(model, test_loader, device):
    entropy_module = Entropy_Module()
    stage = 100
    model.eval()
    psnr_sum, ssim_sum = 0.0, 0.0
    step_beta = 0.01
    
    a = 0.0012
    b = 3.7
    
    a_sum, b_sum= 0,0
    for cnt, batch in tqdm(enumerate(test_loader)):
        
        last_etp = 0
        hazy_images, clear_images, airlight_images, depth_images = batch
        
        print(f'beta_per_stage = {step_beta}')
        
        
        with torch.no_grad():
            hazy_images = hazy_images.to(device)
            clear_images = clear_images.to(device)
            airlight_images = airlight_images.to(device)
            _, init_depth = model.forward(hazy_images)
            _, init_clear_depth = model.forward(clear_images)
            
        init_hazy = (hazy_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        init_clear = (clear_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        init_airlight = (airlight_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        depth_gt = (depth_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        
        init_depth = init_depth.detach().cpu().numpy().transpose(1,2,0)
        init_depth = a*init_depth+b
        #init_depth = depth_gt.copy()
        
        init_clear_depth = init_clear_depth.detach().cpu().numpy().transpose(1,2,0)
        init_clear_depth = a*init_clear_depth+b
        
        _psnr = 0
        _ssim = 0
        step=0
        
        depth = init_depth.copy()
        prediction = None
        last_depth = None
        last_prediction = None
        
        
        for i in range(1,stage):
            
            last_depth = depth.copy()
            last_prediction = prediction
            last_psnr = _psnr
            last_ssim = _ssim
            
            
            with torch.no_grad():
                hazy_images = hazy_images.to(device)
                _, depth = model.forward(hazy_images)
                            
            hazy = (hazy_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
            depth = depth.detach().cpu().numpy().transpose(1,2,0)
            
            
            '''
            print(np.min(depth))
            print(np.max(depth))
            
            a = np.sqrt(np.var(depth_gt)/np.var(depth))
            print(f'{a}')
            depth_gt_mean = np.mean(depth_gt)
            depth_mean = np.mean(depth)
            b = depth_gt_mean - a * depth_mean
            print(f'{b}')
            
            a_sum += a
            b_sum += b
            
            print(f'a_mean = {a_sum/(cnt+1)}, b_mean = {b_sum/(cnt+1)}')
            
            #show_histogram((depth).astype(np.int32),131)
            #show_histogram(((a*depth+b)*255).astype(np.int32),132)
            #show_histogram((depth_gt*255).astype(np.int32),133)
            #plt.show()
            break
            '''
            
            
            
            depth = a*depth+b
            depth = np.minimum(depth,last_depth)
            #depth = depth_gt.copy()
            
            trans = np.exp(depth * step_beta * -1)
            
            prediction = (hazy-init_airlight)/(trans+e) + init_airlight
            prediction = np.clip(prediction,0,1)
            hazy_images = torch.Tensor(((prediction-0.5)/0.5).transpose(2,0,1)).unsqueeze(0)
            
            cur_etp = entropy_module.get_entropy((prediction*255).astype(np.uint8))
            diff_etp = cur_etp - last_etp
            
            #print(f'{i}-stage')
            #print(f'cur_etp = {cur_etp}')
            #print(f'last_etp = {last_etp}')
            #print(f'etp_diff = {diff_etp}')
            
            _psnr = psnr(init_clear,prediction)
            _ssim = ssim(init_clear,prediction).item()
            #print(f'psnr = {_psnr}, ssim = {_ssim}')
            #print()
            
            if diff_etp<0 or i==stage-1:
                step = i-1
                psnr_sum +=last_psnr
                ssim_sum +=last_ssim
                print(f'last_stage = {step}')
                print(f'last_psnr  = {last_psnr}')
                print(f'last_ssim  = {last_ssim}')
                print(f'last_etp   = {last_etp}')
                last_etp = cur_etp
                break
            
            last_etp = cur_etp
        
        #continue
        
        trans = np.exp(init_depth * (step_beta*step) * -1)
        one_shot_prediction = (init_hazy-init_airlight)/(trans+e) + init_airlight
        one_shot_prediction = np.clip(one_shot_prediction,0,1)
        _psnr = psnr(init_clear,one_shot_prediction)
        _ssim = ssim(init_clear,one_shot_prediction).item()
        clear_etp = entropy_module.get_entropy((init_clear*255).astype(np.uint8))
        print(f'clear_etp  = {clear_etp}')
        print(f'one-shot: beta = {step_beta*step}, psnr = {_psnr}, ssim={_ssim}')
        
        last_prediction = cv2.cvtColor(last_prediction.astype(np.float32),cv2.COLOR_BGR2RGB)
        init_hazy_img = cv2.cvtColor(init_hazy,cv2.COLOR_RGB2BGR)
        init_clear_img = cv2.cvtColor(init_clear,cv2.COLOR_RGB2BGR)
        init_airlight_img = cv2.cvtColor(init_airlight,cv2.COLOR_RGB2BGR)
        one_shot_prediction = cv2.cvtColor(one_shot_prediction.astype(np.float32),cv2.COLOR_BGR2RGB)
        
        cv2.imshow('final depth', depth/10)
        cv2.imshow('last stage prediction',last_prediction)
        cv2.imshow("init_hazy",init_hazy_img)
        cv2.imshow("init_clear",init_clear_img)
        cv2.imshow("init_airlight",init_airlight_img)
        cv2.imshow("init_depth", init_depth/10)
        cv2.imshow("init_clear_depth", init_clear_depth/10)
        cv2.imshow("depth_gt",depth_gt)
        cv2.imshow('one_shot_prediction',one_shot_prediction)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    

def test_with_RESIDE_notation(model, test_loader, device):
    entropy_module = Entropy_Module()
    stage = 100
    model.eval()
    psnr_sum, ssim_sum = 0.0, 0.0
    step_beta = 0.005
    
    a = 0.0018
    b = 3.95
    
    a_sum, b_sum= 0,0
    for cnt, batch in tqdm(enumerate(test_loader)):
        
        last_etp = 0
        hazy_images, clear_images, airlight_images, depth_images, airlight_gt, beta_gt= batch
        
        print(f'beta_per_stage = {step_beta}')
        
        
        with torch.no_grad():
            hazy_images = hazy_images.to(device)
            clear_images = clear_images.to(device)
            airlight_images = airlight_images.to(device)
            _, init_depth = model.forward(hazy_images)
            _, init_clear_depth = model.forward(clear_images)
            
        init_hazy = (hazy_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        init_clear = (clear_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        init_airlight = (airlight_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        depth_gt = depth_images.detach().cpu().numpy().transpose(1,2,0).astype(np.float32)
        
        init_depth = init_depth.detach().cpu().numpy().transpose(1,2,0)
        init_depth = a*init_depth+b
        #init_depth = depth_gt
        
        init_clear_depth = init_clear_depth.detach().cpu().numpy().transpose(1,2,0)
        init_clear_depth = a*init_clear_depth+b
        
        _psnr = 0
        _ssim = 0
        step=0
        
        depth = init_depth.copy()
        prediction = None
        last_depth = None
        last_prediction = None
        
        
        for i in range(1,stage):
            
            last_depth = depth.copy()
            last_prediction = prediction
            last_psnr = _psnr
            last_ssim = _ssim
            
            
            with torch.no_grad():
                hazy_images = hazy_images.to(device)
                _, depth = model.forward(hazy_images)
                            
            hazy = (hazy_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
            depth = depth.detach().cpu().numpy().transpose(1,2,0)
            
            
            print(np.min(depth))
            print(np.max(depth))
            
            a = np.sqrt(np.var(depth_gt)/np.var(depth))
            print(f'{a}')
            depth_gt_mean = np.mean(depth_gt)
            depth_mean = np.mean(depth)
            b = depth_gt_mean - a * depth_mean
            print(f'{b}')
            
            a_sum += a
            b_sum += b
            
            print(f'a_mean = {a_sum/(cnt+1)}, b_mean = {b_sum/(cnt+1)}')
            
            #show_histogram((depth).astype(np.int32),131)
            #show_histogram(((a*depth+b)*255).astype(np.int32),132)
            #show_histogram((depth_gt*255).astype(np.int32),133)
            #plt.show()
            break
            
            
            
            
            depth = a*depth+b
            depth = np.minimum(depth,last_depth)
            #depth = depth_gt
            
            trans = np.exp(depth * step_beta * -1)
            
            prediction = (hazy-init_airlight)/(trans+e) + init_airlight
            prediction = np.clip(prediction,0,1)
            hazy_images = torch.Tensor(((prediction-0.5)/0.5).transpose(2,0,1)).unsqueeze(0)
            
            cur_etp = entropy_module.get_entropy((prediction*255).astype(np.uint8))
            diff_etp = cur_etp - last_etp
            
            #print(f'{i}-stage')
            #print(f'cur_etp = {cur_etp}')
            #print(f'last_etp = {last_etp}')
            #print(f'etp_diff = {diff_etp}')
            
            _psnr = psnr(init_clear,prediction)
            _ssim = ssim(init_clear,prediction).item()
            #print(f'psnr = {_psnr}, ssim = {_ssim}')
            #print()
            
            if diff_etp<0 or i==stage-1:
                step = i-1
                psnr_sum +=last_psnr
                ssim_sum +=last_ssim
                print(f'last_stage = {step}')
                print(f'last_psnr  = {last_psnr}')
                print(f'last_ssim  = {last_ssim}')
                print(f'last_etp   = {last_etp}')
                last_etp = cur_etp
                break
            
            last_etp = cur_etp
        
        continue
        
        trans = np.exp(init_depth * (step_beta*step) * -1)
        one_shot_prediction = (init_hazy-init_airlight)/(trans+e) + init_airlight
        one_shot_prediction = np.clip(one_shot_prediction,0,1)
        _psnr = psnr(init_clear,one_shot_prediction)
        _ssim = ssim(init_clear,one_shot_prediction).item()
        clear_etp = entropy_module.get_entropy((init_clear*255).astype(np.uint8))
        print(f'clear_etp  = {clear_etp}')
        print(f'one-shot: beta = {step_beta*step}, psnr = {_psnr}, ssim={_ssim}')
        
        last_prediction = cv2.cvtColor(last_prediction.astype(np.float32),cv2.COLOR_BGR2RGB)
        init_hazy_img = cv2.cvtColor(init_hazy,cv2.COLOR_RGB2BGR)
        init_clear_img = cv2.cvtColor(init_clear,cv2.COLOR_RGB2BGR)
        init_airlight_img = cv2.cvtColor(init_airlight,cv2.COLOR_RGB2BGR)
        one_shot_prediction = cv2.cvtColor(one_shot_prediction.astype(np.float32),cv2.COLOR_BGR2RGB)
        
        cv2.imshow('final depth', depth/10)
        cv2.imshow('last stage prediction',last_prediction)
        cv2.imshow("init_hazy",init_hazy_img)
        cv2.imshow("init_clear",init_clear_img)
        cv2.imshow("init_airlight",init_airlight_img)
        cv2.imshow("init_depth", init_depth/10)
        cv2.imshow("init_clear_depth", init_clear_depth/10)
        cv2.imshow("depth_gt",depth_gt/10)
        cv2.imshow('one_shot_prediction',one_shot_prediction)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def test_stop_when_niqe(init_hazy):
    last_niqe = niqe(cv2.cvtColor(init_hazy,cv2.COLOR_BGR2GRAY))
    print(last_niqe)


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
        
        init_hazy = denormalize(hazy_images)[0].detach().cpu().numpy().transpose(1,2,0)     # H x W x 3
        init_clear = denormalize(clear_images)[0].detach().cpu().numpy().transpose(1,2,0)   # H x W x 3
        
        # Airlight Estimation
        if airlight_images == None:
            init_airlight, _ = airlight_module.LLF(init_hazy)                          # H x W x 3
        else:
            init_airlight = denormalize(airlight_images)[0].numpy().transpose(1,2,0)    # H x W x 3
        clear_airlight = airlight_module.LLF(init_clear)
        
        init_depth = init_depth.detach().cpu().numpy().transpose(1,2,0)
        init_depth = depth_norm(init_depth)
        
        init_clear_depth = init_clear_depth.detach().cpu().numpy().transpose(1,2,0)
        init_clear_depth = depth_norm(init_clear_depth)
        
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
                            
            hazy = denormalize(hazy_images)[0].detach().cpu().numpy().transpose(1,2,0)
            
            if opt.airlight_step_flag == False:
                airlight = init_airlight
            else:
                airlight, _ = airlight_module.LLF(init_hazy)
            
            depth = depth.detach().cpu().numpy().transpose(1,2,0)
            depth = depth_norm(depth)
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
            write_csv(input_name, csv_log)
        
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
            multi_show([init_hazy,     prediction, init_clear, 
                        init_depth,    depth,      init_clear_depth, 
                        init_airlight, airlight,   clear_airlight, 
                        one_shot_prediction])
    
    batch_num = len(test_loader)
    print(f'mean_psnr = {psnr_sum/batch_num}, mean_ssim = {ssim_sum/batch_num}')

if __name__ == '__main__':
    opt = get_args()
    
    # opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)
    print("=========| Option |=========\n", opt)
    
    model = DPTDepthModel(
        path = opt.preTrainedModel,
        scale=1,
        shift=0,
        invert=False,
        backbone=opt.backbone,
        non_negative=True,
        enable_attention_hooks=False,
    )
    
    model = model.to(memory_format=torch.channels_last)
    #model = model.half()
    
    model.to(opt.device)
    
    opt.dataRoot = '/Users/sungyoon-kim/Documents/GitHub/RUS_Dehazing/DMDNet/data_sample/RESIDE-beta/train'
    dataset_test = RESIDE_Beta_Dataset(opt.dataRoot,[opt.imageSize_W, opt.imageSize_H],printName=True,returnName=True)
    loader_test = DataLoader(dataset=dataset_test, batch_size=1,num_workers=0,
                             drop_last=False, shuffle=True)
    
    #test_with_D_Hazy_NYU_notation(model,loader_test,device)
    test_stop_when_threshold(opt, model, loader_test, 
                             one_shot=True, result_show=True, save_log=False)