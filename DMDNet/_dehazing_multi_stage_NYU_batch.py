# User warnings ignore
import warnings
warnings.filterwarnings("ignore")

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import argparse
import numpy as np
import random
from tqdm import tqdm

import torch, torchvision
from dpt.models import DPTDepthModel

from dataset import NYU_Dataset
from torch.utils.data import DataLoader

from Module_Airlight.Airlight_Module import get_Airlight
from Module_Metrics.metrics import get_ssim_batch, get_psnr_batch
from util import misc, save_log, utils



def get_args():
    parser = argparse.ArgumentParser()
    # dataset parameters
    parser.add_argument('--dataset', required=False, default='NYU_dataset',  help='dataset name')
    parser.add_argument('--dataRoot', type=str, default='',  help='data file path')
    parser.add_argument('--norm', type=bool, default=True,  help='Image Normalize flag')
    
    # learning parameters
    parser.add_argument('--seed', type=int, default=101, help='Random Seed')
    parser.add_argument('--batchSize', type=int, default=4, help='test dataloader input batch size')
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


def test_stop_when_threshold(opt, model, test_loader):
    model.eval()
    
    for batch in tqdm(test_loader):
        last_pred = torch.Tensor(opt.batchSize, 3, opt.imageSize_H, opt.imageSize_W)
        csv_log = [[] for _ in range(opt.batchSize)]            # result log save to csv
        
        # Data Init
        hazy_image, clear_image, GT_airlight, GT_depth, input_name = batch
        
        clear_image_ = clear_image.clone() if opt.saveORshow == 'save' else None
        clear_image = clear_image.to(opt.device)
        # airlight = airlight.to(opt.device)
        airlight = get_Airlight(hazy_image).to(opt.device)
        
        # Multi-Step Depth Estimation and Dehazing
        beta = opt.betaStep
        beta_list = [0 for _ in range(opt.batchSize)]
        best_psnr = torch.zeros((opt.batchSize), dtype=torch.float32)
        best_ssim = torch.zeros((opt.batchSize), dtype=torch.float32)
        cur_hazy = hazy_image.clone().to(opt.device)
        stop_flag = []
        for step in range(1, opt.stepLimit + 1):
            # Depth Estimation
            with torch.no_grad():
                cur_hazy = cur_hazy.to(opt.device)
                _, cur_depth = model.forward(cur_hazy)
            cur_depth = cur_depth.unsqueeze(1)
            
            # Transmission Map
            trans = torch.exp(cur_depth * -beta)
            trans = torch.add(trans, opt.eps)
            
            cur_depth = cur_depth.detach().cpu()
            if step == 1:
                init_depth = cur_depth.clone()
            
            # Dehazing
            prediction = (cur_hazy - airlight) / trans + airlight 
            prediction = torch.clamp(prediction, -1, 1)
            
            # Calculate Metrics            
            psnr = get_psnr_batch(prediction, clear_image).detach().cpu()
            ssim = get_ssim_batch(prediction, clear_image).detach().cpu()
            prediction = prediction.detach().cpu()
            for i in range(opt.batchSize):
                if i in stop_flag:
                    continue
                
                if best_psnr[i] < psnr[i]:
                    best_psnr[i] = psnr[i]
                
                    if best_ssim[i] < ssim[i]:
                        best_ssim[i] = ssim[i]
                    
                    last_pred[i] = prediction[i].clone()
                
                    if opt.save_log:
                        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 \
                            = utils.compute_errors(GT_depth[i].numpy(), cur_depth[i].numpy())
                        csv_log[i].append([step, beta, best_psnr, best_ssim, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3])
                
                else:
                    beta_list[i] = round(beta, 4)
                    stop_flag.append(i)
            
            if len(stop_flag) == opt.batchSize:
                break   # Stop Multi Step
            else:
                beta += opt.betaStep    # Set Next Step
                     
            
        # Final Depth Estimation
        with torch.no_grad():
            dehazed = last_pred.to(opt.device)
            _, final_depth = model.forward(dehazed)
        final_depth = final_depth.detach().cpu().unsqueeze(1)
        
        for i in range(opt.batchSize):
            if opt.save_log:
                abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 \
                    = utils.compute_errors(GT_depth[i].numpy(), final_depth[i].numpy())
                csv_log[i].append([step, beta_list[i], best_psnr, best_ssim, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3])
                save_log.write_csv_depth_err(opt.dataRoot, input_name[i], csv_log[i])
                
            if opt.saveORshow == 'save':
                misc.results_save_tensor(opt.dataRoot, input_name[i],
                                         clear_image_[i], hazy_image[i], last_pred[i],
                                         GT_depth[i], init_depth[i], final_depth[i])
        
        if opt.verbose:
            gt_beta = utils.get_GT_beta(input_name[-1])
            print(f'\nlast_beta = {beta_list}({gt_beta})')
            print(f'last_psnr = {best_psnr}')
            print(f'last_ssim = {best_ssim}')
                
            

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
     
    dataset_test = NYU_Dataset.NYU_Dataset(opt.dataRoot, [opt.imageSize_W, opt.imageSize_H], printName=False, returnName=True, norm=opt.norm)
    loader_test = DataLoader(dataset=dataset_test, batch_size=opt.batchSize,
                             num_workers=0, drop_last=False, shuffle=False)
    
    test_stop_when_threshold(opt, model, loader_test)