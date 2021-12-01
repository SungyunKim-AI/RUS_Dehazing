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
    parser.add_argument('--betaStep', type=float, default=0.3, help='beta step')
    parser.add_argument('--stepLimit', type=int, default=250, help='Multi step limit')
    parser.add_argument('--eps', type=float, default=1e-12, help='Epsilon value for non zero calculating')
    
    return parser.parse_args()
    
def tensor2numpy(x):
    return x.detach().cpu().numpy()

def test_stop_when_threshold(opt, model, test_loader):
    
    model.eval()
    
    pbar = tqdm(test_loader)
    for batch in pbar:
        last_pred = torch.Tensor(opt.batchSize, 3, opt.imageSize_H, opt.imageSize_W)
        csv_log = [[] for _ in range(opt.batchSize)]            # result log save to csv
        
        # Data Init
        hazy_image, clear_image, airlight, GT_depth, input_name = batch
        
        clear_image = clear_image.to(opt.device)
        airlight = airlight.to(opt.device)
        
        pbar.set_description(f"{os.path.basename(input_name[0])}~{os.path.basename(input_name[-1])}")
            
        
        # Multi-Step Depth Estimation and Dehazing
        beta = opt.betaStep
        best_psnr = torch.zeros((opt.batchSize), dtype=torch.float32)
        best_ssim = torch.zeros((opt.batchSize), dtype=torch.float32)
        idxs = list(range(opt.batchSize))
        cur_hazy = hazy_image.clone().to(opt.device)
        for step in range(1, opt.stepLimit + 1):
            # Depth Estimation
            with torch.no_grad():
                cur_hazy = cur_hazy.to(opt.device)
                _, cur_depth = model.forward(cur_hazy)
            cur_depth = cur_depth.unsqueeze(1)
            if step == 1:
                init_depth = cur_depth.clone().detach().cpu()
            
            
            # Transmission Map
            trans = torch.exp(cur_depth * -beta)
            trans = torch.add(trans, opt.eps)
            
            # Dehazing
            prediction = (cur_hazy - airlight) / trans + airlight
            prediction = torch.clamp(prediction, -1, 1)
            
            # Calculate Metrics            
            psnr = get_psnr_batch(prediction, clear_image)
            ssim = get_ssim_batch(prediction, clear_image)
            # TODO: denormalize 반영 안되어 있는지 확인
            for i in idxs:
                if best_psnr[i] < psnr[i]:
                    best_psnr[i] = psnr[i]
                
                    if best_ssim[i] < ssim[i]:
                        best_ssim[i] = ssim[i]
                    
                    last_pred[i] = prediction[i].clone().detach().cpu()
                
                    if opt.save_log:
                        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 \
                            = utils.compute_errors(GT_depth[i], prediction[i])
                        csv_log[i].append([step, beta, best_psnr, best_ssim, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3])
                
                else:
                    idxs.remove(i)
            
            if len(idxs) == 0:
                break   # Stop Multi Step
            else:
                # cur_hazy 에서 더이상 PSNR 이 증가하지 않는 batch index는 빼고 새롭게 cur_hazy 생성
                # cur_hazy = cur_hazy.detach().cpu()
                cur_hazy = torch.cat([cur_hazy[i].unsqueeze(0) for i in idxs])    
                beta += opt.betaStep    # Set Next Step
                     
            
        # Final Depth Estimation
        with torch.no_grad():
            dehazed = last_pred.to(opt.device)
            _, final_depth = model.forward(dehazed)
        
        final_depth = final_depth.detach().cpu()
        
        for i in range(opt.batchSize):
            if opt.save_log:
                abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 \
                    = utils.compute_errors(GT_depth[i], final_depth[i].numpy())
                csv_log[i].append([step, beta, best_psnr, best_ssim, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3])
                save_log.write_csv_depth_err(opt.dataRoot, input_name[i], csv_log[i])

            if opt.verbose:
                gt_beta = utils.get_GT_beta(input_name[i])
                print(f'last_step  = {step}')
                print(f'last_beta  = {beta}({gt_beta})')
                print(f'last_psnr  = {best_psnr}')
                print(f'last_ssim  = {best_ssim}')
                
            if opt.saveORshow == 'save':
                """
                clear     init_hazy   psnr_best_prediction
                GT_depth  init_depth  final_depth            
                """
                image_grid = torchvision.utils.make_grid(clear_image[i], hazy_image[i], last_pred[i])
                depth_grid = torchvision.utils.make_grid(GT_depth[i], init_depth[i], final_depth[i])
                misc.results_save_tensor(opt.dataRoot, input_name[i], 'image', image_grid)
                misc.results_save_tensor(opt.dataRoot, input_name[i], 'depth', depth_grid)
                
            

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
    
    opt.dataRoot = 'C:/Users/IIPL/Desktop/data/NYU/'
    # opt.dataRoot = 'D:/data/RESIDE_beta_sample/train'
    # opt.dataRoot = 'D:/data/NYU/'
    dataset_test = NYU_Dataset.NYU_Dataset(opt.dataRoot, [opt.imageSize_W, opt.imageSize_H], printName=False, returnName=True, norm=opt.norm)
    loader_test = DataLoader(dataset=dataset_test, batch_size=opt.batchSize,
                             num_workers=1, drop_last=False, shuffle=False)
    
    test_stop_when_threshold(opt, model, loader_test)