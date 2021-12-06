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
import torch.nn as nn
import torch.optim as optim
from dpt.models import DPTDepthModel
from dpt.discriminator import Discriminator

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
    parser.add_argument('--batchSize', type=int, default=8, help='test dataloader input batch size')
    parser.add_argument('--imageSize_W', type=int, default=640, help='the width of the resized input image to network')
    parser.add_argument('--imageSize_H', type=int, default=480, help='the height of the resized input image to network')
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # model parameters
    parser.add_argument('--preTrainedModel', type=str, default='weights/dpt_hybrid_nyu-2ce69ec7.pt', help='pretrained DPT path')
    parser.add_argument('--backbone', type=str, default="vitb_rn50_384", help='DPT backbone')
    
    # test_stop_when_threshold parameters
    parser.add_argument('--verbose', type=bool, default=True, help='print log')
    parser.add_argument('--betaStep', type=float, default=0.005, help='beta step')
    parser.add_argument('--stepLimit', type=int, default=250, help='Multi step limit')
    parser.add_argument('--eps', type=float, default=1e-12, help='Epsilon value for non zero calculating')
    
    # Discrminator hyperparam
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate for optimizers')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 hyperparam for Adam optimizers')

    return parser.parse_args()


def test_stop_when_threshold(opt, model, netD, test_loader):
    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)).to(opt.device)
    
    model.eval()
    netD.train()
    pbar = tqdm(test_loader)
    for batch in pbar:
        netD.zero_grad()
        csv_log = [[] for _ in range(opt.batchSize)]            # result log save to csv
        
        # Data Init
        hazy_image, clear_image, airlight, GT_depth, input_name = batch
        
        clear_image = clear_image.to(opt.device)
        airlight = airlight.to(opt.device)
        
        # Set tqdm bar
        start_name = os.path.basename(input_name[0])[:-4]
        end_name = os.path.basename(input_name[-1])[:-4]
        pbar.set_description(f"{start_name} ~ {end_name}")
            
        
        # Multi-Step Depth Estimation and Dehazing
        beta = opt.betaStep
        beta_list = [0 for _ in range(opt.batchSize)]
        step_list = [0 for _ in range(opt.batchSize)]
        best_psnr = torch.zeros((opt.batchSize), dtype=torch.float32)
        best_ssim = torch.zeros((opt.batchSize), dtype=torch.float32)
        psnr_preds, ssim_preds = torch.Tensor(), torch.Tensor()
        cur_hazy = hazy_image.clone().to(opt.device)
        stop_flag = []
        errD_fake_list, errD_real_list = [], []
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
            for i in range(opt.batchSize):
                if i in stop_flag:
                    continue
                
                if best_psnr[i] < psnr[i]:
                    best_psnr[i] = psnr[i]
                else:
                    psnr_preds = torch.cat((psnr_preds, prediction[i].clone().unsqueeze(0)))
                    stop_flag.append(i)
                
                if best_ssim[i] < ssim[i]:
                    best_ssim[i] = ssim[i]
                else:
                    ssim_preds = torch.cat((ssim_preds, prediction[i].clone().unsqueeze(0)))
                
                if best_psnr[i] < psnr[i] and best_ssim[i] < ssim[i]:
                    last_preds = torch.cat((last_preds, prediction[i].clone().unsqueeze(0)))
                
                if best_psnr[i] < psnr[i] or best_ssim[i] < ssim[i]:    
                    if opt.save_log:
                        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 \
                            = utils.compute_errors(GT_depth[i].numpy(), cur_depth[i].numpy())
                        csv_log[i].append([step, beta, best_psnr, best_ssim, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3])
                else:   
                    beta_list[i] = round(beta, 4)
                    step_list[i] = step
            
            
            if len(stop_flag) == opt.batchSize:
                for real_image in [clear_image, psnr_preds, ssim_preds]:
                    label = torch.full((real_image.shape[0],), real_label, dtype=torch.float, device=opt.device)
                    output = netD(real_image).view(-1)
                    errD_real = criterion(output, label)
                    errD_real.backward()
                    errD_real_list.append(errD_real.item())
                
                optimizerD.step()
                break   # Stop Multi Step
            else:
                label = torch.full((opt.batchSize,), fake_label, dtype=torch.float, device=opt.device)
                output = netD(last_preds.detach()).view(-1)
                errD_fake = criterion(output, label)
                errD_fake.backward()
                errD_fake_list.append(errD_fake.item())
                optimizerD.step()
                
                beta += opt.betaStep    # Set Next Step
        
        
        if opt.verbose:
            errD_fake = np.mean(np.array(errD_fake_list))
            errD_real = np.mean(np.array(errD_real_list))
            errD = errD_fake + errD_real
            
            print(f'last_psnr = {best_psnr.mean()}')
            print(f'last_ssim = {best_ssim.mean()}')
            print(f"errD      = {errD}")
    

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
            

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
    
    netD = Discriminator()
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)
    
    model = model.to(memory_format=torch.channels_last)
    
    model.to(opt.device)
     
    dataset_test = NYU_Dataset.NYU_Dataset(opt.dataRoot, [opt.imageSize_W, opt.imageSize_H], printName=False, returnName=True, norm=opt.norm)
    loader_test = DataLoader(dataset=dataset_test, batch_size=opt.batchSize,
                             num_workers=2, drop_last=False, shuffle=False)
    
    test_stop_when_threshold(opt, model, netD, loader_test)