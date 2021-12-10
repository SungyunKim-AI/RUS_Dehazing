# User warnings ignore
import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
from tqdm import tqdm
import random
import torch
from dpt.models import DPTDepthModel
from dpt.discriminator import Discriminator

from dpt.Air_UNet import Air_UNet
from Module_Metrics.metrics import get_ssim_batch, get_psnr_batch
from util import misc, save_log, utils

from dataset import NYU_Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.utils import make_grid


def get_args():
    parser = argparse.ArgumentParser()
    # dataset parameters
    parser.add_argument('--dataset', required=False, default='NYU_dataset',  help='dataset name')
    parser.add_argument('--dataRoot', type=str, default='',  help='data file path')
    parser.add_argument('--norm', type=bool, default=True,  help='Image Normalize flag')
    
    # learning parameters
    parser.add_argument('--seed', type=int, default=101, help='Random Seed')
    parser.add_argument('--batchSize', type=int, default=8, help='test dataloader input batch size')
    parser.add_argument('--imageSize_W', type=int, default=256, help='the width of the resized input image to network')
    parser.add_argument('--imageSize_H', type=int, default=256, help='the height of the resized input image to network')
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # model parameters
    parser.add_argument('--preTrainedModel', type=str, default='weights/dpt_hybrid_nyu-2ce69ec7.pt', help='pretrained DPT path')
    parser.add_argument('--backbone', type=str, default="vitb_rn50_384", help='DPT backbone')
    
    # validate parameters
    parser.add_argument('--save_log', type=bool, default=True, help='log save flag')
    parser.add_argument('--saveORshow', type=str, default='save',  help='results show or save')
    parser.add_argument('--verbose', type=bool, default=True, help='print log')
    parser.add_argument('--betaStep', type=float, default=0.005, help='beta step')
    parser.add_argument('--stepLimit', type=int, default=250, help='Multi step limit')
    parser.add_argument('--eps', type=float, default=1e-12, help='Epsilon value for non zero calculating')

    return parser.parse_args()


def validation(opt, epoch, depth_model, air_model, netD, dataloader):
    netD.eval()
    
    for batch in tqdm(dataloader, desc="Validate"):
        csv_log = [[] for _ in range(opt.batchSize)]   
        # Data Init
        hazy_image, clear_image, GT_airlight, GT_depth, input_name = batch
        
        clear_image_ = clear_image.clone() if opt.saveORshow == 'save' else None
        clear_image = clear_image.to(opt.device)
        cur_hazy = hazy_image.clone().to(opt.device)
        init_hazy = hazy_image.clone().to(opt.device)
        
        with torch.no_grad():
            airlight = air_model(init_hazy)
        
        # Multi-Step Depth Estimation and Dehazing
        beta = opt.betaStep
        beta_list = [0 for _ in range(opt.batchSize)]
        step_list = [0 for _ in range(opt.batchSize)]
        
        netD_psnr = torch.zeros((opt.batchSize), dtype=torch.float32)
        netD_ssim = torch.zeros((opt.batchSize), dtype=torch.float32)
        
        preds = torch.Tensor(opt.batchSize, 3, opt.imageSize_H, opt.imageSize_W).to(opt.device)
        step_flag = []
        
        for step in range(1, opt.stepLimit + 1):
            # Depth Estimation
            with torch.no_grad():
                cur_hazy = cur_hazy.to(opt.device)
                _, cur_depth = depth_model.forward(cur_hazy)
            cur_depth = cur_depth.unsqueeze(1)
            
            # Transmission Map
            trans = torch.exp(cur_depth * -beta)
            trans = torch.add(trans, opt.eps)
            
            # Dehazing
            prediction = (init_hazy - airlight) / trans + airlight
            prediction = torch.clamp(prediction, -1, 1)
            
            # Calculate Metrics
            with torch.no_grad():
                output = netD(prediction).view(-1).detach().cpu()
            
            for i in range(output.shape[0]):              
                if i in step_flag:
                    continue
                else:
                    # print(step, output[i])
                    beta_list[i] = beta
                    if output[i] > 0.5:
                        preds[i] = prediction[i].clone()
                        step_flag.append(i)
                        step_list[i] = step
                    else:
                        # Discriminator가 종료 못시켰을 때
                        gt_beta = utils.get_GT_beta(input_name[i])
                        if beta_list[i] > (gt_beta + 0.1):
                            preds[i] = prediction[i].clone()
                            step_flag.append(i)
                            step_list[i] = step
            
            
            if len(step_flag) == opt.batchSize:
                netD_psnr = get_psnr_batch(preds, clear_image).detach().cpu()
                netD_ssim = get_ssim_batch(preds, clear_image).detach().cpu()
                
                # hazy_grid = make_grid(utils.denormalize(hazy_image.detach().cpu()))
                # clear_gird = make_grid(utils.denormalize(clear_image.detach().cpu()))
                # pred_gird = make_grid(utils.denormalize(preds.detach().cpu()))
                # images = torch.cat((hazy_grid, clear_gird, pred_gird), 1)
                # show(images)
                
                break   # Stop Multi Step
            else:
                cur_hazy = prediction.clone()
                beta += opt.betaStep    # Set Next Step
        
        for i in range(opt.batchSize):
            if opt.save_log:
                csv_log[i].append([step_list[i], beta_list[i], netD_psnr[i].item(), netD_ssim[i].item()])
                save_log.write_csv_netD_err(opt.dataRoot, epoch, input_name[i], csv_log[i])
                
            if opt.saveORshow == 'save':
                misc.results_save_tensor_2(opt.dataRoot, epoch, input_name[i],
                                            clear_image_[i], hazy_image[i], preds[i])
        
        if opt.verbose:
            print(f'\n netD_psnr = {netD_psnr}')
            print(f' netD_ssim = {netD_ssim}')

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

       
if __name__=='__main__':
    opt = get_args()
    
    # opt.seed = random.randint(1, 10000)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    print("=========| Option |=========\n", opt)
    print()
    
    # opt.dataRoot = 'D:/data/NYU'
    opt.dataRoot = 'C:/Users/IIPL/Desktop/data/NYU'
    val_set = NYU_Dataset.NYU_Dataset(opt.dataRoot + '/val', [opt.imageSize_W, opt.imageSize_H], printName=False, returnName=True, norm=opt.norm)
    val_loader = DataLoader(dataset=val_set, batch_size=opt.batchSize,
                             num_workers=2, drop_last=True, shuffle=True)
    
    depth_model = DPTDepthModel(
        path = opt.preTrainedModel,
        scale=0.00030, shift=0.1378, invert=True,
        backbone=opt.backbone,
        non_negative=True,
        enable_attention_hooks=False,
    )
    depth_model = depth_model.to(memory_format=torch.channels_last)
    depth_model.to(opt.device)
    depth_model.eval()
    
    air_model = Air_UNet(input_nc=3, output_nc=3, nf=8).to(opt.device)
    checkpoint = torch.load(f'weights/Air_UNet_epoch_19.pt')
    air_model.load_state_dict(checkpoint['model_state_dict'])
    air_model.eval()

    # Load saved model
    netD = Discriminator().to(opt.device)
    checkpoint = torch.load('weights/Discriminator_epoch_01_100.pt')
    netD.load_state_dict(checkpoint['model_state_dict'])
    # netD.to(opt.device)
    
    epoch = 1
    validation(opt, epoch, depth_model, air_model, netD, val_loader)
    