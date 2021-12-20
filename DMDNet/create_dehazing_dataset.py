# import warnings
# warnings.filterwarnings("ignore")
import argparse
import random
import numpy as np
from tqdm import tqdm
import h5py

import torch
from models.depth_models import DPTDepthModel
from models.air_models import UNet

from dataset import NYU_Dataset, RESIDE_Beta_Dataset
from torch.utils.data import DataLoader

from utils import air_renorm, get_ssim_batch, get_psnr_batch


def get_args():
    parser = argparse.ArgumentParser()
    # dataset parameters
    parser.add_argument('--dataset', required=False, default='NYU',  help='dataset name')
    parser.add_argument('--dataRoot', type=str, default='',  help='data file path')
    
    # learning parameters
    parser.add_argument('--seed', type=int, default=101, help='Random Seed')
    parser.add_argument('--batchSize', type=int, default=1, help='test dataloader input batch size')
    parser.add_argument('--imageSize_W', type=int, default=620, help='the width of the resized input image to network')
    parser.add_argument('--imageSize_H', type=int, default=460, help='the height of the resized input image to network')
    parser.add_argument('--norm', type=bool, default=True,  help='Image Normalize flag')
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # model parameters
    parser.add_argument('--preTrainedModel', type=str, default='weights/dpt_hybrid-midas-501f0c75.pt', help='pretrained DPT path')
    parser.add_argument('--backbone', type=str, default="vitb_rn50_384", help='DPT backbone')
    
    # test_stop_when_threshold parameters
    parser.add_argument('--saveORshow', type=str, default='save',  help='results show or save')
    parser.add_argument('--verbose', type=bool, default=True, help='print log')
    parser.add_argument('--betaStep', type=float, default=0.005, help='beta step')
    parser.add_argument('--stepLimit', type=int, default=250, help='Multi step limit')
    parser.add_argument('--eps', type=float, default=1e-12, help='Epsilon value for non zero calculating')
    
    return parser.parse_args()


def save_dehazing_image(opt, depth_model,air_model, dataloader):
    
    for batch in tqdm(dataloader, desc="Train"):
        hazy_image, clear_image, GT_airlight, GT_depth, input_name = batch
        
        clear_image = clear_image.to(opt.device)
        init_hazy = hazy_image.clone().to(opt.device)
        cur_hazy = hazy_image.clone().to(opt.device)
        
        with torch.no_grad():
            airlight = air_model(init_hazy)
            airlight = air_renorm(opt.dataset, airlight)
        
        # Multi-Step Depth Estimation and Dehazing
        images = {}
        beta = opt.betaStep
        best_psnr, best_ssim = np.zeros(opt.batchSize), np.zeros(opt.batchSize)
        psnr_best_img = torch.Tensor(opt.batchSize, 3, opt.imageSize_H, opt.imageSize_W).to(opt.device)
        ssim_best_img = torch.Tensor(opt.batchSize, 3, opt.imageSize_H, opt.imageSize_W).to(opt.device)
        last_pred = torch.Tensor().to(opt.device)
        stop_flag_psnr, stop_flag_ssim = [], []
        
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
            psnr = get_psnr_batch(prediction, clear_image).detach().cpu().numpy()
            ssim = get_ssim_batch(prediction, clear_image).detach().cpu().numpy()
            
            for i in range(opt.batchSize):
                if i not in stop_flag_psnr:
                    if best_psnr[i] <= psnr[i]:
                        best_psnr[i] = psnr[i]
                    else:
                        psnr_best_img[i] = prediction[i].clone()
                        stop_flag_psnr.append(i)
                
                if i not in stop_flag_ssim:      
                    if best_ssim[i] <= ssim[i]:
                        best_ssim[i] = ssim[i]
                    else:
                        ssim_best_img[i] = prediction[i].clone()
                        stop_flag_ssim.append(i)
                
                if (i not in stop_flag_psnr) and (i not in stop_flag_ssim):
                    pred = prediction[i].clone().unsqueeze(0)
                    last_pred = torch.cat((last_pred, pred), 0)
                    
            if (len(stop_flag_psnr) == opt.batchSize) and (len(stop_flag_ssim) == opt.batchSize):
                images['clear_image'] = clear_image.detach().cpu()
                images['psnr_best_image'] = psnr_best_img.detach().cpu()
                images['ssim_best_image'] = ssim_best_img.detach().cpu()
                # for i in range(last_pred.shape[0]):
                
                save_h5py(images)
                
                
                break   # Stop Multi Step
            else:
                cur_hazy = prediction.clone()
                beta += opt.betaStep    # Set Next Step
                
        if opt.verbose:    
            print(f'\nlast_psnr = {best_psnr}')
            print(f'last_ssim = {best_ssim}')
            
def save_h5py(save_path, input_name, images):
    folder = input_name.split('/')[-2]
    file_name = input_name.split('/')[-1][:-4]
    save_path = os.path.join(save_path, input_name)
    with h5py.File('data.h5', 'w') as hf: 
        for image_name, image in images.items():
            if image_name.split('_')[-1] == 'image':
                imgSet = hf.create_dataset(
                    name=image_name,
                    data=image,
                    shape=(HEIGHT, WIDTH, CHANNELS),
                    maxshape=(HEIGHT, WIDTH, CHANNELS),
                    compression="gzip",
                    compression_opts=9)
            else:
                metricSet = hf.create_dataset(
                    name=image_name,
                    data=image
                    shape=(1,),
                    maxshape=(None,),
                    compression="gzip",
                    compression_opts=9)
                
            

if __name__ == '__main__':
    opt = get_args()
    
    # opt.seed = random.randint(1, 10000)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    print("=========| Option |=========\n", opt)
    
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
    
    air_model = UNet([opt.imageSize_W, opt.imageSize_H], in_channels=3, out_channels=1, bilinear=True)
    air_model.to(device=opt.device)
    
    checkpoint = torch.load(opt.air_model_path)
    air_model.load_state_dict(checkpoint['model_state_dict'])
    
    
    dataset_args = dict(img_size=[opt.imageSize_W, opt.imageSize_H], norm=opt.norm)
    if opt.dataset == 'NYU':
        dataset = NYU_Dataset(opt.dataRoot + '/train', **dataset_args)
    elif opt.dataset == 'RESIDE_beta':
        dataset = RESIDE_Beta_Dataset(opt.dataRoot + '/train', **dataset_args)
    
    dataloader = DataLoader(dataset=dataset, batch_size=opt.batchSize,
                             num_workers=0, drop_last=False, shuffle=False)
    
    save_dehazing_image(opt, depth_model, air_model, dataloader)