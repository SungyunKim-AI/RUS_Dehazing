from skimage.io import imread
import os
from glob import glob
import csv
import torch
import numpy as np
import cv2
from metrics import psnr, ssim
from entropy_module import Entropy_Module
from tqdm import tqdm

if __name__ == '__main__':
    device = 'cuda'
    # hazy_imgs = glob('D:/data/RESIDE_V0_outdoor/val/hazy/*/*.jpg')
    hazy_imgs = glob('D:/data/RESIDE_V0_outdoor/RTTS/*')
    # clear_img_dir = 'D:/data/RESIDE_V0_outdoor/val/clear'

    output_dir = 'D:/data/output_dehaze/RTTS_MSBDN'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    f = open(f'{output_dir}/RTTS_MSBDN.csv', 'w', newline='')
    wr = csv.writer(f)
    
    model_dir = 'models/model.pkl'
    
    model = torch.load(model_dir, map_location=lambda storage, loc: storage)
    model.to(device)
    model.eval()
    entropy_module = Entropy_Module()
    
    for hazy_img in tqdm(hazy_imgs):
        file_name = os.path.basename(hazy_img);
        token = file_name.split('_')
        # clear_img = clear_img_dir+'/'+token[0]+'.png'
        # beta = token[-1][:-4]
        
        haze = np.asarray(cv2.resize(imread(hazy_img),[256,256]).transpose((2,0,1)))/255
        # clear = np.asarray(cv2.resize(imread(clear_img),[256,256]).transpose((2,0,1)))/255
        
        haze = torch.Tensor(haze).unsqueeze(0).to(device)
        # clear = torch.Tensor(clear).unsqueeze(0).to(device)
        
        if haze.shape != torch.Size([1,3,256,256]):
            continue
        
        with torch.no_grad():
            pred = model(haze)
        
        # pred_psnr = psnr(pred, clear)
        # pred_ssim = ssim(pred, clear)
        
        pred = pred[0].detach().cpu().numpy().transpose(1,2,0)
        # clear = clear[0].detach().cpu().numpy().transpose(1,2,0)
        pred_entropy, _, _ = entropy_module.get_cur(pred)
        # pred = cv2.cvtColor((np.clip(pred,0,1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        # wr.writerow([file_name, beta, pred_psnr, pred_ssim.item(), pred_entropy])
        wr.writerow([file_name, pred_entropy])
        
        # cv2.imshow('pred', cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
        # cv2.imshow('clear', cv2.cvtColor(clear, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        pred = cv2.cvtColor((np.clip(pred,0,1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_dir+'/'+file_name, pred)
        
    f.close()
        
        
        
        