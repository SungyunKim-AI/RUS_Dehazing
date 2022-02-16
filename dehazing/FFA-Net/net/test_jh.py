import os
from tkinter import W
import numpy as np
import csv
from PIL import Image
from models import *

import torch
import torch.nn as nn
import torchvision.transforms as tfs 
import torchvision.utils as vutils
# import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import cv2
from metrics import psnr
from entropy_module import Entropy_Module

if __name__=='__main__':
    gps=3
    blocks=19
    
    hazy_imgs = glob('D:/data/RESIDE_V0_outdoor/val/hazy/*/*.jpg')
    clear_img_dir = 'D:/data/RESIDE_V0_outdoor/val/clear'
    
    output_dir = 'D:/data/output_dehaze/pred_FFA_SOTS'
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    f = open('D:/data/output_dehaze/pred_FFA_SOTS/FFA_SOTS.csv', 'w', newline='')
    wr = csv.writer(f)
    
    model_dir = f'trained_models/ots_train_ffa_{gps}_{blocks}.pk'
    
    device = 'cuda'
    ckp = torch.load(model_dir, map_location=device)
    net = FFA(gps=gps, blocks=blocks)
    net = nn.DataParallel(net)
    net.load_state_dict(ckp['model'])
    net.eval()
    entropy_module = Entropy_Module()
    
    transform = tfs.Compose([tfs.ToTensor(),
                            tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])])
    transform_ = tfs.Compose([tfs.ToTensor()])
    
    for hazy_img in tqdm(hazy_imgs):
        file_name = os.path.basename(hazy_img);
        token = file_name.split('_')
        clear_img = clear_img_dir+'/'+token[0]+'.png'
        beta = token[-1][:-4]
        haze = transform(Image.open(hazy_img)).unsqueeze(0).to(device)
        clear = transform_(Image.open(clear_img)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = net(haze)
        
        pred_psnr = psnr(pred, clear)
        #print(pred_psnr)

        pred = pred[0].detach().cpu().numpy().transpose(1,2,0)
        clear = clear[0].detach().cpu().numpy().transpose(1,2,0)
        pred_entropy, _, _ = entropy_module.get_cur(pred)
        #print(pred_entropy)
        
        wr.writerow([file_name, beta, pred_psnr, pred_entropy])
        
        
        pred = cv2.cvtColor((np.clip(pred,0,1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_dir+'/'+file_name, pred)
        # cv2.imshow('pred', cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
        # cv2.imshow('clear', cv2.cvtColor(clear, cv2.COLOR_RGB2BGR))
        
        # cv2.waitKey(0)
        
        
    f.close()