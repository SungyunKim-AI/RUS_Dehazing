from torch.autograd import Variable
import PIL.Image as Image
import os
from glob import glob
import csv
import torch
import numpy as np
import cv2
from metrics import psnr
from entropy_module import Entropy_Module
import torchvision.transforms as transforms
from tqdm import tqdm

if __name__ == '__main__':
    device = 'cuda'
    hazy_imgs = glob('D:/data/RESIDE_V0_outdoor/val/hazy/*/*.jpg')
    clear_img_dir = 'D:/data/RESIDE_V0_outdoor/val/clear'

    output_dir = 'D:/data/output_dehaze/pred_AOD_SOTS'
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    f = open('D:/data/output_dehaze/pred_AOD_SOTS/AOD_SOTS.csv', 'w', newline='')
    wr = csv.writer(f)
    
    model_dir = f'model_pretrained/AOD_net_epoch_relu_10.pth'
    
    model = torch.load(model_dir)
    model.to(device)
    model.eval()
    entropy_module = Entropy_Module()
    
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        ]
    )
    
    transform_ = transforms.Compose([
        transforms.ToTensor()]
    )
    
    for hazy_img in tqdm(hazy_imgs):
        file_name = os.path.basename(hazy_img);
        token = file_name.split('_')
        clear_img = clear_img_dir+'/'+token[0]+'.png'
        beta = token[-1][:-4]
        
        haze = transform(Image.open(hazy_img).convert('RGB')).unsqueeze_(0)
        clear = transform_(Image.open(clear_img).convert('RGB')).unsqueeze_(0)
        
        varIn = Variable(haze).cuda()
        
        with torch.no_grad():
            pred = model(varIn)
            
        pred_psnr = psnr(pred, clear)
        
        pred = pred.data.cpu().numpy().squeeze().transpose((1,2,0))
        clear = clear.data.cpu().numpy().squeeze().transpose((1,2,0))
        
        pred_entropy, _, _ = entropy_module.get_cur(pred)
        # pred = cv2.cvtColor((np.clip(pred,0,1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        wr.writerow([file_name, beta, pred_psnr, pred_entropy])
        # cv2.imshow('pred', cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
        # cv2.imshow('clear', cv2.cvtColor(clear, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        pred = cv2.cvtColor((np.clip(pred,0,1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_dir+'/'+file_name, pred)
        
    f.close()
        
        
        
        