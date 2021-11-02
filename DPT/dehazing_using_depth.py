import torch
import cv2
import math
import numpy as np

from torchvision.transforms import Compose
from torchvision.models import vgg16
from loss import LossNetwork as PerLoss
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dpt.models import DPTDepthModel
from dpt.transforms import NormalizeImage
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
from HazeDataset import Dataset, O_Haze_Dataset, RESIDE_Beta_Dataset
from metrics import psnr,ssim
from dpt.blocks import Interpolate

def test(model,test_loader,device):
    
    model.eval()
    for batch in tqdm(test_loader):
        hazy_images, clear_images = batch
        with torch.no_grad():
            hazy_images = hazy_images.to(device)
            clear_images = clear_images.to(device)
            _, hazy_depth = model.forward(hazy_images)
            _, clear_depth = model.forward(clear_images)
        
        hazy = (hazy_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        clear = (clear_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        hazy_depth = hazy_depth.detach().cpu().numpy().transpose(1,2,0)/255
        clear_depth = clear_depth.detach().cpu().numpy().transpose(1,2,0)/255
        
        beta = -3
        airlight = 0.8
        hazy_trans = np.exp(hazy_depth * beta) + 1e-5
        calc_airlight = (hazy - clear*hazy_trans)/(1-hazy_trans+1e-5)
        calc_trans = (hazy-airlight)/(clear-airlight+1e-5)
        
        avg_r = np.average(calc_airlight[:,:,0])
        avg_g = np.average(calc_airlight[:,:,1])
        avg_b = np.average(calc_airlight[:,:,2])
        
        avg = np.array([avg_r,avg_g,avg_b])
        avg = avg.reshape((1,1,)+avg.shape)
        print(avg.shape)
        print(avg)
        calc_airlight = avg
        
        clear_prediction = (hazy-calc_airlight)/(hazy_trans+1e-5) + calc_airlight
        
        hazy = cv2.cvtColor(hazy,cv2.COLOR_BGR2RGB)
        clear = cv2.cvtColor(clear,cv2.COLOR_BGR2RGB)
        calc_trans = cv2.cvtColor(calc_trans,cv2.COLOR_BGR2GRAY)
        clear_prediction = cv2.cvtColor(clear_prediction,cv2.COLOR_BGR2RGB)
        
        cv2.imshow("input",hazy)
        cv2.imshow("clear",clear)
        cv2.imshow("hazy_depth",hazy_depth)
        #cv2.imshow("clear_depth",clear_depth)
        cv2.imshow("hazy_trans", hazy_trans)
        cv2.imshow("clear_prediction",clear_prediction)
        cv2.imshow("airlight",calc_airlight)
        cv2.imshow("calc_trans",calc_trans)
        cv2.waitKey(0)
            
            
            
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    epochs = 100
    net_w = 320
    net_h = 240
    input_path = 'input'
    output_path = 'output_dehazed'
    
    model = DPTDepthModel(
        path = 'weights/dpt_hybrid_kitti-cb926ef4.pt',
        scale=0.00006016,
        shift=0.00579,
        invert=True,
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
    )
    
    model = model.to(memory_format=torch.channels_last)
    #model = model.half()
    
    model.to(device)
    
    dataset_train=O_Haze_Dataset('D:/data/O-Haze/train',[net_w,net_h])
    loader_test = DataLoader(
                dataset=dataset_train,
                batch_size=1,
                num_workers=0,
                drop_last=True,
                shuffle=True)
    
    test(model,loader_test,device)