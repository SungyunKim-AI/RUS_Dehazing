import torch
import cv2
import math
import numpy as np
import wandb

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
from HazeDataset import *
from metrics import psnr,ssim
from dpt.blocks import Interpolate

def calc_trans(hazy,clear,airlight):
    trans = (hazy-airlight)/(clear-airlight+1e-8)
    #trans = np.clip(trans,0,1)
    #trans = np.mean(trans,2)
    return trans

def calc_trans_tensor(hazy,clear,airlight):
    trans = (hazy-airlight)/(clear-airlight+1e-8)
    #trans = torch.clamp(trans,0,1)
    #trans = torch.mean(trans,1)
    #trans = trans.unsqueeze(1)
    return trans

def lr_schedule_cosdecay(t,T,init_lr):
    lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
    return lr

def train(model, train_loader, loader_valid, optim, criterion, epochs, device):
    batch_num = len(train_loader)
    #steps = batch_num * epochs
    
    losses = []
    i = 0
    mse_epoch = 0.0
    ssim_epoch = 0.0
    psnr_epoch = 0.0
    
    step = 1 
    for epoch in range(epochs):
        model.train()
        tail.train()
        i=0
        mse_epoch = 0.0
        ssim_epoch = 0.0
        psnr_epoch = 0.0
        for batch in tqdm(train_loader):
            optim.zero_grad()
            hazy_images, clear_images, airlight_images = batch
            
            
            hazy_images = hazy_images.to(device)
            clear_images = clear_images.to(device)
            airlight_images = airlight_images.to(device)
            tf_pred, depth_pred = model.forward(hazy_images)

            trans_pred = tail(tf_pred)
            
            #trans_pred = trans_pred.unsqueeze(1)
            trans_images = calc_trans_tensor(hazy_images,clear_images,airlight_images)

            #trans_pred = trans_pred.repeat([1,3,1,1])
            #trans_images = trans_images.repeat([1,3,1,1])

            clear_pred = (hazy_images-airlight_images)/(trans_pred+1e-8)+airlight_images
            loss = criterion[0](trans_pred, trans_images)
            loss2 = criterion[1](trans_pred, trans_images)
            #loss3 = criterion[0](trans_pred,trans_images)
            ssim_ = ssim(clear_pred, clear_images)
            psnr_ = psnr(clear_pred, clear_images)
            MSELoss = nn.MSELoss()
            mse_ = MSELoss(clear_pred, clear_images)

            loss = loss + 0.04*loss2 + ssim_ + psnr_ + mse_
            loss.backward()
            
            optim.step()
            losses.append(loss.item())
            
            mse_epoch += mse_.cpu().item()
            ssim_epoch += ssim_.cpu().item()
            psnr_epoch += psnr_
            i += 1
            
            step+=1
            
        mse_epoch /= i
        ssim_epoch /= i
        psnr_epoch /= i
        print("mse: + "+str(mse_epoch) + " | ssim: "+ str(ssim_epoch) + " | psnr:"+str(psnr_epoch))
        print()
        
        '''
        model.eval()
        
        for batch in loader_valid:
            hazy_images, clear_images, airlight_images = batch
            with torch.no_grad():
                hazy_images = hazy_images.to(device)
                clear_images = clear_images.to(device)
                airlight_images = airlight_images.to(device)
                _, hazy_trans = model.forward(hazy_images)
            
            hazy_trans = hazy_trans[0].unsqueeze(2).detach().cpu().numpy()
            hazy = (hazy_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
            clear = (clear_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
            airlight = (airlight_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
            
            prediction = (hazy-airlight)/(hazy_trans+1e-8) + airlight
            print(hazy_trans)
            trans_calc = calc_trans(hazy,clear,airlight)
            
            hazy = cv2.cvtColor(hazy,cv2.COLOR_BGR2RGB)
            clear = cv2.cvtColor(clear,cv2.COLOR_BGR2RGB)
            prediction = cv2.cvtColor(prediction,cv2.COLOR_BGR2RGB)
            airlight = cv2.cvtColor(airlight,cv2.COLOR_BGR2RGB)
            
            #hazy_depth = cv2.applyColorMap((hazy_depth/5*255).astype(np.uint8),cv2.COLORMAP_JET)
            
            cv2.imshow("hazy",hazy)
            cv2.imshow("clear",clear)
            
            cv2.imshow("hazy_trans", hazy_trans)
            cv2.imshow("calc_trans", trans_calc)
            
            cv2.imshow("airlight",airlight)
            
            cv2.imshow("clear_prediction",prediction)
            cv2.waitKey(0)
            break
        '''
        
        
        wandb.log({"MSE" : mse_epoch,
                   "SSIM": ssim_epoch,
                   "PSNR" : psnr_epoch,
                   "global_step" : epoch+1})
        
        weight_path = f'weights/dpt_hybrid-midas-501f0c75_trans_{epoch+1:03}.pt'  #path for storing the weights of genertaor
        torch.save(model.state_dict(), weight_path)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    epochs = 20
    net_w = 256
    net_h = 256
    batch_size = 27
    lr = 0.01
    input_path = 'input'
    output_path = 'output_dehazed'
    
    config_defaults = {
        'model_name' : 'DPT_finetuning',
        'init_lr' : lr,
        'epochs' : epochs,
        'dataset' : 'NTIRE_Dataset',
        'batch_size': batch_size,
        'image_size': [net_w,net_h]}
    wandb.init(config=config_defaults, project='Dehazing', entity='rus')
    wandb.run.name = config_defaults['model_name']
    
    model = DPTDepthModel(
        path = 'weights/dpt_hybrid-midas-501f0c75.pt',
        scale=0.00006016,
        shift=0.00579,
        invert=True,
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
    )
    
    model = model.to(memory_format=torch.channels_last)
    model.to(device)

    tail =  nn.Sequential( #t_map
        nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
        Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(True),
        nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
        nn.ReLU(True)
    )
    tail.to(device)
    
    dataset_train=NTIRE_Dataset('D:/data',[net_w,net_h],flag='train',verbose=False)
    loader_train = DataLoader(
                dataset=dataset_train,
                batch_size=batch_size,
                num_workers=0,
                drop_last=False,
                shuffle=True)
    
    dataset_valid=NTIRE_Dataset('D:/data',[net_w,net_h],flag='val',verbose=False)
    loader_valid = DataLoader(
                dataset=dataset_train,
                batch_size=1,
                num_workers=0,
                drop_last=False,
                shuffle=False)
    criterion = []
    criterion.append(nn.L1Loss().to(device))
    vgg_model = vgg16(pretrained=True).features[:16]
    vgg_model = vgg_model.to(device)
    for param in vgg_model.parameters():
        param.requires_grad = False
    criterion.append(PerLoss(vgg_model).to(device))
    optimizer = optim.Adam(model.parameters(),lr, betas = (0.9, 0.999), eps=1e-08)
    
    train(model, loader_train,loader_valid, optimizer, criterion, epochs, device)