import torch
import cv2
import math

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
from HazeDataset import NYU_Dataset, O_Haze_Dataset
from metrics import psnr,ssim
from dpt.blocks import Interpolate

def lr_schedule_cosdecay(t,T,init_lr):
    lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
    return lr

def train(model, tail, tail2,train_loader, optim, criterion, epochs, init_lr, device):
    batch_num = len(train_loader)
    steps = batch_num * epochs
    
    losses = []
    i = 0
    mse_epoch = 0.0
    ssim_epoch = 0.0
    psnr_epoch = 0.0
    
    step = 1 
    model.eval()
    tail.train()
    tail2.train()
    for epoch in range(epochs):
        i=0
        mse_epoch = 0.0
        ssim_epoch = 0.0
        psnr_epoch = 0.0
        for batch in tqdm(train_loader):
            optim.zero_grad()
            lr = lr_schedule_cosdecay(step, steps, init_lr)
            for param_group in optim.param_groups:
                param_group["lr"] = lr
            hazy_images, clear_images = batch
            
            with torch.no_grad():
                hazy_images = hazy_images.to(device)
                clear_images = clear_images.to(device)
                tf_prediction, prediction = model.forward(hazy_images)
            t_maps = tail(tf_prediction)
            A = tail2(tf_prediction)
            
            outputs = (hazy_images-(A-0.5))/(t_maps+1e-5) + A
            
            loss = criterion[0](outputs, clear_images)
            loss2 = criterion[1](outputs, clear_images)
            loss = loss + 0.04*loss2
            loss.backward()
            
            optim.step()
            losses.append(loss.item())
            
            MSELoss = nn.MSELoss()
            mse_ = MSELoss(outputs, clear_images)
            ssim_ = ssim(outputs, clear_images)
            psnr_ = psnr(outputs, clear_images)
            
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
        wandb.log({"MSE" : mse_epoch,
                   "SSIM": ssim_epoch,
                   "PSNR" : psnr_epoch,
                   "global_step" : epoch+1})
        '''
        
        path_of_tail_weight = f'weights/weight_tail_{epoch+1:03}.pth'  #path for storing the weights of genertaor
        torch.save(tail.state_dict(), path_of_tail_weight)
        
        path_of_tail2_weight = f'weights/weight_tail2_{epoch+1:03}.pth'  #path for storing the weights of genertaor
        torch.save(tail2.state_dict(), path_of_tail2_weight)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    epochs = 20
    net_w = 640
    net_h = 480
    batch_size = 4
    lr = 0.001
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
    
    
    
    tail =  nn.Sequential( #airlight
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    tail2 = nn.Sequential( #beta
            nn.Conv2d(256, 32, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    model = model.to(memory_format=torch.channels_last)
    #model = model.half()
    model.to(device)
    tail.to(device)
    tail2.to(device)
    
    dataset_train=O_Haze_Dataset('D:/data/O-Haze/train',[net_w,net_h])
    loader_train = DataLoader(
                dataset=dataset_train,
                batch_size=batch_size,
                num_workers=0,
                drop_last=True,
                shuffle=True)
    
    criterion = []
    criterion.append(nn.L1Loss().to(device))
    vgg_model = vgg16(pretrained=True).features[:16]
    vgg_model = vgg_model.to(device)
    for param in vgg_model.parameters():
        param.requires_grad = False
    criterion.append(PerLoss(vgg_model).to(device))
    optimizer = optim.Adam([{'params':tail.parameters()},{'params':tail2.parameters()}],lr, betas = (0.9, 0.999), eps=1e-08)
    
    train(model, tail, tail2, loader_train, optimizer, criterion, epochs, lr, device)