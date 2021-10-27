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
from HazeDataset import O_Haze_Dataset
from metrics import psnr,ssim
from dpt.blocks import Interpolate

def test(model, tail, tail2,test_loader,device):
    
    model.eval()
    tail.eval()
    tail2.eval()
    for batch in tqdm(test_loader):
        hazy_images, clear_images = batch
        with torch.no_grad():
            hazy_images = hazy_images.to(device)
            clear_images = clear_images.to(device)
            tf_prediction, prediction = model.forward(hazy_images)
        
        
        t_maps = tail(tf_prediction)
        b = tail2(tf_prediction)
        
        hazy = (hazy_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        clear = (clear_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        depth = prediction.detach().cpu().numpy().transpose(1,2,0)
        t_map = t_maps[0].detach().cpu().numpy().transpose(1,2,0)
        
        outputs = hazy_images * t_maps + b
        output = (outputs[0] * 0.5 + 0.5).detach().cpu().numpy()
        output = output.transpose(1,2,0)
        
        cv2.imshow("input",hazy)
        cv2.imshow("output",output)
        cv2.imshow("clear",clear)
        cv2.imshow("depth",depth/255)
        cv2.imshow("t_map",t_map)
        cv2.waitKey(0)
            
            
            
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    epochs = 100
    net_w = 640
    net_h = 480
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
    
    
    
    tail =  nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    tail2 = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    model = model.to(memory_format=torch.channels_last)
    #model = model.half()
    tail.load_state_dict(torch.load('weights/weight_tail_020.pth'))
    tail2.load_state_dict(torch.load('weights/weight_tail2_020.pth'))
    
    model.to(device)
    tail.to(device)
    tail2.to(device)
    
    dataset_train=O_Haze_Dataset('D:/data/O-Haze/train',[net_w,net_h])
    loader_test = DataLoader(
                dataset=dataset_train,
                batch_size=1,
                num_workers=0,
                drop_last=True,
                shuffle=True)
    
    test(model, tail, tail2, loader_test,device)