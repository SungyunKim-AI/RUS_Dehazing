from cv2 import transform
import torch
import torchvision.transforms as transforms
import os

def normalize(x, norm=False, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    if norm:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
    return transform(x)

def denormalize(x, norm=False, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    if norm:
        # 3, H, W, B
        ten = x.clone().permute(1, 2, 3, 0)
        for t, m, s in zip(ten, mean, std):
            t.mul_(s).add_(m)
        # B, 3, H, W
        return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)
    else:
        return x

def depth_norm(depth):
    a = 0.0012
    b = 3.7
    return a * depth + b


def get_GT_beta(input_name):
    fileName = os.path.basename(input_name[0])[:-4]
    beta = float(fileName.split('_')[-1])
    return beta
