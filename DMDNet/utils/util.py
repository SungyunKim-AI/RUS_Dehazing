from cv2 import transform
import torch
import torchvision.transforms as transforms
import os
import numpy as np

def normalize(x, norm=False, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    x = x / 255.0
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

def denormalize(x, norm=True, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    if norm:
        # 3, H, W, B
        ten = x.clone().permute(1, 2, 3, 0)
        for t, m, s in zip(ten, mean, std):
            t.mul_(s).add_(m)
        # B, 3, H, W
        return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)
    else:
        return x
    

def air_renorm(dataset, airlight, dataset_mean=0.5, dataset_std=0.5):
    # denorm
    if dataset == 'NYU':
        air_list = torch.Tensor([0.8, 0.9, 1.0])
    elif dataset == 'RESIDE_beta':
        air_list = torch.Tensor([0.8, 0.85, 0.9, 0.95, 1.0])
    mean, std = air_list.mean(), air_list.std()
    airlight = (airlight * std) + mean
    airlight = torch.clamp(airlight, 0, 1)
    
    # renorm
    # dataset_mean = torch.Tensor([dataset_mean])
    # dataset_std = torch.Tensor([dataset_std])
    airlight = (airlight - dataset_mean) / dataset_std
    
    return airlight


def depth_norm(depth):
    a = 0.0012
    b = 3.7
    return a * depth + b


def get_GT_beta(input_name):
    fileName = os.path.basename(input_name)[:-4]
    beta = float(fileName.split('_')[-1])
    return beta


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3]