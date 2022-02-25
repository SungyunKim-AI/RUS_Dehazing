import torch
import torchvision.transforms as transforms
import os
import numpy as np
import cv2

#numpy -> torch
def normalize(x, norm=False, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    if np.mean(x)>1:
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

#torch -> torch
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
    
#torch -> torch
def air_renorm(dataset, norm, airlight, dataset_mean = 0.5, dataset_std = 0.5):
    # denorm
    airlight = air_denorm(dataset, airlight)

    # norm
    if norm:
        airlight = (airlight - dataset_mean) / dataset_std
    
    return airlight

#torch->torch
def air_denorm(dataset,norm, airlight):
    if norm:
        if dataset == 'NYU':
            air_list = torch.Tensor([0.8, 0.9, 1.0])
        elif dataset == 'RESIDE':
            air_list = torch.Tensor([0.8, 0.85, 0.9, 0.95, 1.0])
        elif dataset == 'KITTI':
            return airlight
        mean, std = air_list.mean(), air_list.std(unbiased=False)
        airlight = (airlight * std) + mean
        airlight = torch.clamp(airlight, 0, 1)
            

    return airlight

#numpy->torch
def air_norm(dataset,norm, airlight):
    if norm:
        if dataset == 'NYU':
            air_list = torch.Tensor([0.8, 0.9, 1.0])
        elif dataset == 'RESIDE':
            air_list = torch.Tensor([0.8, 0.85, 0.9, 0.95, 1.0])
        mean, std = air_list.mean(), air_list.std()
        airlight = (airlight * std) + mean
        airlight = torch.clamp(airlight, 0, 1)
    
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

def visualize_depth_inverse(depth): #input : torch(1 X W X H)

    depth_1 = 1/(depth+1)
    depth_1 = ((depth_1-torch.min(depth_1))/(torch.max(depth_1)-torch.min(depth_1)))

    min = torch.min(depth)
    max = torch.max(depth)
    depth_2 = 1 - (depth-min)/(max-min)

    depth = torch.cat((depth_1, depth_2), 2)
    depth = (depth.detach().cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)

    return depth

def visualize_depth_inverse_single(depth): #input : torch(1 X W X H)
    min = torch.min(depth)
    max = torch.max(depth)
    inv_depth = 1 - (depth-min)/(max-min)
    
    inv_depth = (inv_depth.detach().cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)
    inv_depth = cv2.applyColorMap(inv_depth, cv2.COLORMAP_MAGMA)

    return inv_depth

def visualize_depth(depth): #input : torch(1 X W X H)
    min = torch.min(depth)
    max = torch.max(depth)
    depth = (depth-min)/(max-min)
    depth = (depth.detach().cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)
    print(min,max)

    return depth

def visualize_depth_gray(depth):
    min = torch.min(depth)
    max = torch.max(depth)
    depth = (depth-min)/(max-min)
    depth = depth.repeat(3,1,1)
    depth = (depth.detach().cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)

    return depth

def visualize_depth_inverse_gray(depth):
    depth_1 = 1/(depth+1)
    depth_1 = ((depth_1-torch.min(depth_1))/(torch.max(depth_1)-torch.min(depth_1)))

    min = torch.min(depth)
    max = torch.max(depth)
    depth_2 = 1 - (depth-min)/(max-min)

    depth = torch.cat([depth_1, depth_2],2).repeat(3,1,1)
    depth = 255 - (depth.detach().cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)

    return depth