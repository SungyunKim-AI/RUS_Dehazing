import torch

def denormalize(x, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    # 3, H, W, B
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

def depth_norm(depth):
    a = 0.0012
    b = 3.7
    return a * depth + b
