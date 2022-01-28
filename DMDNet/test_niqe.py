from torch.utils.data import DataLoader
from dataset import *
from niqe import niqe
import cv2
import numpy as np

def main():
    val_set   = RESIDE_Dataset('D:/data/RESIDE_V0_outdoor' + '/val',  img_size=[256,256], norm=False)
    loader_args = dict(batch_size=1, num_workers=1, drop_last=False, shuffle=False)
    val_loader = DataLoader(dataset=val_set, **loader_args)
    
    for batch in (val_loader):
        hazy_images, clear_images, _, _, gt_betas, input_names = batch
        hazy_image_gray = np.mean(hazy_images[0].numpy(),axis=0)
        
        niqe_score = niqe.niqe(hazy_image_gray)
        print(gt_betas[0], niqe_score)
        

if __name__ == '__main__':
    main()