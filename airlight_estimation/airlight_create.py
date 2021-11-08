import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from airlight_estimation import Airlight_Module
from data.HazeDataset import BeDDE_Dataset, RESIDE_Beta_Dataset

if __name__=='__main__':
    if not cv2.useOptimized():
        cv2.setUseOptimized(True)
    
    dataset_cofig = {'BeDDE' : BeDDE_Dataset(), 
                    'Dense_Haze' : , 
                    'NH_Haze' : , 
                    'RESIDE_beta': RESIDE_Beta_Dataset()
                    }
    
    for datasetName, dataset in dataset_cofig.items():
        
        dataloader = DataLoader()
        module = Airlight_Module()
        
        for data in tqdm(dataloader):
            hazy = data
            imgname = os.path.basename(hazy)
            
            hazy = module.AWC(hazy)
            airlight_hat, _ = module.LLF(hazy)
            
            save_path = 
            cv2.imwrite(save_path, airlight_hat)