import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import HazeDataset
import models

def train():
    

if __name__=='__main__':
    # ============== Config Init ==============
    # fix the seed for reproducibility
    seed = 123
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    lr = 0.0001
    weight_decay=1e-4
    lr_drop = 200
    epochs = 15
    soft = 0.1
    img_size = (256,256)
    batch_size = 16
    
    # ============== Data Load ==============
    data_path = '/Users/sungyoon-kim/Documents/GitHub/RUS_Dehazing/data_sample'
    dataset_train = HazeDataset.Dataset(data_path, img_size, train_flag='train', verbose=False)
    dataset_test  = HazeDataset.Dataset(data_path, img_size, train_flag='test', verbose=False)
    
    print("Train Set Num : ", dataset_train.__len__())  # 70000 + 40 + 50 = 70090
    print("Test Set Num : ", dataset_test.__len__())    # 2135 + 5 + 5 = 2145
    
    loader_train = DataLoader(dataset_train, batch_size, num_workers=0, shuffle=True)
    loader_test  = DataLoader(dataset_test, batch_size, num_workers=0)
    
    model = models.Transformer(d_model=256, nhead=8, num_encoder_layers=6,
                               num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                               activation="relu", normalize_before=False,
                               return_intermediate_dec=False)
    
    
    optimizer = torch.optim.AdamW(param_dicts, lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduㅁler.StepLR(optimizer, lr_drop)
    
    