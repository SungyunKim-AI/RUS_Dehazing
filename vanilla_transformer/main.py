import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import time
import datetime

from data import HazeDataset
import models
from engine import train_one_epoch, evaluate
    

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
    start_epoch = 0
    epochs = 15
    soft = 0.1
    img_size = (256,256)
    batch_size = 16
    
    # ============== Data Load ==============
    # data_path = '/Users/sungyoon-kim/Documents/GitHub/RUS_Dehazing/data_sample'
    data_path = 'C:/Users/IIPL/Desktop/data'
    dataset_train = HazeDataset.Dataset(data_path, img_size, train_flag='train', verbose=False)
    dataset_test  = HazeDataset.Dataset(data_path, img_size, train_flag='test', verbose=False)
    
    print("Train Set Num : ", dataset_train.__len__())  # 70000 + 40 + 50 = 70090
    print("Test Set Num : ", dataset_test.__len__())    # 2135 + 5 + 5 = 2145
    
    loader_train = DataLoader(dataset_train, batch_size, num_workers=0, shuffle=True, drop_last=True)
    loader_test  = DataLoader(dataset_test, batch_size, num_workers=0, drop_last=True)
    
    model = models.VT(num_queries=100, hidden_dim=256)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_drop)
    loss_fn = nn.MSELoss()
    
    # ============== Train ==============
    print("Start Training")
    start_time = time.time()
    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch(loader_train, model, loss_fn, optimizer, device, epoch)
        print(f"[{(epoch+1):2d}/{epochs:4d}] loss: {train_loss:.6f}")
        
        test_loss = evaluate(loader_test, model, loss_fn, device)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')
        
    