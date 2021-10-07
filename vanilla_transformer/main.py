import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
import torchvision
from torch.utils.data import DataLoader

from data import HazeDataset



if __name__=='__main__':
    # ============== Config Init ==============
    lr = 0.0001
    epochs = 15
    soft = 0.1
    img_size = (256,256)
    batch_size = 16
    dataset_train=HazeDataset.Dataset('C:/Users/IIPL/Desktop/data', img_size, train_flag='train')
    dataset_test=HazeDataset.Dataset('C:/Users/IIPL/Desktop/data', img_size, train_flag='test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Train Set : ", dataset_train.__len__())  # 70000 + 40 + 50 = 70090
    print("Test Set : ", dataset_test.__len__())    # 2135 + 5 + 5 = 2145
    
    # # ============== Data Load =============
    # loader_train = DataLoader(
    #             dataset=dataset_train,
    #             batch_size=batch_size,
    #             num_workers=0,
    #             drop_last=True,
    #             shuffle=True)
    
    # loader_test = DataLoader(
    #             dataset=dataset_train,
    #             batch_size=batch_size,
    #             num_workers=0,
    #             drop_last=True,
    #             shuffle=True)