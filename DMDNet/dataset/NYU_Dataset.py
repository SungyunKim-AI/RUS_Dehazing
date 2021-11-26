import os
import cv2
from glob import glob
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class NYU_Dataset(Dataset):
    """
    => return (NYU_Dataset)
        images : 480x640x3 (HxWx3) Tensor of RGB images.
        depths : 480x640 (HxW) matrix of in-painted depth map. The values of the depth elements are in meters.
    """
    def __init__(self, dataRoot):
        super().__init__()
        self.images = glob(dataRoot + '/images/*.jpg')
        self.depths = glob(dataRoot + '/depths/*.npy')
        self.toTensor = transforms.ToTensor()
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        name = os.path.basename(self.images[index])[:-4]
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.toTensor(image)
        
        depth = np.load(self.depths[index])
        
        return image, depth, name