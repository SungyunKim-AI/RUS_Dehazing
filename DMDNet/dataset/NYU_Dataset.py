import os
from posixpath import pardir
import cv2
from glob import glob
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from utils.io import *

class NYU_Dataset_clear(Dataset):
    """
    => return (NYU_Dataset)
        images : 480x640x3 (HxWx3) Tensor of RGB images.
        depths : 480x640 (HxW) matrix of in-painted depth map. The values of the depth elements are in meters.
    """
    def __init__(self, dataRoot):
        super().__init__()
        self.images = glob(dataRoot + '/clear/*.jpg')
        self.depths = glob(dataRoot + '/depth/*.npy')
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
    
    
class NYU_Dataset(Dataset):
    def __init__(self, path, img_size, norm=False, verbose=False, selection=[]):
        super().__init__()
        self.norm = norm
        # clear images
        self.images_clear_list = glob(path + '/clear/*.jpg')
        self.depths_list = glob(path + '/depth/*.npy')
        
        # hazy images
        self.hazy_lists = []
        for images_hazy_folder in glob(path+'/hazy/*/'):
            if verbose:
                print(images_hazy_folder + ' dataset ready!')
            self.hazy_lists.append(glob(images_hazy_folder+'*.jpg'))

        self.img_size = img_size
        
        self.images_count = len(self.hazy_lists[0])
        # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        self.transform = make_transform(img_size, norm=self.norm)
        
    def __len__(self):
        return len(self.hazy_lists) * self.images_count
        
    def __getitem__(self,index):
        haze = self.hazy_lists[index//self.images_count][index%self.images_count]
        filename = os.path.basename(haze)
        clear = self.images_clear_list[index%self.images_count]
        GT_depth = np.load(self.depths_list[index%self.images_count])
        GT_depth = cv2.resize(GT_depth, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_CUBIC)
        GT_depth = np.expand_dims(GT_depth, axis=0)
        
        GT_airlight = np.array(float(filename.split('_')[-2]))
        GT_beta = np.array(float(filename.split('_')[-1][:-4]))

        if self.norm:
            air_list = np.array([0.8, 0.9, 1.0])
            GT_airlight = (GT_airlight - air_list.mean()) / air_list.std()
        
        GT_airlight = np.expand_dims(GT_airlight, axis=0)
        GT_beta = np.expand_dims(GT_beta, axis=0)
        
        hazy_input, clear_input = load_item(haze, clear, self.transform)
        
        return hazy_input, clear_input, GT_depth, GT_airlight, GT_beta, filename
        
    