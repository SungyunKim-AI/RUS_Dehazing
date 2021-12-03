import os
import cv2
from glob import glob
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from dataset import utils

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
    def __init__(self, path, img_size, printName=False, returnName=False ,norm=False, verbose=False):
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
        self.printName = printName
        self.returnName = returnName
        
        self.images_count = len(self.hazy_lists[0])
        self.transform = utils.make_transform(img_size, norm=self.norm)
        
    def __len__(self):
        return len(self.hazy_lists) * self.images_count
        
    def __getitem__(self,index):
        haze = self.hazy_lists[index//self.images_count][index%self.images_count]
        clear = self.images_clear_list[index%self.images_count]
        GT_depth = np.load(self.depths_list[index%self.images_count])
        GT_depth = np.expand_dims(GT_depth, axis=0)
        
        airlight = float(os.path.basename(haze).split('_')[-2])
        if self.norm:
            airlight = (airlight - 0.5) / 0.5
        airlight_input = np.full((3, self.img_size[1], self.img_size[0]), airlight).astype(np.float32)
        
        if self.printName:
            print(haze)
        
        hazy_input, clear_input = utils.load_item_2(haze, clear, self.transform)
        if self.returnName:
            return hazy_input, clear_input, airlight_input, GT_depth, haze
        else:
            return hazy_input, clear_input, airlight_input, GT_depth  
        
class NYU_Dataset2(Dataset):
    def __init__(self, path, img_size, printName=False, returnName=False ,norm=False, verbose=False):
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
        self.printName = printName
        self.returnName = returnName
        
        self.images_count = len(self.hazy_lists[0])
        self.transform = utils.make_transform(img_size, norm=self.norm)
        
    def __len__(self):
        return len(self.hazy_lists) * self.images_count
        
    def __getitem__(self,index):
        haze = self.hazy_lists[index//self.images_count][index%self.images_count]        
        clear = self.images_clear_list[index%self.images_count]
        depth = np.load(self.depths_list[index%self.images_count]).T
        
        airlight = float(os.path.basename(haze).split('_')[-2])
        airlight = np.full(self.img_size, airlight)
        
        if self.printName:
            print(haze)
        
        hazy_input, clear_input = utils.load_item_2(haze, clear, self.transform)
        gt_airlight = (airlight-0.5)/0.5
        if self.returnName:
            return hazy_input, clear_input, depth, gt_airlight, haze
        else:
            return hazy_input, clear_input, depth, gt_airlight


import shutil

if __name__ == '__main__':
    # dataset random split
    dataRoot = 'C:/Users/IIPL/Desktop/data/NYU'
    clear_images_len = 1449
    clear_images = glob(dataRoot + '/clear/*.jpg')
    depths = glob(dataRoot + '/depth/*.npy')
    
    val_len = round(clear_images_len*0.2)
    np.random.seed(1449)
    val_set_clear = np.random.choice(clear_images, val_len, replace=False)
    val_set_depth = np.random.choice(depths, val_len, replace=False)
    
    for i in range(val_len):
        fileName = os.path.basename(val_set_clear[i])
        shutil.move(val_set_clear[i], dataRoot + '/val/clear/' + fileName)
        
        fileName = os.path.basename(val_set_depth[i])
        shutil.move(val_set_depth[i], dataRoot + '/val/clear/' + fileName)
    