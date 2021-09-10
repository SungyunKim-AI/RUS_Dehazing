import torch
import cv2
from PIL import Image
import PIL
import torchvision.transforms.functional as F
import glob
import random


def to_tensor(img):
    img_t = F.to_tensor(img).float()
    return img_t
    
def load_item(haze, clear):
        hazy_image = cv2.imread(haze)
        clear_image = cv2.imread(clear)
        
        hazy_image = Image.fromarray(hazy_image)
        clear_image = Image.fromarray(clear_image)
 
        hazy_resize = hazy_image.resize((256,256), resample=PIL.Image.BICUBIC)
        clear_resize = clear_image.resize((256,256), resample=PIL.Image.BICUBIC)
        
        hazy_resize = to_tensor(hazy_resize).cuda()
        clear_resize = to_tensor(clear_resize).cuda()
        
        return hazy_resize, clear_resize

class O_Haze_Dataset(torch.utils.data.Dataset):
    def __init__(self,path):
        super().__init__()
        images_clear_path = path+'/clear/*.jpg'
        images_hazy_path = path+'/hazy/*.jpg'
        self.images_clear_list=glob.glob(images_clear_path)
        self.images_hazy_list =glob.glob(images_hazy_path)
    
    def __len__(self):
        return len(self.images_clear_list)
    
    def __getitem__(self,index):
        haze, clear = load_item(self.images_hazy_list[index], self.images_clear_list[index])
        return haze, clear
        
class RESIDE_Beta_Dataset(torch.utils.data.Dataset):
    def __init__(self,path,folders_num):
        super().__init__()
        images_clear_path = path+'/clear/*.jpg'
        self.images_clear_list = glob.glob(images_clear_path)
        
        images_hazy_folders_path = path+'/hazy/*/'
        self.images_hazy_lists = []
        images_hazy_folders=glob.glob(images_hazy_folders_path)
        for folder_num in folders_num:
            images_hazy_folder = images_hazy_folders[folder_num]
            print(images_hazy_folder + ' dataset ready!')
            self.images_hazy_lists.append(glob.glob(images_hazy_folder+'*.jpg'))
            
        self.folders_count = len(folders_num)
        self.images_count = len(self.images_hazy_lists[0])
        
    def __len__(self):
        return self.folders_count * self.images_count
    
    def __getitem__(self,index):
        haze, clear = load_item(self.images_hazy_lists[index//self.images_count][index%self.images_count],
                                self.images_clear_list[index%self.images_count])
        return haze, clear


class Train_Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        super().__init__()
        self.RBTD = RESIDE_Beta_Dataset(path+'/RESIDE-beta/train')
        self.OHTD = O_Haze_Dataset(path+'/O-Haze/train')
        
    def __len__(self):
        return self.RBTD.__len__() + self.OHTD.__len__()

    def __getitem__(self, index):
        if index >= self.RBTD.__len__():
            haze, clear = self.OHTD[index-self.RBTD.__len__()]
        else:
            haze, clear = self.RBTD[index]
        return haze, clear
    
if __name__ == '__main__':
    t = Train_Dataset('D:/data')
    print(t.__len__())
    while(True):
        t[random.randint(0,50000)]