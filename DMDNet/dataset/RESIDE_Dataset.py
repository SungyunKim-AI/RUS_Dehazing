from glob import glob
import numpy as np
import cv2
import mat73
from torch.utils.data import Dataset
from dpt.transforms import Resize
from dataset import utils


class RESIDE_Beta_Dataset(Dataset):
    def __init__(self, path, img_size, printName=False, returnName=False ,norm=False, verbose=False):
        super().__init__()
        self.img_size = img_size
        images_clear_path = path+'/clear/*.jpg'
        self.images_clear_list = glob(images_clear_path)
        self.printName = printName
        self.returnName = returnName
        
        images_hazy_folders_path = path+'/hazy/*/'
        self.images_hazy_lists = []
        for images_hazy_folder in glob(images_hazy_folders_path):
            if verbose:
                print(images_hazy_folder + ' dataset ready!')
            self.images_hazy_lists.append(glob(images_hazy_folder+'*.jpg'))
        
        images_airlight_folders_path = path+'/airlight/*/'
        self.images_airlight_list = []
        for images_airlight_folder in glob(images_airlight_folders_path):
            if verbose:
                print(images_airlight_folder + ' dataset ready!')
            self.images_airlight_list.append(glob(images_airlight_folder+'*.jpg'))
        self.airlgiht_flag = False if len(self.images_airlight_list) == 0 else True
        
        self.images_count = len(self.images_hazy_lists[0])
        self.transform = utils.make_transform(img_size, norm=norm)
        
    def __len__(self):
        return len(self.images_hazy_lists) * self.images_count
        
    def __getitem__(self,index):
        haze = self.images_hazy_lists[index//self.images_count][index%self.images_count]
        clear = self.images_clear_list[index%self.images_count]
        airlight = self.images_airlight_list[index//self.images_count][index%self.images_count] if self.airlgiht_flag else None
        
        if self.printName:
            print(self.images_hazy_lists[index//self.images_count][index%self.images_count])
        
        if airlight is None:
            hazy_input, clear_input = utils.load_item_2(haze, clear, self.transform)
            if self.returnName:
                return hazy_input, clear_input, haze
            else:
                return hazy_input, clear_input 
        else:
            hazy_input, clear_input, airlight_input = utils.load_item_3(haze,clear,airlight,self.transform)
            if self.returnName:
                return hazy_input, clear_input, airlight_input, haze
            else:
                return hazy_input, clear_input, airlight_input
    
class RESIDE_Beta_Dataset_With_Notation(Dataset):
    def __init__(self, path, img_size, printName=False, returnName=False):
        super().__init__()
        self.img_size = img_size
        self.printName = printName
        self.returnName= returnName
        
        images_clear_path = path+'/clear/*.jpg'
        self.images_clear_list = glob(images_clear_path)
        
        images_hazy_folders_path = path+'/hazy/*/'
        self.images_hazy_lists = []
        for images_hazy_folder in glob(images_hazy_folders_path):
            self.images_hazy_lists.append(glob(images_hazy_folder+'*.jpg'))
            
        depth_path = path+'/depth/*.mat'
        self.depth_list = glob(depth_path)
        
        images_airlight_folders_path = path+'/airlight/*/'
        self.images_airlight_list = []
        for images_airlight_folder in glob(images_airlight_folders_path):
            self.images_airlight_list.append(glob(images_airlight_folder+'*.jpg'))
        

        self.images_count = len(self.images_hazy_lists[0])
        self.transform = utils.make_transform(img_size)
        
        self.depth_resize = Resize(
                img_size[0],
                img_size[1],
                resize_target=None,
                keep_aspect_ratio=False,
                ensure_multiple_of=32,
                resize_method="",
                image_interpolation_method=cv2.INTER_AREA,
            )
        
    def __len__(self):
        return len(self.images_hazy_lists) * self.images_count
        #return 10
        
    def __getitem__(self,index):
        haze = self.images_hazy_lists[index//self.images_count][index%self.images_count]
        token = haze.split('_')
        airlight_input = token[2]
        beta = token[3].split('\\')[0]
        
        airlight_input, beta_input = float(airlight_input), float(beta)
        airlight_input = np.full((3,self.img_size[0],self.img_size[1]),airlight_input,dtype='float32')

        clear = self.images_clear_list[index%self.images_count]
        depth_mat = mat73.loadmat(self.depth_list[index%self.images_count])
        depth_input = self.depth_resize({"image": depth_mat['depth']})["image"]
        
        airlight_post = self.images_airlight_list[index//self.images_count][index%self.images_count]
        

        if self.printName:
            print(haze)
        
        haze_input, clear_input, airlight_post = utils.load_item_3(haze,clear, airlight_post,self.transform)
        if self.returnName:
            return haze_input, clear_input, airlight_post, depth_input, airlight_input, beta_input, haze
        else:
            return haze_input, clear_input, airlight_post, depth_input, airlight_input, beta_input
