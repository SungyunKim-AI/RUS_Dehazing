from glob import glob
import h5py
import numpy as np
import cv2
import mat73

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision.transforms import Compose

from dpt.transforms import Resize, NormalizeImage, PrepareForNet
from util import io

def to_tensor(img):
    img_t = F.to_tensor(img).float()
    return img_t
    
def load_item(haze, clear, img_size):
        hazy_image = cv2.imread(haze)
        hazy_image = cv2.resize(hazy_image,img_size)/255
        
        clear_image = cv2.imread(clear)
        clear_image = cv2.resize(clear_image,img_size)/255
        
        hazy_resize = to_tensor(hazy_image)
        clear_resize = to_tensor(clear_image)
        
        return hazy_resize, clear_resize
    
def load_item_2(haze, clear,transform):
    haze = io.read_image(haze)
    clear = io.read_image(clear)
    
    haze_input  = transform({"image": haze})["image"]
    clear_input = transform({"image": clear})["image"]

    return haze_input, clear_input

def load_item_3(haze, clear, airlight, transform):
    haze = io.read_image(haze)
    clear = io.read_image(clear)
    airlight = io.read_image(airlight)
    
    haze_input  = transform({"image": haze})["image"]
    clear_input = transform({"image": clear})["image"]
    airlight_input = transform({"image": airlight})["image"]

    return haze_input, clear_input, airlight_input

def make_transform(img_size):
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = Compose(
        [
            Resize(
                img_size[0],
                img_size[1],
                resize_target=None,
                keep_aspect_ratio=False,
                ensure_multiple_of=32,
                resize_method="",
                image_interpolation_method=cv2.INTER_AREA,
            ),
            normalization,
            PrepareForNet(),
        ]
    )
    return transform
    

class BeDDE_Dataset(Dataset):
    def __init__(self,path,img_size,printName=False,returnName=False):
        super().__init__()
        self.img_size = img_size
        images_hazy_path = path+'/*/fog/*.png'
        self.images_hazy_list =glob(images_hazy_path)

        images_airlight_path = path+'/*/airlight/*.png'
        self.images_airlight_list = glob(images_airlight_path)
        
        self.printName = printName
        self.returnName= returnName
        self.transform = make_transform(self.img_size)
    
    def __len__(self):
        return len(self.images_hazy_list)
    
    def __getitem__(self,index):
        image_hazy_path = self.images_hazy_list[index]
        if self.printName:
            print(image_hazy_path)
        image_airlight_path = self.images_airlight_list[index]
            
        image_clear_path_slice = image_hazy_path.split('\\')
        image_clear_path = ''
        for i in range(len(image_clear_path_slice)-2):
            image_clear_path+=(image_clear_path_slice[i]+'\\')
        image_clear_path = image_clear_path+'gt/'+image_clear_path_slice[-3]+'_clear.png'
        
        hazy_input, clear_input, airlight_input = load_item_3(image_hazy_path,image_clear_path,image_airlight_path, self.transform)
        if self.returnName:
            return hazy_input, clear_input, airlight_input, image_hazy_path
        else:
            return hazy_input, clear_input, airlight_input
        
class D_Hazy_Middlebury_Dataset(Dataset):
    def __init__(self,path,img_size,printName=False,returnName=False):
        super().__init__()
        self.img_size = img_size
        images_clear_path = path+'/clear/*.png'
        images_hazy_path = path+'/hazy/*.bmp'
        images_airlight_path = path+'/airlight/*.bmp'
        self.images_clear_list=glob(images_clear_path)
        self.images_hazy_list =glob(images_hazy_path)
        self.images_airlight_list =glob(images_airlight_path)
        self.printName = printName
        self.returnName = returnName
        self.transform = make_transform(self.img_size)
    
    def __len__(self):
        return len(self.images_clear_list)
        
    def __getitem__(self,index):
        if self.printName:
            print(self.images_hazy_list[index])
            
        hazy_input, clear_input, airlight_input = load_item_3(self.images_hazy_list[index], self.images_clear_list[index], self.images_airlight_list[index], self.transform)
        if self.returnName:
            return hazy_input, clear_input, airlight_input, self.images_hazy_list[index]
        else:
            return hazy_input, clear_input, airlight_input
        
class D_Hazy_NYU_Dataset(Dataset):
    def __init__(self,path,img_size,printName=False,returnName=False):
        super().__init__()
        self.img_size = img_size
        images_clear_path = path+'/clear/*.bmp'
        images_hazy_path = path+'/hazy/*.bmp'
        images_airlight_path = path+'/airlight/*.bmp'
        self.images_clear_list=glob(images_clear_path)
        self.images_hazy_list =glob(images_hazy_path)
        self.images_airlight_list =glob(images_airlight_path)
        self.printName = printName
        self.returnName = returnName
        self.transform = make_transform(self.img_size)
    
    def __len__(self):
        return len(self.images_clear_list)
        
    def __getitem__(self,index):
        if self.printName:
            print(self.images_hazy_list[index])
            
        hazy_input, clear_input, airlight_input = load_item_3(self.images_hazy_list[index], self.images_clear_list[index], self.images_airlight_list[index], self.transform)
        if self.returnName:
            return hazy_input, clear_input, airlight_input, self.images_hazy_list[index]
        else:
            return hazy_input, clear_input, airlight_input
        
class D_Hazy_NYU_Dataset_With_Notation(Dataset):
    def __init__(self,path,img_size,printName=False,returnName=False):
        super().__init__()
        self.img_size = img_size
        images_clear_path = path+'/clear/*.bmp'
        images_hazy_path = path+'/hazy/*.bmp'
        images_airlight_path = path+'/airlight/*.bmp'
        images_depth_path = path+'/depth/*.bmp'
        
        self.images_clear_list=glob(images_clear_path)
        self.images_hazy_list =glob(images_hazy_path)
        self.images_airlight_list =glob(images_airlight_path)
        self.images_depth_lsit = glob(images_depth_path)
        
        self.printName = printName
        self.returnName = returnName
        self.transform = make_transform(self.img_size)
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
        return len(self.images_clear_list)
        
    def __getitem__(self,index):
        if self.printName:
            print(self.images_hazy_list[index])
            
        hazy_input, clear_input, airlight_input = load_item_3(self.images_hazy_list[index], self.images_clear_list[index], self.images_airlight_list[index], self.transform)
        depth_input = io.read_image(self.images_depth_lsit[index])
        depth_input = self.transform({"image": depth_input})["image"]
        if self.returnName:
            return hazy_input, clear_input, airlight_input, depth_input, self.images_hazy_list[index]
        else:
            return hazy_input, clear_input, airlight_input, depth_input
        
class O_Haze_Dataset(Dataset):
    def __init__(self,path,img_size,printName=False,returnName=False):
        super().__init__()
        self.img_size = img_size
        images_clear_path = path+'/clear/*.jpg'
        images_hazy_path = path+'/hazy/*.jpg'
        images_airlight_path = path+'/airlight/*.jpg'
        self.images_clear_list=glob(images_clear_path)
        self.images_hazy_list =glob(images_hazy_path)
        self.images_airlight_list =glob(images_airlight_path)
        self.printName = printName
        self.returnName = returnName
        self.transform = make_transform(self.img_size)
    
    def __len__(self):
        return len(self.images_clear_list)
        
    def __getitem__(self,index):
        if self.printName:
            print(self.images_hazy_list[index])
            
        hazy_input, clear_input, airlight_input = load_item_3(self.images_hazy_list[index], self.images_clear_list[index], self.images_airlight_list[index], self.transform)
        if self.returnName:
            return hazy_input, clear_input, airlight_input, self.images_hazy_list[index]
        else:
            return hazy_input, clear_input, airlight_input
    
class NH_Haze_Dataset(Dataset):
    def __init__(self,path,img_size,printName=False,returnName=False):
        super().__init__()
        self.img_size = img_size
        images_clear_path = path+'/clear/*.png'
        images_hazy_path = path+'/hazy/*.png'
        images_airlight_path = path+'/airlight/*.png'
        self.images_clear_list=glob(images_clear_path)
        self.images_hazy_list =glob(images_hazy_path)
        self.images_airlight_list =glob(images_airlight_path)
        self.printName = printName
        self.returnName = returnName
        self.transform = make_transform(self.img_size)
    
    def __len__(self):
        return len(self.images_clear_list)
    
    def __getitem__(self,index):
        if self.printName:
            print(self.images_hazy_list[index])
        hazy_input, clear_input, airlight_input = load_item_3(self.images_hazy_list[index], self.images_clear_list[index], self.images_airlight_list[index], self.transform)
        if self.returnName:
            return hazy_input, clear_input, airlight_input, self.images_hazy_list[index]
        else:
            return hazy_input, clear_input, airlight_input,
    
class Dense_Haze_Dataset(Dataset):
    def __init__(self,path,img_size,printName=False,returnName=False):
        super().__init__()
        self.img_size = img_size
        images_clear_path = path+'/clear/*.png'
        images_hazy_path = path+'/hazy/*.png'
        images_airlight_path = path+'/airlight/*.png'
        self.images_clear_list=glob(images_clear_path)
        self.images_hazy_list =glob(images_hazy_path)
        self.images_airlight_list =glob(images_airlight_path)
        self.printName = printName
        self.returnName = returnName
        self.transform = make_transform(self.img_size)
    
    def __len__(self):
        return len(self.images_clear_list)
    
    def __getitem__(self,index):
        if self.printName:
            print(self.images_hazy_list[index])
        hazy_input, clear_input, airlight_input = load_item_3(self.images_hazy_list[index], self.images_clear_list[index], self.images_airlight_list[index], self.transform)
        if self.returnName:
            return hazy_input, clear_input, airlight_input, self.images_hazy_list[index]
        else:
            return hazy_input, clear_input, airlight_input
            
class NYU_Dataset_With_Notation(Dataset):
    def __init__(self, path, img_size, printName=False, returnName=False):
        super().__init__()
        self.img_size = img_size
        h5s_path = path+'/*.h5'
        self.h5s_list = glob(h5s_path)
        self.printName = printName
        self.returnName = returnName
        self.transform = make_transform(img_size)
        
        self.air_resize = Resize(
            img_size[0],
            img_size[1],
            resize_target=None,
            keep_aspect_ratio=False,
            ensure_multiple_of=32,
            resize_method="minimal",
            image_interpolation_method=cv2.INTER_NEAREST,
        )
        
    def __len__(self):
        return len(self.h5s_list)
        
    def __getitem__(self,index):
        h5 = self.h5s_list[index]
        f = h5py.File(h5,'r')
        
        haze = f['haze'][:]
        clear = f['gt'][:]
        trans = f['trans'][:]
        airlight = f['ato'][:]
        
        haze_input  = self.transform({"image": haze})["image"]
        clear_input  = self.transform({"image": clear})["image"]
        airlight_input = self.transform({"image": airlight})["image"]
        trans_input = self.transform({"image": trans})["image"]
        
        if self.returnName:
            return haze_input, clear_input, airlight_input, trans_input, h5
        else :
            return haze_input, clear_input, airlight_input

class NYU_Dataset(Dataset):
    def __init__(self, path, img_size, printName=False, returnName=False):
        super().__init__()
        self.img_size = img_size
        h5s_path = path+'/*.h5'
        airlight_path = path+'_airlight/*.png'
        self.h5s_list = glob(h5s_path)
        self.airglith_list = glob(airlight_path)
        self.printName = printName
        self.returnName = returnName
        self.transform = make_transform(img_size)
        
        self.air_resize = Resize(
            img_size[0],
            img_size[1],
            resize_target=None,
            keep_aspect_ratio=False,
            ensure_multiple_of=32,
            resize_method="",
            image_interpolation_method=cv2.INTER_NEAREST,
        )
        
    def __len__(self):
        return len(self.h5s_list)
        
    def __getitem__(self,index):
        h5 = self.h5s_list[index]
        f = h5py.File(h5,'r')
        
        haze = f['haze'][:]
        clear = f['gt'][:]
        airlight = io.read_image(self.airglith_list[index])
        
        haze_input  = self.transform({"image": haze})["image"]
        clear_input  = self.transform({"image": clear})["image"]
        airlight_input = self.transform({"image": airlight})["image"]
        
        if(self.printName):
            print(h5)
        if self.returnName:
            return haze_input, clear_input, airlight_input, h5
        else:
            return haze_input, clear_input, airlight_input
        
class RESIDE_Beta_Dataset(Dataset):
    def __init__(self, path, img_size, printName=False, returnName=False ,verbose=False):
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
        self.transform = make_transform(img_size)
        
    def __len__(self):
        return len(self.images_hazy_lists) * self.images_count
        
    def __getitem__(self,index):
        haze = self.images_hazy_lists[index//self.images_count][index%self.images_count]
        clear = self.images_clear_list[index%self.images_count]
        airlight = self.images_airlight_list[index//self.images_count][index%self.images_count] if self.airlgiht_flag else None
        
        if self.printName:
            print(self.images_hazy_lists[index//self.images_count][index%self.images_count])
        
        #return load_item(haze,clear,self.img_size)
        #return load_item_2(haze,clear,self.transform)
        if airlight == None:
           return load_item_2(haze,clear,self.transform) 
        else:
            hazy_input, clear_input, airlight_input = load_item_3(haze,clear,airlight,self.transform)
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
        self.transform = make_transform(img_size)
        
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
        
        haze_input, clear_input, airlight_post = load_item_3(haze,clear, airlight_post,self.transform)
        if self.returnName:
            return haze_input, clear_input, airlight_post, depth_input, airlight_input, beta_input, haze
        else:
            return haze_input, clear_input, airlight_post, depth_input, airlight_input, beta_input
    
class NTIRE_Dataset(Dataset):
    def __init__(self, path, img_size, flag='train',verbose=False):
        super().__init__()
        self.OD = O_Haze_Dataset(path + f'/O_Haze/{flag}', img_size,verbose)
        self.DD = Dense_Haze_Dataset(path + f'/Dense_Haze/{flag}', img_size,verbose)
        self.ND = NH_Haze_Dataset(path + f'/NH_Haze/{flag}', img_size,verbose)
                
        print("O_Haze len:     ", self.OD.__len__())
        print("Dense_Haze len: ", self.DD.__len__())
        print("NH_Haze len:    ", self.ND.__len__())
        
    def __len__(self):
        return self.OD.__len__() + self.DD.__len__() + self.ND.__len__()

    def __getitem__(self, index):
        if index >= self.OD.__len__() and index < (self.OD.__len__()+self.DD.__len__()):
            haze, clear, airlight = self.DD[index-self.OD.__len__()]
        elif index >= (self.OD.__len__()+self.DD.__len__()):
            haze, clear, airlight = self.ND[index-(self.OD.__len__()+self.DD.__len__())]
        else:
            haze, clear, airlight = self.OD[index]
        return haze, clear, airlight
    
if __name__ == '__main__':
    print("ths is main")
    