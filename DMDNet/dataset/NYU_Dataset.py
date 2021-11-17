import cv2
import h5py
from glob import glob

from torch.utils.data import Dataset

from dpt.transforms import Resize
from dataset import utils
from util import io

class NYU_Dataset_With_Notation(Dataset):
    def __init__(self, path, img_size, printName=False, returnName=False):
        super().__init__()
        self.img_size = img_size
        h5s_path = path+'/*.h5'
        self.h5s_list = glob(h5s_path)
        self.printName = printName
        self.returnName = returnName
        self.transform = utils.make_transform(img_size)
        
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
        self.transform = utils.make_transform(img_size)
        
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
        