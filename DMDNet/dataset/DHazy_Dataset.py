from glob import glob
from torch.utils.data import Dataset
from dpt.transforms import Resize
from utils import *

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
        