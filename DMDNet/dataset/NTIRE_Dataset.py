from glob import glob
from torch.utils.data import Dataset
from utils import *

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