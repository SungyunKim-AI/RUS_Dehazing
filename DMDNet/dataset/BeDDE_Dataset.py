from glob import glob
from torch.utils.data import Dataset
from utils import *

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
            