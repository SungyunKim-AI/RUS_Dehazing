import glob, os
import torch
import cv2
import torchvision.transforms.functional as F
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
import util.io

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
    

class BeDDE_Dataset(torch.utils.data.Dataset):
    def __init__(self,path,img_size,printName=False):
        super().__init__()
        self.img_size = img_size
        images_hazy_path = path+'/*/fog/*.png'
        self.images_hazy_list =glob.glob(images_hazy_path)
        self.printName = printName
    
    def __len__(self):
        return len(self.images_hazy_list)
    
    def __getitem__(self,index):
        image_hazy_path = self.images_hazy_list[index]
        image_clear_path_slice = image_hazy_path.split('\\')
        image_clear_path = ''
        for i in range(len(image_clear_path_slice)-2):
            image_clear_path+=(image_clear_path_slice[i]+'\\')
        image_clear_path = image_clear_path+'gt/'+image_clear_path_slice[-3]+'_clear.png'
        haze, clear = load_item(image_hazy_path, image_clear_path,self.img_size)
        if self.printName:
            print(image_hazy_path)
        return haze, clear

class O_Haze_Dataset(torch.utils.data.Dataset):
    def __init__(self,path,img_size,printName=False):
        super().__init__()
        self.img_size = img_size
        images_clear_path = path+'/clear/*.jpg'
        images_hazy_path = path+'/hazy/*.jpg'
        self.images_clear_list=glob.glob(images_clear_path)
        self.images_hazy_list =glob.glob(images_hazy_path)
        self.printName = printName
        
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.transform = Compose(
            [
                Resize(
                    self.img_size[0],
                    self.img_size[1],
                    resize_target=None,
                    keep_aspect_ratio=False,
                    ensure_multiple_of=32,
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                normalization,
                PrepareForNet(),
            ]
        )
    
    def __len__(self):
        return len(self.images_clear_list)
    
    def __getitem__(self,index):
        #haze, clear = load_item(self.images_hazy_list[index], self.images_clear_list[index], self.img_size)
        
        haze = util.io.read_image(self.images_hazy_list[index])
        clear = util.io.read_image(self.images_clear_list[index])
        
        haze_input  = self.transform({"image": haze})["image"]
        clear_input = self.transform({"image": clear})["image"]
        
        if self.printName:
            print(self.images_hazy_list[index])
        
        return haze_input, clear_input
    
class NH_Haze_Dataset(torch.utils.data.Dataset):
    def __init__(self,path,img_size,printName=False):
        super().__init__()
        self.img_size = img_size
        images_clear_path = path+'/clear/*.png'
        images_hazy_path = path+'/hazy/*.png'
        self.images_clear_list=glob.glob(images_clear_path)
        self.images_hazy_list =glob.glob(images_hazy_path)
        self.printName = printName
    
    def __len__(self):
        return len(self.images_clear_list)
    
    def __getitem__(self,index):
        haze, clear = load_item(self.images_hazy_list[index], self.images_clear_list[index], self.img_size)
        if self.printName:
            print(self.images_hazy_list[index])
        return haze, clear
        
class RESIDE_Beta_Dataset(torch.utils.data.Dataset):
    def __init__(self, path, img_size, printName=False, verbose=True):
        super().__init__()
        self.img_size = img_size
        images_clear_path = path+'/clear/*.jpg'
        self.images_clear_list = glob.glob(images_clear_path)
        self.printName = printName
        
        images_hazy_folders_path = path+'/hazy/*/'
        self.images_hazy_lists = []
        for images_hazy_folder in glob.glob(images_hazy_folders_path):
            if verbose:
                print(images_hazy_folder + ' dataset ready!')
            self.images_hazy_lists.append(glob.glob(images_hazy_folder+'*.jpg'))
        
        self.images_count = len(self.images_hazy_lists[0])
        
    def __len__(self):
        return len(self.images_hazy_lists) * self.images_count
        #return 10
        
    def __getitem__(self,index):
        haze, clear = load_item(self.images_hazy_lists[index//self.images_count][index%self.images_count],
                                self.images_clear_list[index%self.images_count],
                                self.img_size)
        if self.printName:
            print(self.images_hazy_lists[index//self.images_count][index%self.images_count])
        return haze, clear


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, img_size, train_flag='train', verbose=True):
        super().__init__()
        self.RBTD = RESIDE_Beta_Dataset(path + f'/RESIDE-beta/{train_flag}', img_size, verbose)
        self.OHTD = O_Haze_Dataset(path + f'/O-Haze/{train_flag}', img_size)
        self.NHTD = NH_Haze_Dataset(path + f'/NH-Haze/{train_flag}', img_size)
                
        # print("RESIDE-beta len : ", self.RBTD.__len__())
        # print("O-Haze len      : ", self.OHTD.__len__())
        # print("NH-Haze len     : ", self.NHTD.__len__())
        
    def __len__(self):
        return self.RBTD.__len__() + self.OHTD.__len__() + self.NHTD.__len__()

    def __getitem__(self, index):
        if index >= self.RBTD.__len__() and index <= (self.RBTD.__len__()+self.OHTD.__len__()):
            haze, clear = self.OHTD[index-self.RBTD.__len__()]
        elif index >= (self.RBTD.__len__()+self.OHTD.__len__()):
            haze, clear = self.NHTD[index-(self.RBTD.__len__()+self.OHTD.__len__())]
        else:
            haze, clear = self.RBTD[index]
        return haze, clear
    
if __name__ == '__main__':
    # data_path = 'D:/data'
    data_path = 'C:/Users/IIPL/Desktop/data'
    # data_path = '/Users/sungyoon-kim/Documents/GitHub/RUS_Dehazing/data_sample'
    
    train_set = Dataset(data_path, [256,256],train_flag='train') 
    test_set = Dataset(data_path, [256,256],train_flag='test')
    
    print("Train Set : ", train_set.__len__())  # 70000 + 40 + 50 = 70090
    print("Test Set : ", test_set.__len__())    # 2135 + 5 + 5 = 2145
    