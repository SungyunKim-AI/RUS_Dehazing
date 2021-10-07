import torch
import cv2
import torchvision.transforms.functional as F
import glob
import random

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
    
    def __len__(self):
        return len(self.images_clear_list)
    
    def __getitem__(self,index):
        haze, clear = load_item(self.images_hazy_list[index], self.images_clear_list[index], self.img_size)
        if self.printName:
            print(self.images_hazy_list[index])
        return haze, clear
    
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
    def __init__(self,path,folders_num,img_size,printName=False):
        super().__init__()
        self.img_size = img_size
        images_clear_path = path+'/clear/*.jpg'
        self.images_clear_list = glob.glob(images_clear_path)
        self.printName = printName
        
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
        #return 10
        
    def __getitem__(self,index):
        haze, clear = load_item(self.images_hazy_lists[index//self.images_count][index%self.images_count],
                                self.images_clear_list[index%self.images_count],
                                self.img_size)
        if self.printName:
            print(self.images_hazy_lists[index//self.images_count][index%self.images_count])
        return haze, clear


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, img_size, train_flag='train'):
        super().__init__()
        self.RBTD = RESIDE_Beta_Dataset(path + f'/RESIDE-beta/{train_flag}',range(35), img_size)
        self.OHTD = O_Haze_Dataset(path + f'/O-Haze/{train_flag}', img_size)
        self.NHTD = NH_Haze_Dataset(path + f'/NH-Haze/{train_flag}', img_size)
        
        
    def __len__(self):
        return self.RBTD.__len__() + self.OHTD.__len__() + self.NHTD.__len__()

    def __getitem__(self, index):
        if index >= self.RBTD.__len__() and index <= (self.RBTD.__len__()+self.OHTD.__len__()):
            haze, clear = self.OHTD[index-self.RBTD.__len__()]
        elif index >= (self.RBTD.__len__()+self.OHTD.__len__()):
            haze, clear = self.OHTD[index-(self.RBTD.__len__()+self.OHTD.__len__())]
        else:
            haze, clear = self.RBTD[index]
        return haze, clear
    
if __name__ == '__main__':
    # train_set = Dataset('D:/data', train_flag='train')
    # test_set = Dataset('D:/data', train_flag='test')
    
    train_set = Dataset('C:/Users/IIPL/Desktop/data', [256,256],train_flag='train')
    test_set = Dataset('C:/Users/IIPL/Desktop/data', [256,256],train_flag='test')
    
    print("Train Set : ", train_set.__len__())  # 70000 + 40 + 50 = 70090
    print("Test Set : ", test_set.__len__())    # 2135 + 5 + 5 = 2145
    