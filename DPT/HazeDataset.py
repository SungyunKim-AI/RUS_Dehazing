import glob, os
import h5py
from numpy import float32
import torch
import cv2
import torchvision.transforms.functional as F
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
import util.io
import mat73

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
    haze = util.io.read_image(haze)
    clear = util.io.read_image(clear)
    
    haze_input  = transform({"image": haze})["image"]
    clear_input = transform({"image": clear})["image"]

    return haze_input, clear_input

def load_item_3(haze, clear, airlight, transform):
    haze = util.io.read_image(haze)
    clear = util.io.read_image(clear)
    airlight = util.io.read_image(airlight)
    
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
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )
    return transform
    

class BeDDE_Dataset(torch.utils.data.Dataset):
    def __init__(self,path,img_size,printName=False):
        super().__init__()
        self.img_size = img_size
        images_hazy_path = path+'/*/fog/*.png'
        self.images_hazy_list =glob.glob(images_hazy_path)
        self.printName = printName
        self.transform = make_transform(self.img_size)
    
    def __len__(self):
        return len(self.images_hazy_list)
    
    def __getitem__(self,index):
        image_hazy_path = self.images_hazy_list[index]
        if self.printName:
            print(image_hazy_path)
            
        image_clear_path_slice = image_hazy_path.split('\\')
        image_clear_path = ''
        for i in range(len(image_clear_path_slice)-2):
            image_clear_path+=(image_clear_path_slice[i]+'\\')
        image_clear_path = image_clear_path+'gt/'+image_clear_path_slice[-3]+'_clear.png'
        
        return load_item_2(image_hazy_path,image_clear_path,self.transform)

class O_Haze_Dataset(torch.utils.data.Dataset):
    def __init__(self,path,img_size,printName=False):
        super().__init__()
        self.img_size = img_size
        images_clear_path = path+'/clear/*.jpg'
        images_hazy_path = path+'/hazy/*.jpg'
        images_airlight_path = path+'/airlight/*.jpg'
        self.images_clear_list=glob.glob(images_clear_path)
        self.images_hazy_list =glob.glob(images_hazy_path)
        self.images_airlight_list =glob.glob(images_airlight_path)
        self.printName = printName
        self.transform = make_transform(self.img_size)
    
    def __len__(self):
        return len(self.images_clear_list)
        
    def __getitem__(self,index):
        if self.printName:
            print(self.images_hazy_list[index])
        #return load_item(self.images_hazy_list[index], self.images_clear_list[index], self.img_size)
        return load_item_3(self.images_hazy_list[index], self.images_clear_list[index], self.images_airlight_list[index], self.transform)
    
class NH_Haze_Dataset(torch.utils.data.Dataset):
    def __init__(self,path,img_size,printName=False):
        super().__init__()
        self.img_size = img_size
        images_clear_path = path+'/clear/*.png'
        images_hazy_path = path+'/hazy/*.png'
        images_airlight_path = path+'/airlight/*.png'
        self.images_clear_list=glob.glob(images_clear_path)
        self.images_hazy_list =glob.glob(images_hazy_path)
        self.images_airlight_list =glob.glob(images_airlight_path)
        self.printName = printName
        self.transform = make_transform(self.img_size)
    
    def __len__(self):
        return len(self.images_clear_list)
    
    def __getitem__(self,index):
        if self.printName:
            print(self.images_hazy_list[index])
        #return load_item(self.images_hazy_list[index], self.images_clear_list[index], self.img_size)
        return load_item_3(self.images_hazy_list[index], self.images_clear_list[index], self.images_airlight_list[index], self.transform)
    
class Dense_Haze_Dataset(torch.utils.data.Dataset):
    def __init__(self,path,img_size,printName=False):
        super().__init__()
        self.img_size = img_size
        images_clear_path = path+'/clear/*.png'
        images_hazy_path = path+'/hazy/*.png'
        images_airlight_path = path+'/airlight/*.png'
        self.images_clear_list=glob.glob(images_clear_path)
        self.images_hazy_list =glob.glob(images_hazy_path)
        self.images_airlight_list =glob.glob(images_airlight_path)
        self.printName = printName
        self.transform = make_transform(self.img_size)
    
    def __len__(self):
        return len(self.images_clear_list)
    
    def __getitem__(self,index):
        if self.printName:
            print(self.images_hazy_list[index])
        #return load_item(self.images_hazy_list[index], self.images_clear_list[index], self.img_size)
        return load_item_3(self.images_hazy_list[index], self.images_clear_list[index], self.images_airlight_list[index], self.transform)
    
class NYU_Dataset_with_Notation(torch.utils.data.Dataset):
    def __init__(self, path, img_size, printName=False, verbose=True):
        super().__init__()
        self.img_size = img_size
        h5s_path = path+'/*.h5'
        self.h5s_list = glob.glob(h5s_path)
        self.printName = printName
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
        
        return haze_input, clear_input, airlight_input, trans_input

class NYU_Dataset(torch.utils.data.Dataset):
    def __init__(self, path, img_size, printName=False, verbose=True):
        super().__init__()
        self.img_size = img_size
        h5s_path = path+'/*.h5'
        self.h5s_list = glob.glob(h5s_path)
        self.printName = printName
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
        
        haze_input  = self.transform({"image": haze})["image"]
        clear_input  = self.transform({"image": clear})["image"]
        
        if(self.printName):
            print(h5)
        
        return haze_input, clear_input
    
    
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
        self.transform = make_transform(img_size)
        
    def __len__(self):
        return len(self.images_hazy_lists) * self.images_count
        #return 10
        
    def __getitem__(self,index):
        haze = self.images_hazy_lists[index//self.images_count][index%self.images_count]
        clear = self.images_clear_list[index%self.images_count]
        
        if self.printName:
            print(self.images_hazy_lists[index//self.images_count][index%self.images_count])
        
        #return load_item(haze,clear,self.img_size)
        return load_item_2(haze,clear,self.transform)
    
class RESIDE_Beta_Dataset_With_Notation(torch.utils.data.Dataset):
    def __init__(self, path, img_size, printName=False, verbose=True):
        super().__init__()
        self.img_size = img_size
        self.printName = printName
        
        images_clear_path = path+'/clear/*.jpg'
        self.images_clear_list = glob.glob(images_clear_path)
        
        images_hazy_folders_path = path+'/hazy/*/'
        self.images_hazy_lists = []
        for images_hazy_folder in glob.glob(images_hazy_folders_path):
            if verbose:
                print(images_hazy_folder + ' dataset ready!')
            self.images_hazy_lists.append(glob.glob(images_hazy_folder+'*.jpg'))
            
        depth_path = path+'/depth/*.mat'
        self.depth_list = glob.glob(depth_path)
        
        self.images_count = len(self.images_hazy_lists[0])
        self.transform = make_transform(img_size)
        
        self.depth_resize = Resize(
                img_size[0],
                img_size[1],
                resize_target=None,
                keep_aspect_ratio=False,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_NEAREST,
            )
        
    def __len__(self):
        return len(self.images_hazy_lists) * self.images_count
        #return 10
        
    def __getitem__(self,index):
        haze = self.images_hazy_lists[index//self.images_count][index%self.images_count]
        token = haze.split('_')
        airlight = token[2]
        beta = token[3].split('\\')[0]
        
        airlight_input, beta_input = float(airlight), float(beta)
        
        clear = self.images_clear_list[index%self.images_count]
        depth_mat = mat73.loadmat(self.depth_list[index%self.images_count])
        depth_input = self.depth_resize({"image": depth_mat['depth']})["image"]
        
        
        if self.printName:
            print(self.images_hazy_lists[index//self.images_count][index%self.images_count])
        
        haze_input, clear_input = load_item_2(haze,clear,self.transform)
        return haze_input, clear_input, depth_input, airlight_input, beta_input

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
    
class NTIRE_Dataset(torch.utils.data.Dataset):
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
    # data_path = 'D:/data'
    data_path = 'C:/Users/IIPL/Desktop/data'
    # data_path = '/Users/sungyoon-kim/Documents/GitHub/RUS_Dehazing/data_sample'
    
    train_set = Dataset(data_path, [256,256],train_flag='train') 
    test_set = Dataset(data_path, [256,256],train_flag='test')
    
    print("Train Set : ", train_set.__len__())  # 70000 + 40 + 50 = 70090
    print("Test Set : ", test_set.__len__())    # 2135 + 5 + 5 = 2145
    