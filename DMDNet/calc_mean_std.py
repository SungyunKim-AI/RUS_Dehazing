from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from util import io
from tqdm import tqdm



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
        
    def __len__(self):
        return len(self.images_hazy_lists) * self.images_count
        
    def __getitem__(self,index):
        haze = self.images_hazy_lists[index//self.images_count][index%self.images_count]
        clear = self.images_clear_list[index%self.images_count]
        airlight = self.images_airlight_list[index//self.images_count][index%self.images_count] if self.airlgiht_flag else None
        
        if self.printName:
            print(self.images_hazy_lists[index//self.images_count][index%self.images_count])
        
        hazy_input = io.read_image(haze)
        clear_input = io.read_image(clear)
        
        return hazy_input, clear_input
        
            
if __name__ == '__main__':
    dataRoot = 'C:/Users/IIPL/Desktop/data/RESIDE_beta/train'
    dataset = RESIDE_Beta_Dataset(dataRoot,[256, 256], printName=False, returnName=False) 
    dataloader = DataLoader(dataset, shuffle=False) 
    mean_hazy, std_hazy = torch.zeros(3), torch.zeros(3)
    mean_clear, std_clear = torch.zeros(3), torch.zeros(3)
    for inputs, clear_input in tqdm(dataloader): 
        for i in range(3): 
            mean_hazy[i] += inputs[:,i,:,:].mean() 
            mean_hazy[i] += inputs[:,i,:,:].std() 
            mean_hazy.div_(len(dataset)) 
            std_hazy.div_(len(dataset))
            
            mean_clear[i] += inputs[:,i,:,:].mean() 
            std_clear[i] += inputs[:,i,:,:].std() 
            mean_clear.div_(len(dataset)) 
            std_clear.div_(len(dataset))
    
    print(mean_hazy, std_hazy)
    print(mean_clear, std_clear)