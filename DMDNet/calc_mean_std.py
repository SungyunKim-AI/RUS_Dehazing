import cv2
from glob import glob
import numpy
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm



class RESIDE_Beta_Dataset(Dataset):
    def __init__(self, path, verbose=False):
        super().__init__()
        images_clear_path = path+'/clear/*.jpg'
        self.images_clear_list = glob(images_clear_path)
        
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
        
        haze = cv2.imread(haze)
        hazy_input = torch.Tensor(cv2.cvtColor(haze, cv2.COLOR_BGR2RGB)).float()
        
        clear = cv2.imread(clear)
        clear_input = torch.Tensor(cv2.cvtColor(clear, cv2.COLOR_BGR2RGB)).float()
        
        return hazy_input, clear_input
        
            
if __name__ == '__main__':
    dataRoot = 'C:/Users/IIPL/Desktop/data/RESIDE_beta/train'
    dataset = RESIDE_Beta_Dataset(dataRoot) 
    dataloader = DataLoader(dataset, shuffle=False, num_workers=2) 
    mean_hazy, std_hazy = torch.zeros(3), torch.zeros(3)
    mean_clear, std_clear = torch.zeros(3), torch.zeros(3)
    for hazy_input, clear_input in tqdm(dataloader):         
        
        for i in range(3): 
            mean_hazy[i] += hazy_input[:,i,:,:].mean() 
            mean_hazy[i] += hazy_input[:,i,:,:].std() 
            mean_hazy.div_(len(dataset)) 
            std_hazy.div_(len(dataset))
            
            mean_clear[i] += clear_input[:,i,:,:].mean() 
            std_clear[i] += clear_input[:,i,:,:].std() 
            mean_clear.div_(len(dataset)) 
            std_clear.div_(len(dataset))
    
    print(mean_hazy, std_hazy)      # tensor([7.5563e-13, 5.2434e-08, 3.6742e-03]) tensor([0., 0., 0.])
    print(mean_clear, std_clear)    # tensor([5.3374e-13, 3.6720e-08, 2.6036e-03]) tensor([2.3084e-13, 1.5983e-08, 1.0928e-03])