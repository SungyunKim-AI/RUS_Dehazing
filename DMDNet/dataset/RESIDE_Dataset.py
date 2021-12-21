from glob import glob
import cv2
import mat73
import os
from torch.utils.data import Dataset
from utils.io import *


class RESIDE_Beta_Dataset(Dataset):
    def __init__(self, path, img_size, norm=True, verbose=False):
        super().__init__()
        self.img_size = img_size
        self.norm = norm

        self.clear_list = glob(path + '/clear/*.jpg')
        self.depth_list = glob(path + '/depth/*.mat')
        
        self.hazy_lists = []
        for hazy_folder in glob(path+'/hazy/*/'):
            if verbose:
                print(hazy_folder + ' dataset ready!')
            self.hazy_lists.append(glob(hazy_folder + '*.jpg'))
        
        self.images_count = len(self.hazy_lists[0])
        # mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152]
        self.transform = make_transform(img_size, norm=norm)
        
    def __len__(self):
        return len(self.hazy_lists) * self.images_count
        
    def __getitem__(self,index):
        haze = self.hazy_lists[index//self.images_count][index%self.images_count]
        clear = self.clear_list[index%self.images_count]

        GT_depth = mat73.loadmat(self.depth_list[index%self.images_count])
        GT_depth = cv2.resize(GT_depth['depth'], (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_CUBIC)
        GT_depth = np.expand_dims(GT_depth, axis=0)
        
        GT_airlight = np.array(float(os.path.basename(haze).split('_')[-2]))
        GT_beta = np.array(float(haze.split('_')[3].split('\\')[0]))

        if self.norm:
            air_list = np.array([0.8, 0.85, 0.9, 0.95, 1.0])
            GT_airlight = (GT_airlight - air_list.mean()) / air_list.std()
        GT_airlight = np.expand_dims(GT_airlight, axis=0)

        hazy_input, clear_input = load_item(haze, clear, self.transform)
        
        return hazy_input, clear_input, GT_depth, GT_airlight, GT_beta, haze