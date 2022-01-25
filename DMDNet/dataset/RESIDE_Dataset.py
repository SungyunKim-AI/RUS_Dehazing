from glob import glob
import cv2
import mat73
import os
from torch.utils.data import Dataset
from utils.io import *


class RESIDE_Dataset(Dataset):
    def __init__(self, path, img_size, norm=True, verbose=False):
        super().__init__()
        self.path = path
        self.img_size = img_size
        self.norm = norm

        self.hazy_lists = []
        self.hazy_count = 0
        
        for hazy_folder in glob(path+'/hazy/*/'):
            if verbose:
                print(hazy_folder + ' dataset ready!')
            for hazy_image in glob(hazy_folder + '*.jpg'):
                self.hazy_lists.append(hazy_image)
                self.hazy_count+=1
        self.transform = make_transform(img_size, norm=norm)
        
    def __len__(self):
        return self.hazy_count
        # return 100
        
    def __getitem__(self,index):
        haze = self.hazy_lists[index]
        filename = os.path.basename(haze)
        airlight_input = np.array(float(filename.split('_')[1]))
        beta_input = np.array(float(filename.split('_')[2][:-4]))
        
        og_filename = filename.split('_')[0]
        
        clear = self.path + '/clear/' + og_filename + '.jpg'
        if not os.path.isfile(clear):
            clear = self.path + '/clear/' + og_filename + '.png'

        depth = self.path + '/depth/' + og_filename + '.mat'
        if os.path.isfile(depth):
            depth_input = mat73.loadmat(depth)
            depth_input = cv2.resize(depth_input['depth'], (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_CUBIC)
            depth_input = np.expand_dims(depth_input, axis=0)
            depth_input = depth_input.astype(np.float32)
        else:
            depth_input = np.array((1,self.img_size[0], self.img_size[1]))

        if self.norm:
            air_list = np.array([0.8, 0.85, 0.9, 0.95, 1.0])
            airlight_input = (airlight_input - air_list.mean()) / air_list.std()
        airlight_input = np.expand_dims(airlight_input, axis=0)
        beta_input = np.expand_dims(beta_input, axis=0)

        hazy_input, clear_input = load_item(haze, clear, self.transform)
        
        return hazy_input, clear_input, depth_input, airlight_input, beta_input, filename