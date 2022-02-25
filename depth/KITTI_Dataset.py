from glob import glob
import cv2
import mat73
import os
from torch.utils.data import Dataset
from utils.io import *
import numpy as np


class KITTI_Dataset(Dataset):
    def __init__(self, path, img_size, norm=True, verbose=False):
        super().__init__()
        self.path = path
        self.img_size = img_size
        self.norm = norm

        self.hazy_lists = []
        self.hazy_count = 0
        
        # for hazy_folder in glob(path+'/hazy/*/'):
        #     if verbose:
        #         print(hazy_folder + ' dataset ready!')
        #     for hazy_image in glob(hazy_folder + '*.png'):
        #         self.hazy_lists.append(hazy_image)
        #         self.hazy_count+=1
        
        hazy_folder =path+'/hazy/0.06/'
        if verbose:
            print(hazy_folder + ' dataset ready!')
        for hazy_image in glob(hazy_folder + '*.png'):
            self.hazy_lists.append(hazy_image)
            self.hazy_count+=1
        
        self.transform = make_transform(img_size, norm=norm)
        #self.airlights = np.load(path+'/airlight.npz')['data']
        
    def __len__(self):
        return self.hazy_count
        # return 100
        
    def __getitem__(self,index):
        haze = self.hazy_lists[index]
        filename = os.path.basename(haze)
        airlight_input = np.array(float(filename.split('_')[1]))
        beta_input = np.array(float(filename.split('_')[2][:-4]))
        
        og_filename = filename.split('_')[0]
        
        clear = self.path + '/clear/' + og_filename + '.png'

        depth = self.path + '/dense_depth/' + og_filename + '.npy'
        if os.path.isfile(depth):
            depth_input = np.load(depth)
            depth_input = cv2.resize(depth_input, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_CUBIC)
            depth_input = np.expand_dims(depth_input, axis=0)
            depth_input = depth_input.astype(np.float32)
        else:
            depth_input = np.array((1,self.img_size[0], self.img_size[1]))
        airlight_input = np.expand_dims(airlight_input, axis=0)
        beta_input = np.expand_dims(beta_input, axis=0)

        hazy_input, clear_input = load_item(haze, clear, self.transform)
        
        return hazy_input, clear_input, depth_input, airlight_input, beta_input, filename


class KITTI_Dataset_dehazed(KITTI_Dataset):
    def __init__(self, path, img_size, model_name, norm=True, verbose=False):
        super().__init__(path, img_size, norm, verbose)
        self.model_name = model_name
        
    def __len__(self):
        return self.hazy_count
        # return 100
        
    def __getitem__(self, index):
        filename = os.path.basename(self.hazy_lists[index])[:-4]
        og_filename = filename.split('_')[0]
        
        # hazy, clear and dehazed images
        haze = self.hazy_lists[index]
        clear = self.path + '/clear/' + og_filename + '.png'
        dehazed = f'D:/data/output_dehaze/KITTI_{self.model_name}/{filename}.png'
        hazy_input, clear_input, dehazed_input = load_item_2(haze, clear, dehazed, self.transform)
        
        # ground-truth depth map
        depth = self.path + '/dense_depth/' + og_filename + '.npy'
        if os.path.isfile(depth):
            depth_input = np.load(depth)
            depth_input = cv2.resize(depth_input, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_CUBIC)
            depth_input = np.expand_dims(depth_input, axis=0)
            depth_input = depth_input.astype(np.float32)
        else:
            depth_input = np.array((1,self.img_size[0], self.img_size[1]))
        
        # atmospheric light
        airlight_input = np.array(float(filename.split('_')[1]))
        airlight_input = np.expand_dims(airlight_input, axis=0)
        
        # attenuation coefficient
        beta_input = np.array(float(filename.split('_')[-1]))
        beta_input = np.expand_dims(beta_input, axis=0)
        
        return hazy_input, clear_input, dehazed_input, depth_input, airlight_input, beta_input, filename