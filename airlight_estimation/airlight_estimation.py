"""
Reference Paper
Image Haze Removal Using Airlight White Correction, Local Light Filter, and Aerial Perspective Prior
Yan-Tsung Peng et al.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn

import os
from glob import glob
from tqdm import tqdm

def show_plt(x, y):
    plt.bar(x, y, align='center')
    plt.xlabel('deg.')
    plt.xlim([x[0], x[-1]])
    plt.ylabel('prob.')
    for i in range(len(y)):
        plt.vlines(x[i], 0, y[i])
    
    plt.show()
    
def show_img(imgName, img):
    cv2.imshow(imgName, img.astype('uint8'))
    cv2.waitKey(0)
    
class Airlight_Module():
    def __init__(self, color_cast_threshold=5):
        self.color_cast_threshold = color_cast_threshold

    # Airlight White Correction (AWC)
    def AWC(self, image):
        if os.path.isfile(image):
            image = cv2.imread(image)
        
        # 1. RGB to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        
        # 2. Hue histogram labeling
        val, cnt = np.unique(hue, return_counts=True)
        img_size = hue.shape[0] * hue.shape[1]
        prob = cnt / img_size    # PMF
        T_H = 1 / 360
        
        prob[prob > T_H] = 1
        prob[prob <= T_H] = 0
        binarized_hue = np.zeros(hue.shape, int)
        for i in range(hue.shape[0]):
            for j in range(hue.shape[1]):
                hue[i][j]
        
        for i in range(hue.shape[0]):
            for j in range(hue.shape[1]):
                idx = np.where(val == hue[i][j])
                if prob[idx] > T_H:
                    binarized_hue[i][j] = 1
        
        # 3. Color cast attenuation
        binarized_img = binarized_hue.astype('uint8')
        num_labels, _ = cv2.connectedComponents(binarized_img)   # Connected Component Labeling (CCL)
        if num_labels < self.color_cast_threshold:   # has color cast
            flatten_s = saturation.flatten()
            idx = int(len(flatten_s) / 100)
            least_idx = np.argpartition(flatten_s, idx)     # bottom 1% least saturated pixels
            least_value = flatten_s[least_idx[:idx]].mean()
            
            I_s = saturation - least_value
            I_s[I_s < 0] = 0
            saturation = I_s.astype('uint8')
        
        # 4. HSV to RGB
        hsv = cv2.merge((hue, saturation, value))
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) 
        
        return rgb
            
    # Local Light Filter (LLF)
    def LLF(self, image):
        # 1. PMF of minimum channel calculation
        R, G, B = cv2.split(image)
        min_channel = np.minimum(R, G, B)
        # cv2.imshow('Minimum channel', min_channel.astype('uint8'))
        # cv2.waitKey(0)
        
        val, cnt = np.unique(min_channel, return_counts=True)
        img_size = image.shape[0] * image.shape[1]
        prob = cnt / img_size    # PMF
        
        l = np.zeros((256))
        for i, v in enumerate(val):
            l[v] = prob[i]
        # show_plt(range(256), l)
        
        # 2. 1D minimum filter
        P_m = []
        l = np.insert(l, 0, np.array([1,1,1,1,1]))
        l = np.append(l, np.array([1,1,1,1,1]))
        for i in range(0+5, 255+6):
            P_m.append(l[i-5:i+6].min())
        P_m = np.array(P_m)
        # show_plt(range(256), P_m)
        
        # 3. Airlight color estimation
        threshold = 0.0
        idx = []
        for i in range(255,-1, -1):
            if P_m[i] > 0.0:
                threshold += P_m[i]
                idx.append(i)
                if threshold >= 0.01:
                    break

        avg = {'r': [], 'g': [], 'b': []}
        RGB = {'r': R, 'g': G, 'b': B}
        for i in idx:
            row, col = np.where(min_channel == i)
            for j, _ in enumerate(row):
                for c in ['r','g','b']:
                    avg[c].append(RGB[c][row[j]][col[j]])
        
        
        avgR = round(np.array(avg['r']).mean())
        avgG = round(np.array(avg['g']).mean())
        avgB = round(np.array(avg['b']).mean())
        
        r = np.full((image.shape[0], image.shape[1]), avgR).astype('uint8')
        g = np.full((image.shape[0], image.shape[1]), avgG).astype('uint8')
        b = np.full((image.shape[0], image.shape[1]), avgB).astype('uint8')
        rgb = cv2.merge((r, g, b))
        
        return rgb, [avgR, avgG, avgB]


def data_loader_NYU(path):
    hazy_path = os.path.join(path, 'hazy')
    hazy_list = glob(hazy_path + '/*')
    
    airlight_path = os.path.join(path, 'airlight')
    airlight_list = glob(airlight_path + '/*')
    
    data_list = []
    for i in range(len(hazy_list)):
        data_list.append([hazy_list[i], airlight_list[i]])
    print("Data Length : ", len(data_list))
    
    return data_list

def data_loader_OHaze(path):
    hazy_path = os.path.join(path, 'hazy')
    hazy_list = glob(hazy_path + '/*')
    
    return hazy_list
    
    
        
  
if __name__ == "__main__":
    if not cv2.useOptimized():
        cv2.setUseOptimized(True)
        
    path = '/Users/IIPL/Desktop/data/O-Haze'
    data_loader = data_loader_OHaze(path)
    
    airlight_module = Airlight_Module()
    for hazy in tqdm(data_loader):
        imgname = os.path.basename(hazy)
        
        input_hazy = cv2.imread(hazy)
        input_hazy2 = airlight_module.AWC(input_hazy)
        airlight_hat, _ = airlight_module.LLF(input_hazy2)
    
        cv2.imwrite(f'airlight_validate/O-Haze/{imgname}', airlight_hat)
    
    
    # criterion = nn.MSELoss()
    # loss = 0.0
    # for hazy, airlight in tqdm(data_list):
    #     imgname = os.path.basename(hazy)
    #     input_hazy = cv2.imread(hazy)
    #     airlight_module = Airlight_Module()
    #     input_hazy2 = airlight_module.AWC(input_hazy)
    #     airlight_hat, _ = airlight_module.LLF(input_hazy2)
        
    #     airlight_GT = cv2.imread(airlight)
    #     airlight_hat_tensor = torch.Tensor(airlight_hat).unsqueeze(0)
    #     airlight_GT_tensor = torch.Tensor(airlight_GT).unsqueeze(0)
    #     loss += criterion(airlight_hat_tensor, airlight_GT_tensor).item()
        
    #     cv2.imwrite(f'airlight_validate/O-Haze/{imgname}', airlight_hat)
        
    # print("Average Loss : ", loss/len(data_list))
    
    # start = time.time()
    # # image = "test01.jpg"
    # image = cv2.imread("test01.jpg")
    # airlight_module = Airlight_Module()
    # image = airlight_module.AWC(image)
    # airlight, [R, G, B] = airlight_module.LLF(image)
    # print("time :", time.time() - start)
    
    # cv2.imshow(f"Airlight ({R}, {G}, {B})", airlight)
    # cv2.waitKey(0)
    
    