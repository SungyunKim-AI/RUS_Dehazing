"""
Reference Paper
Image Haze Removal Using Airlight White Correction, Local Light Filter, and Aerial Perspective Prior
Yan-Tsung Peng et al.
"""
import os, cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import time


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
    def AWC(self, image, is_Image='True', color='BGR'):
        # 1. RGB to HSV
        if not is_Image:
            image = cv2.imread(image)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            if color == 'RGB':
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color == 'BGR':
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            else:
                raise ValueError('color is RGB or BGR')
        hue, saturation, value = cv2.split(hsv)
        
        # 2. Hue histogram labeling
        val, cnt = np.unique(hue, return_counts=True)
        img_size = hue.shape[0] * hue.shape[1]
        prob = cnt / img_size    # PMF
        T_H = 1 / 360
        
        prob[prob > T_H] = 1
        prob[prob <= T_H] = 0
        
        binarized_hue = np.zeros(hue.shape, int)
        for v in val:
            row, col = np.where(hue == v)
            for i in range(len(row)):
                binarized_hue[row[i]][col[i]] = 1
        
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
    def LLF(self, image, mReturn):
        # 1. PMF of minimum channel calculation
        R, G, B = cv2.split(image)
        min_channel = np.amin([R, G, B],0)
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
        
        # 3. Airlight color estimation to RGB to RGB
        if mReturn == 'RGB':
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
        
        
        elif mReturn == 'gray':
            # 3. Airlight color estimation to RGB to Gray-Scale
            threshold = 0.0
            idx = np.array([])
            for i in range(255,-1, -1):
                if P_m[i] > 0.0:
                    threshold += P_m[i]
                    idx = np.append(idx, i)
                    if threshold >= 0.01:
                        break
            
            mean_val = round(np.mean(idx))
            airlight = np.full((image.shape[0], image.shape[1]), mean_val).astype('uint8')
            
            return airlight,  mean_val
        
        else:
            raise ValueError('mReturn must be RGB or gray')
          
  
if __name__ == "__main__":
    if not cv2.useOptimized():
        cv2.setUseOptimized(True)
    
    start = time.time()
    image = cv2.imread("test01.jpg")
    # image = cv2.resize(image, dsize=(0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    airlight_module = Airlight_Module()
    image = airlight_module.AWC(image)
    airlight, single_val = airlight_module.LLF(image, mReturn='gray')    # mReturn = 'RGB' or 'gray
    print("Single Value : ", single_val)
    print("Operation Time(s) : ", round(time.time() - start, 3))
    
    if airlight.shape[0] == 3:
        airlight = cv2.cvtColor(airlight, cv2.COLOR_RGB2BGR) 
    
    cv2.imshow(f"Airlight", airlight)
    cv2.waitKey(0)
    
    