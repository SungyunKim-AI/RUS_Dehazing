"""
Reference Paper
Image Haze Removal Using Airlight White Correction, Local Light Filter, and Aerial Perspective Prior
Yan-Tsung Peng et al.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def show_plt(x, y):
    plt.bar(x, y, align='center')
    plt.xlabel('deg.')
    plt.xlim([0, 359])
    plt.ylabel('prob.')
    for i in range(len(y)):
        plt.vlines(x[i], 0, y[i])
    
    plt.show()

# Airlight White Correction (AWC)
def AWC_Module(image, color_cast_threshold=5):
    # 1. RGB to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    
    # 2. Hue histogram labeling
    val, cnt = np.unique(hue, return_counts=True)
    img_size = hue.shape[0] * hue.shape[1]
    prob = cnt / img_size    # PMF
    T_H = 1 / 360
    binarized_hue = np.zeros(hue.shape, int)
    for i in range(hue.shape[0]):
        for j in range(hue.shape[1]):
            idx = np.where(val == hue[i][j])
            if prob[idx] > T_H:
                binarized_hue[i][j] = 1
    
    # 3. Color cast attenuation
    binarized_img = binarized_hue.astype('uint8')
    num_labels, _ = cv2.connectedComponents(binarized_img)   # Connected Component Labeling (CCL)
    if num_labels < color_cast_threshold:   # has color cast
        flatten_s = saturation.flatten()
        idx = int(len(flatten_s) / 100)
        least_idx = np.argpartition(flatten_s, idx)
        least_value = flatten_s[least_idx[:idx]].mean()
        
        I_s = saturation - least_value
        I_s[I_s < 0] = 0
        saturation = I_s.astype('uint8')
    
    # 4. HSV to RGB
    hsv = cv2.merge((hue, saturation, value))
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) 
    
    return rgb
        
    
# def LLF_Module():
    # 1. PMF of minimum channel calculation
    
    # 2. 1D minimum filter
    
    # 3. Airlight color estimation
  
if __name__ == "__main__":
    image = cv2.imread("testImage.jpeg")
    AWC_Module(image)
    