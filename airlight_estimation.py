"""
Reference Paper
Image Haze Removal Using Airlight White Correction, Local Light Filter, and Aerial Perspective Prior
Yan-Tsung Peng et al.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Airlight White Correction (AWC)
def AWC_Module(image, color_cast_threshold=5):
    # 1. RGB to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, _, _ = cv2.split(hsv)
    
    # 2. Hue histogram labeling
    val, cnt = np.unique(hue, return_counts=True)
    img_size = hue.shape[0]*hue.shape[1]
    
    P_H = cnt / img_size    # PMF        
    T_H = img_size / 360    # ?
    
    binarized_hue = np.zeros((hue.shape[0], hue.shape[1]), int)
    for i, h in enumerate(val):
        row, col = np.where(hue == h)
        
        for j in row:
            for k in col:
                if P_H[i] > T_H:
                    binarized_hue[j][k] = 1
    
    # 3. Color cast attenuation
    num_labels, labels = cv2.connectedComponents(binarized_hue)   # Connected Component Labeling (CCL)
    
    # 4. HSV to RGB
    
def LLF_Module():
    # 1. PMF of minimum channel calculation
    
    # 2. 1D minimum filter
    
    # 3. Airlight color estimation
  
if __name__ == "__main__":
    image = cv2.imread("testImage.jpeg")
    # image = cv2.imread("testImage.jpeg", cv2.IMREAD_GRAYSCALE)
    hue = getHue(image)
    PMF_H(hue)
    