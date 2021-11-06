"""
Reference Paper
Image Haze Removal Using Airlight White Correction, Local Light Filter, and Aerial Perspective Prior
Yan-Tsung Peng et al.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def show_plt(x, y):
    plt.bar(x, y, align='center')
    plt.xlabel('deg.')
    plt.xlim([x[0], x[-1]])
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
        
    
def LLF_Module(image):
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
    
    
    r = round(np.array(avg['r']).mean())
    g = round(np.array(avg['g']).mean())
    b = round(np.array(avg['b']).mean())
    
    return r, g, b
                
def airlight_show(r, g, b):
    r = np.full((image.shape[0], image.shape[1]), R).astype('uint8')
    g = np.full((image.shape[0], image.shape[1]), G).astype('uint8')
    b = np.full((image.shape[0], image.shape[1]), B).astype('uint8')
    rgb = cv2.merge((r, g, b))
    
    cv2.imshow(f"Airlight ({R}, {G}, {B})", rgb)
    cv2.waitKey(0)
    
  
if __name__ == "__main__":
    image = cv2.imread("test01.jpg")
    image = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_AREA)
    
    start = time.time()
    image = AWC_Module(image)
    R, G, B = LLF_Module(image)
    print("time :", time.time() - start)
    
    airlight_show(R, G, B)
    
    