from math import nan
import cv2
import numpy as np
import torch
from .util import denormalize


def get_Airlight(images, norm=True, color_cast_threshold=5):
    # image input must be Bx3xWxH
    if norm:
        images = torch.round(((images * 0.5) + 0.5) * 255.0).type(torch.uint8)
    else:
        images = torch.round(images * 255).type(torch.uint8)
    images = torch.clamp(images, 0, 255)
    
    airlight = []
    for image in images:
        img_size = image.shape[1] * image.shape[2]
        
        # 1. RGB to HSV
        hue, saturation, value = rgb2hsv(image)
        
        # 2. Hue histogram labeling
        val, cnt = torch.unique(hue, return_counts=True)
        prob = cnt / img_size    # PMF
        T_H = 1 / 360
        
        binarized_hue = np.zeros((hue.shape), np.uint8)
        for v, p in zip(val, prob):
            if p > T_H:
                row, col = torch.where(hue == v)
                for r, c in zip(row, col):
                    binarized_hue[r][c] = 1
        binarized_hue = binarized_hue.astype('uint8')
        
        # 3. Color cast attenuation
        num_labels, _ = cv2.connectedComponents(binarized_hue)   # Connected Component Labeling (CCL)
        if num_labels < color_cast_threshold:   # has color cast
            bottom_1p = round(img_size * 0.01)

            avg_least_values = torch.topk(hue.flatten(), bottom_1p, largest=False).values.float().mean().round().item()
            hue = hue - avg_least_values
            hue[hue < 0] = 0
            
            avg_least_values = torch.topk(saturation.flatten(), bottom_1p, largest=False).values.float().mean().round().item()
            saturation = saturation - avg_least_values
            saturation[saturation < 0] = 0
            
            avg_least_values = torch.topk(value.flatten(), bottom_1p, largest=False).values.float().mean().round().item()
            value = value - avg_least_values
            value[value < 0] = 0
        
        # 4. HSV to RGB
        image = hsv2rgb(hue, saturation, value).round().int().clamp(0, 255)
        min_channel = torch.amin(image, 0)
        
        val, cnt = torch.unique(min_channel, return_counts=True)
        prob = cnt / img_size
        
        l = np.zeros((256))
        for i, v in enumerate(val):
            l[v] = prob[i]
        l = torch.Tensor(l)
        
        # 2. 1D minimum filter
        P_m = []
        for i in range(256):
            if i <= 5:
                P_m.append(torch.min(l[:i+6]))
            elif 5 < i <= 249:
                P_m.append(torch.min(l[i-5:i+6]))
            else:
                P_m.append(torch.min(l[i-5:]))
        
        # 3. Airlight color estimation to RGB to RGB
        threshold, idx = 0.0, []
        for i in range(255, -1, -1):
            if P_m[i] > 0.0:
                threshold += P_m[i]
                idx.append(i)
                if threshold >= 0.01:
                    break
        
        avg = {'r': [], 'g': [], 'b': []}
        for i in idx:
            row, col = torch.where(min_channel == i)
            for r, c in zip(row, col):
                avg['r'].append(image[0][r][c])
                avg['g'].append(image[1][r][c])
                avg['b'].append(image[2][r][c])
        
        RGB = []
        for i, c in enumerate(['r', 'g', 'b']):
            mean = torch.Tensor(avg[c]).mean().round().div(255.0)
            RGB.append(np.full((image.shape[1], image.shape[2]), mean))
        
        airlight.append(RGB)
    
    airlight = torch.Tensor(airlight)
    if norm:
        return airlight.sub_(0.5).div_(0.5)
    else:
        return airlight
    

def get_Airlight2(images, norm=True, color_cast_threshold=5):
    # image input must be Bx3xWxH
    images = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
    if norm:
        images = (np.rint(((images * 0.5) + 0.5) * 255)).astype(np.uint8)
    else:
        images = (np.rint(images * 255)).astype(np.uint8)
    images = np.clip(images, 0, 255)
    
    airlight = []
    for image in images:
        img_size = image.shape[0] * image.shape[1]
        
        # 1. RGB to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        
        # 2. Hue histogram labeling
        val, cnt = np.unique(hue, return_counts=True)
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
        if num_labels < color_cast_threshold:   # has color cast
            for k in [hue, saturation, value]:
                k = k.flatten()
                idx = int(len(k) / 100)
                least_idx = np.argpartition(k, idx)     # bottom 1% least saturated pixels
                least_value = k[least_idx[:idx]].mean()
                
                k = k - least_value
                k[k < 0] = 0
                k = k.astype('uint8')
        
        # 4. HSV to RGB
        hsv = cv2.merge((hue, saturation, value))
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        
        # 1. PMF of minimum channel calculation
        c1, c2, c3 = cv2.split(image)
        min_channel = np.amin([c1, c2, c3], 0)
        
        val, cnt = np.unique(min_channel, return_counts=True)
        img_size = image.shape[0] * image.shape[1]
        prob = cnt / img_size    # PMF
        
        l = np.zeros((256))
        for i, v in enumerate(val):
            l[v] = prob[i]
        
        # 2. 1D minimum filter
        P_m = []
        l = np.insert(l, 0, np.array([1,1,1,1,1]))
        l = np.append(l, np.array([1,1,1,1,1]))
        for i in range(0+5, 255+6):
            P_m.append(l[i-5:i+6].min())
        P_m = np.array(P_m)
        
        # 3. Airlight color estimation to RGB to RGB
        threshold = 0.0
        idx = []
        for i in range(255, -1, -1):
            if P_m[i] > 0.0:
                threshold += P_m[i]
                idx.append(i)
                if threshold >= 0.01:
                    break
        
        avg = {'r': [], 'g': [], 'b': []}
        RGB = {'r': c1, 'g': c2, 'b': c3}
        for i in idx:
            row, col = np.where(min_channel == i)
            for j, _ in enumerate(row):
                for c in ['r','g','b']:
                    avg[c].append(RGB[c][row[j]][col[j]]) 
        
        mean_r = np.mean(np.array(avg['r']))
        mean_g = np.mean(np.array(avg['g']))
        mean_b = np.mean(np.array(avg['b']))

        if np.isnan(mean_r):
            mean_r = 0
        if np.isnan(mean_g):
            mean_g = 0
        if np.isnan(mean_b):
            mean_b = 0

        avgR = round(mean_r)/ 255.0 
        avgG = round(mean_g) / 255.0
        avgB = round(mean_b) / 255.0
        
        r = np.full((image.shape[0], image.shape[1]), avgR)
        g = np.full((image.shape[0], image.shape[1]), avgG)
        b = np.full((image.shape[0], image.shape[1]), avgB)
         
        airlight.append(cv2.merge((r, g, b)))
    
    airlight = np.array(airlight)
    if norm:
        airlight = ((airlight / 255.0) - 0.5) / 0.5
        airlight = airlight.transpose(0, 3, 1, 2)
    else:
        airlight = (airlight / 255.0).transpose(0, 3, 1, 2)
    
    return torch.Tensor(airlight).float()
        

        

def rgb2hsv(input, epsilon=1e-10):
    input.unsqueeze_(0)

    r, g, b = input[:, 0], input[:, 1], input[:, 2]
    max_rgb, argmax_rgb = input.max(1)
    min_rgb, argmin_rgb = input.min(1)

    max_min = max_rgb - min_rgb + epsilon

    h1 = 60.0 * (g - r) / max_min + 60.0
    h2 = 60.0 * (b - g) / max_min + 180.0
    h3 = 60.0 * (r - b) / max_min + 300.0

    h = torch.stack((h2, h3, h1), dim=0).gather(dim=0, index=argmin_rgb.unsqueeze(0)).squeeze()
    s = (max_min / (max_rgb + epsilon)).squeeze()
    v = (max_rgb).squeeze()
    
    return h, s, v


def hsv2rgb(h, s, v):
    h.unsqueeze_(0)
    s.unsqueeze_(0)
    v.unsqueeze_(0)
    
    h_ = (h - torch.floor(h / 360) * 360) / 60
    c = s * v
    x = c * (1 - torch.abs(torch.fmod(h_, 2) - 1))

    zero = torch.zeros_like(c)
    y = torch.stack((
        torch.stack((c, x, zero), dim=1),
        torch.stack((x, c, zero), dim=1),
        torch.stack((zero, c, x), dim=1),
        torch.stack((zero, x, c), dim=1),
        torch.stack((x, zero, c), dim=1),
        torch.stack((c, zero, x), dim=1),
    ), dim=0)

    index = torch.repeat_interleave(torch.floor(h_).unsqueeze(1), 3, dim=1).unsqueeze(0).to(torch.long)
    rgb = (y.gather(dim=0, index=index) + (v - c)).squeeze()
    
    return rgb


class Airlight_Module():
    def __init__(self, color_cast_threshold=5):
        self.color_cast_threshold = color_cast_threshold

    # Airlight White Correction (AWC)
    def AWC(self, image, is_Image=False, color='BGR'):
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
    def LLF(self, image, mReturn='RGB'):
        # 1. PMF of minimum channel calculation  
        if np.max(image) <= 1.0:
            image = np.rint(image*255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
            
        c1, c2, c3 = cv2.split(image)
        min_channel = np.amin([c1, c2, c3], 0)
            
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
        threshold = 0.0
        idx = []
        for i in range(255, -1, -1):
            if P_m[i] > 0.0:
                threshold += P_m[i]
                idx.append(i)
                if threshold >= 0.01:
                    break
        
        avg = {'r': [], 'g': [], 'b': []}
        RGB = {'r': c1, 'g': c2, 'b': c3}
        for i in idx:
            row, col = np.where(min_channel == i)
            for j, _ in enumerate(row):
                for c in ['r','g','b']:
                    avg[c].append(RGB[c][row[j]][col[j]]) 
        
        mean_r = np.mean(np.array(avg['r']))
        mean_g = np.mean(np.array(avg['g']))
        mean_b = np.mean(np.array(avg['b']))

        if np.isnan(mean_r):
            mean_r = 0
        if np.isnan(mean_g):
            mean_g = 0
        if np.isnan(mean_b):
            mean_b = 0

        avgR = round(mean_r)/ 255.0 
        avgG = round(mean_g) / 255.0
        avgB = round(mean_b) / 255.0
        
        r = np.full((image.shape[0], image.shape[1]), avgR)
        g = np.full((image.shape[0], image.shape[1]), avgG)
        b = np.full((image.shape[0], image.shape[1]), avgB)
        
        if mReturn == 'RGB':    
            airlight = cv2.merge((r, g, b))
            return airlight, [avgR, avgG, avgB]
        
        elif mReturn == 'BGR':
            airlight = cv2.merge((b, g, r))
            return airlight, [avgB, avgG, avgR]
            
        elif mReturn == 'gray':
            # 3. Airlight color estimation to RGB to Gray-Scale
            rgb = cv2.merge((r, g, b))
            airlight = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            mean_val = airlight[0][0]
            return airlight, mean_val
        
        else:
            raise ValueError('mReturn must be RGB, BGR or gray')

    def get_airlight(self, image, norm):
        image_numpy = (denormalize(image, norm)[0].detach().cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)
        awc_rgb = self.AWC(image_numpy, True, 'BGR')
        _, airlight = self.LLF(awc_rgb)
        return airlight[0]