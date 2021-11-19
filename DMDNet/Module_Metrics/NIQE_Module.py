import cv2
import numpy as np
from skvideo.measure import niqe

class NIQE_Module():
    def __init__(self):
        self.cur_value = None
        self.last_value = None
    
    def reset(self, init_img, color='RGB'):
        self.cur_value = self.get_cur(init_img, color)
        self.last_value = self.cur_value
        
    
    def get_cur(self, img, color='RGB'):
        if np.max(img) <= 1.0:
            img = np.rint(img*255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
            
        if color=='BGR':
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        return niqe(gray_img)[0]
    
    def get_diff(self, cur_img, color='RGB'):
        self.cur_value = self.get_cur(cur_img, color)
        diff_etp = self.last_value - self.cur_value
        self.last_value = self.cur_value
        return diff_etp