import cv2
import numpy as np

class Entropy_Module():
    def __init__(self, init_img):
        self.eps = np.finfo(float).eps
        self.cur_value = self.get_cur(init_img)
        self.last_value = self.cur_value
        
    def get_cur(self, img, channel_first=False):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        val, cnt = np.unique(gray_img, return_counts=True)
        
        if channel_first:
            self.img_size = img.shape[1] * img.shape[2]
        else:
            self.img_size = img.shape[0] * img.shape[1]
        
        prob = cnt / self.img_size    # PMF
        
        H_s = (-1) * np.sum(prob * np.log2(prob + self.eps))
        
        return H_s
    
    def get_diff(self, cur_img):
        self.cur_value = self.get_cur(cur_img)
        diff_etp = self.last_value - self.cur_value
        self.last_value = self.cur_value
        return diff_etp
    

if __name__=='__main__':
    dataRoot = 'C:/Users/IIPL/Desktop/data/O_Haze/train/hazy/'
    hazy_img = dataRoot + '01_outdoor_hazy.jpg'
    
    img = cv2.imread(hazy_img)
    H_s = Entropy_Module().get_entropy(img)
    print(H_s)
    
            
        