import cv2
import numpy as np

class Entropy_Module():
    def __init__(self):
        self.eps = np.finfo(float).eps
        
    def get_entropy(self, img, channel_first=False):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        val, cnt = np.unique(gray_img, return_counts=True)
        
        if channel_first:
            self.img_size = img.shape[1] * img.shape[2]
        else:
            self.img_size = img.shape[0] * img.shape[1]
        
        prob = cnt / self.img_size    # PMF
        
        H_s = (-1) * np.sum(prob * np.log2(prob + self.eps))
        
        return H_s


if __name__=='__main__':
    dataRoot = 'C:/Users/IIPL/Desktop/data/O_Haze/train/hazy/'
    hazy_img = dataRoot + '01_outdoor_hazy.jpg'
    
    img = cv2.imread(hazy_img)
    H_s = Entropy_Module().get_entropy(img)
    print(H_s)
    
            
        