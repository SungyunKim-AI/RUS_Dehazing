import cv2
import numpy as np

        # img should be 0~1
class Entropy_Module():
    def __init__(self):
        self.eps = np.finfo(float).eps
        self.cur_value = None
        self.last_value = None
        self.best_value = 0.0
        
    def reset(self, init_img):
        self.cur_value = self.get_cur(init_img)
        self.last_value = self.cur_value
        self.best_value = 0.0
        

    def get_cur(self, img):

        img = (img*255).astype(np.uint8)
        self.img_size = img.shape[0] * img.shape[1]
        entropys = np.array([0.0, 0.0, 0.0])
        
        for i in range(3):
            channel = img[:,:,i]
            val, cnt = np.unique(channel, return_counts=True)            
            prob = cnt / self.img_size    # PMF
            H_s = (-1) * np.sum(prob * np.log2(prob + self.eps))
            entropys[i] = H_s
        return np.mean(entropys), np.max(entropys), np.min(entropys)
    
    def get_diff(self, cur_img):
        self.cur_value = self.get_cur(cur_img)
        diff_etp = self.cur_value - self.last_value
        self.last_value = self.cur_value
        return diff_etp
    

if __name__=='__main__':
    dataRoot = 'C:/Users/IIPL/Desktop/data/O_Haze/train/hazy/'
    hazy_img = dataRoot + '01_outdoor_hazy.jpg'
    
    img = cv2.imread(hazy_img)
    H_s = Entropy_Module().get_entropy(img)
    print(H_s)
    
            
        