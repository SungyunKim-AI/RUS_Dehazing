import cv2
from skvideo.measure import niqe

class NIQE_Module():
    def __init__(self, init_img):
        self.cur_value = self.get_cur(init_img)
        self.last_value = self.cur_value
        
    def get_cur(self, img, color='BGR'):
        if color=='BGR':
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        return niqe(gray_img)
    
    def get_diff(self, cur_img):
        self.cur_value = self.get_cur(cur_img)
        diff_etp = self.last_value - self.cur_value
        self.last_value = self.cur_value
        return diff_etp