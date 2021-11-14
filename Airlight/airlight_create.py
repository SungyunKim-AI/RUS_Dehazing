import cv2, os
from glob import glob
from tqdm import tqdm

from Airlight_Module import Airlight_Module

if __name__=='__main__':
    if not cv2.useOptimized():
        cv2.setUseOptimized(True)
    
    dataRoot = 'C:/Users/IIPL/Desktop/data/O_Haze/train'
    file_list = glob(os.path.join(dataRoot, 'hazy') + '/*')
 
    module = Airlight_Module()
    for file in tqdm(file_list):
        imgname = os.path.basename(file)
        
        hazy = cv2.imread(file)
        # hazy = cv2.resize(hazy, dsize=(0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        hazy = module.AWC(hazy, color='BGR')
        
        # hazy = module.AWC(file)
        # airlight_hat, air = module.LLF(hazy, mReturn='RGB')
        
        if hazy.shape[2] == 3:
            airlight_hat = cv2.cvtColor(hazy, cv2.COLOR_RGB2BGR)
            save_path = os.path.join(dataRoot, 'AWC/')
        else:
            save_path = os.path.join(dataRoot, 'airlight_gray/')
        
        cv2.imwrite(save_path + imgname, airlight_hat)