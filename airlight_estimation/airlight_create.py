import cv2, os
from glob import glob
from tqdm import tqdm

from Airlight_Module import Airlight_Module

if __name__=='__main__':
    if not cv2.useOptimized():
        cv2.setUseOptimized(True)
    
    dataRoot = 'D:\\RUS_ksy\\data\\NYU\\train'
    file_list = glob(os.path.join(dataRoot, 'hazy') + '/*')

    module = Airlight_Module()
    for file in tqdm(file_list):
        imgname = os.path.basename(file)
        hazy = cv2.imread(file)
        
        hazy = module.AWC(hazy, color='RGB')
        airlight_hat, _ = module.LLF(hazy)
        cv2.cvtColor(airlight_hat, cv2.COLOR_RGB2BGR)
        
        save_path = os.path.join(dataRoot, 'airlight/')
        cv2.imwrite(save_path + imgname, airlight_hat)