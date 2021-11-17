import cv2, os
from glob import glob
from tqdm import tqdm
import time

from Airlight_Module import Airlight_Module

def Airlight_Module_test():
    if not cv2.useOptimized():
        cv2.setUseOptimized(True)
    
    start = time.time()
    image = cv2.imread("test.jpg")
    # image = cv2.resize(image, dsize=(0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    airlight_module = Airlight_Module()
    image = airlight_module.AWC(image)
    airlight, single_val = airlight_module.LLF(image, mReturn='gray')    # mReturn = 'RGB' or 'gray
    print("Single Value : ", single_val)
    print("Operation Time(s) : ", round(time.time() - start, 3))
    
    if airlight.shape[0] == 3:
        airlight = cv2.cvtColor(airlight, cv2.COLOR_RGB2BGR) 
    
    cv2.imshow(f"Airlight", airlight)
    cv2.waitKey(0)

def save_airlight_image():
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

if __name__=='__main__':
    Airlight_Module_test()
    # save_airlight_image()