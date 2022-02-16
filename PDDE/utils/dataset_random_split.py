import os
import shutil
import numpy as np
from glob import glob

if __name__=='__main__':
    # dataset random split
    dataRoot = 'C:/Users/IIPL/Desktop/data/NYU'
    clear_images_len = 1449
    clear_images = glob(dataRoot + '/clear/*.jpg')
    depths = glob(dataRoot + '/depth/*.npy')
    
    val_len = round(clear_images_len*0.2)
    np.random.seed(1449)
    val_set_clear = np.random.choice(clear_images, val_len, replace=False)
    np.random.seed(1449)
    val_set_depth = np.random.choice(depths, val_len, replace=False)
    
    for i in range(val_len):
        fileName = os.path.basename(val_set_clear[i])
        shutil.move(val_set_clear[i], dataRoot + '/val/clear/' + fileName)
        
        fileName = os.path.basename(val_set_depth[i])
        shutil.move(val_set_depth[i], dataRoot + '/val/depth/' + fileName)