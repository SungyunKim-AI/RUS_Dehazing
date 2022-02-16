from glob import glob
import os
from PIL import Image
import numpy as np

def file_load(path, target_img_size=(256, 256)):
    return np.array(Image.open(path).resize(target_img_size))


if __name__ == '__main__':
    dataRoot = 'D:\\data\\RESIDE_V0_outdoor\\val'
    outputRoot = 'D:\\data\\output_dehaze'
    save_path = 'D:\\data\\output_dehaze\\_result_set'
    target_img_size=(256, 256)
    
    for folder in glob(dataRoot + '\\hazy\\*'):
        for file_path in glob(folder + '\\*.jpg'):
            file_name = os.path.basename(file_path)
            orgin_name = file_name.split('_')[0]
            
            hazy = file_load(file_path)
            dcp = file_load(f'{outputRoot}\\SOTS_DCP\\{file_name}')
            aod = file_load(f'{outputRoot}\\SOTS_AOD\\{file_name}')
            ffa = file_load(f'{outputRoot}\\SOTS_FFA\\{file_name}')
            msbdn = file_load(f'{outputRoot}\\SOTS_MSBDN\\{file_name}')
            ours = file_load(f'{outputRoot}\\SOTS_Ours_pretrained_KITTI_50\\{file_name}')
            clear = file_load(f'{dataRoot}\\clear\\{orgin_name}.png')
            
            division_line = np.full((target_img_size[0], 10, 3), 255, dtype=np.uint8)
            
            total_img = np.hstack([hazy, division_line, dcp, division_line, aod, division_line, 
                                   ffa, division_line, msbdn, division_line, ours, division_line, clear])
            total_img = Image.fromarray(total_img)
            total_img.save(f'{save_path}\\{file_name}')
            # pil_img.show()
            # exit()
    
    