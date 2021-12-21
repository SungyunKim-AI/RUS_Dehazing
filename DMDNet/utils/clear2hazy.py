import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from dataset.NYU_Dataset import NYU_Dataset_clear
from torch.utils.data import DataLoader
import operator

def clear2hazy(clear, airlight, depth, beta):
    trans = np.exp(-beta * depth)

    hazy = (clear * trans) + ((1 - trans) * airlight)
    hazy = np.rint(hazy*255).astype(np.uint8)
    hazy = np.clip(hazy, 0, 255)
    
    return hazy

def save_crop_file(path, file_name, clear, depth):
    save_path_clear = f"{path}/crop_clear"
    if not os.path.exists(save_path_clear):
        os.makedirs(save_path_clear)
    save_path_clear += f'/{file_name}.jpg'
    
    clear = np.rint(clear*255).astype(np.uint8)
    if len(clear) == 3:
        clear = clear.transpose(1,2,0)
    
    Image.fromarray(clear).save(save_path_clear)
    
    save_path_depth = f"{path}/crop_depth"
    if not os.path.exists(save_path_depth):
        os.makedirs(save_path_depth)
    save_path_depth += f'/{file_name}'
    
    np.save(save_path_depth, depth)

def save_hazy(path, hazy, airlight, beta, fileName, show=False):
    save_path = f"{path}/hazy/NYU_{airlight}_{beta}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path += f'/{fileName}_{airlight}_{beta}.jpg'
    
    if len(hazy) == 3:
        hazy = hazy.transpose(1,2,0)
    
    if show:
        Image.fromarray(hazy).show()
    Image.fromarray(hazy).save(save_path)
    
def cropND(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]
    

if __name__ == '__main__':
    dataRoot = 'C:/Users/IIPL/Desktop/data/NYU'
    
    for split in ['/train', '/val']:
        path = dataRoot + split
        dataset = NYU_Dataset_clear(path)
        dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=2, drop_last=False, shuffle=False)
        
        for data in tqdm(dataloader):
            clear = data[0].squeeze().numpy()
            clear = cropND(clear, (3, 460, 620))
            GT_depth = data[1].squeeze().numpy()
            GT_depth = cropND(GT_depth, (460, 620))
            
            fileName = data[2][0]
            
            save_crop_file(path, os.path.basename(fileName), clear, GT_depth)
           
            airlight_list = [0.8, 0.9, 1.0]
            beta_list = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7]
            for airlight in airlight_list:
                for beta in beta_list:
                    hazy = clear2hazy(clear, airlight, GT_depth, beta)
                    save_hazy(path, hazy, airlight, beta, fileName)
                