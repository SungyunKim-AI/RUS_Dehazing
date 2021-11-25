import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from NYU_Dataset import NYU_Dataset
from torch.utils.data import DataLoader

def clear2hazy(image, airlight, depth, beta):
    trans = np.exp(-beta * depth)
    
    if not isinstance(airlight, list):
        airlight_image = np.full((trans.shape), airlight)
    else:
        airlight_image = np.empty([3, trans.shape[0], trans.shape[1]])
        for i, a in enumerate(airlight):
            airlight_image[i] = np.full((trans.shape), a)
    
    hazy = (image * trans) + (airlight_image * (1 - trans))
    hazy = np.rint(hazy*255).astype(np.uint8)
    hazy = np.clip(hazy, 0, 255)
    
    return hazy

def save_hazy(dataRoot, hazy, airlight, beta, fileName):
    save_path = f"{dataRoot}/hazy/NYU_{airlight}_{beta}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path += f'/{fileName}_{airlight}_{beta}.jpg'
    
    if len(hazy) == 3:
        hazy = hazy.transpose(1,2,0)
    
    # Image.fromarray(hazy).show()
    Image.fromarray(hazy).save(save_path)
    

if __name__ == '__main__':
    dataRoot = 'C:/Users/IIPL/Desktop/data/NYU'
    dataset = NYU_Dataset(dataRoot)
    dataloader = DataLoader(dataset=dataset, batch_size=1,
                            num_workers=2, drop_last=False, shuffle=False)
    
    for data in tqdm(dataloader):
        image = data[0].squeeze().numpy()
        GT_depth = data[1].squeeze().numpy()
        fileName = data[2][0]
        
        # airlight_list = [0.8, 0.85, 0.9, 0.95, 1.0]
        # beta_list = [0.04, 0.06, 0.08, 0.1, 0.12, 0.16, 0.2]
        airlight_list = [0.85, 0.9, 0.95, 1.0]
        beta_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for airlight in airlight_list:
            for beta in beta_list:
                hazy = clear2hazy(image, airlight, GT_depth, beta)
                save_hazy(dataRoot, hazy, airlight, beta, fileName)
                