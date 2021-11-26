import csv
import numpy as np
from glob import glob

import torch
from torch.utils.data import DataLoader
from torchvision import utils

from dataset import NYU_Dataset
from dpt.models import DPTDepthModel


        
def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3]

if __name__=='__main__':
    # Init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_image = True
    
    # Init Dataset - NYU_depth_v2_labeled
    dataRoot = 'C:/Users/IIPL/Desktop/data/NYU'
    dataset = NYU_Dataset(dataRoot)
    dataloader = DataLoader(dataset=dataset, batch_size=1,
                             num_workers=0, drop_last=False, shuffle=False)

    # # Init Depth Estimation Model - DPT
    model = DPTDepthModel(path = 'weights/dpt_hybrid-midas-501f0c75.pt',
                          scale=0.00030, shift=0.1378, invert=True,
                          backbone='vitb_rn50_384',
                          non_negative=True,
                          enable_attention_hooks=False).to(memory_format=torch.channels_last)
    model.to(device).eval()
    
    err_list = []
    for data in dataloader:
        image, GT_depth, fileName = data[0].to(device), data[1], data[2][0]
        
        with torch.no_grad():
            _, pred_depth = model.forward(image)            
        pred_depth = pred_depth.detach().cpu()
        
        err = compute_errors(GT_depth.numpy(), pred_depth.numpy())
        err_list.append(err) 
        
        if save_image:
            save_path = dataRoot + '/results/' + fileName + '.jpg'
            utils.save_image(utils.make_grid([GT_depth/10, pred_depth/10]), save_path)
    
    with open(f'{dataRoot}/err_log.csv','w',newline='') as csv_file:
        csv_wr = csv.writer(csv_file)
        csv_wr.writerow(['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3'])
        csv_wr.writerows(err_list)
