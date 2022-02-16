import os
import numpy as np
from PIL import Image
from models import *

import torch
import torch.nn as nn
import torchvision.transforms as tfs 
import torchvision.utils as vutils
# import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from glob import glob
from tqdm import tqdm

# def tensorShow(tensors,titles=['haze']):
#         fig=plt.figure()
#         for tensor,tit,i in zip(tensors,titles,range(len(tensors))):
#             img = make_grid(tensor)
#             npimg = img.numpy()
#             ax = fig.add_subplot(221+i)
#             ax.imshow(np.transpose(npimg, (1, 2, 0)))
#             ax.set_title(tit)
#         plt.show()

if __name__ == '__main__':
    abs=os.getcwd()+'/'
    gps=3
    blocks=19

    img_dir = 'D:/BaiduNetdiskDownload/UnannotatedHazyImages/'
    output_dir= 'D:/data/pred_FFA_ots/'
    print("pred_dir:",output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    model_dir=abs+f'trained_models/ots_train_ffa_{gps}_{blocks}.pk'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckp = torch.load(model_dir, map_location=device)
    net = FFA(gps=gps, blocks=blocks)
    net = nn.DataParallel(net)
    net.load_state_dict(ckp['model'])
    # net.to(device)
    net.eval()

    transform = tfs.Compose([tfs.ToTensor(),
                            tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])])
    for im in tqdm(glob(img_dir + '/*.jpeg')):
        file_name = os.path.basename(im)[:-4]
        print(f'\r {im}')
        haze = transform(Image.open(im)).unsqueeze(0).to(device)
        if haze.shape[2] > 1000:
            continue
        print(haze.shape)
        
        # haze_no = tfs.ToTensor()(haze)[None,::]
        
        with torch.no_grad():
            pred = net(haze)
        
        pred = pred.clamp(0,1).cpu()
        ts = torch.squeeze(pred)
        # tensorShow([haze_no,pred.clamp(0,1).cpu()],['haze','pred'])
        vutils.save_image(ts, output_dir + file_name + '_FFA.jpeg')
        # exit()
