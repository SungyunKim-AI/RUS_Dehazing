import torch
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import ConcatDataset, Dataset
from torchvision import transforms
from net import FastDVDnet
from tqdm import tqdm
    
    
class EmptyDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
    def __len__(self):
        return 0;
    def __getitem__(self,idx):
        return None;
    
class HoleDataset(torch.utils.data.Dataset):
    def __init__(self, path, img_transfrom, mask_transform, mask_path='./data/mask'):
        super().__init__()
        self.img_path_list = glob(f'{path}/*.jpg')
        self.mask_path_list = glob(f'{mask_path}/*.png')
        self.img_transfrom = img_transfrom
        self.mask_transform = mask_transform
        self.custom_mask = (mask_path != './data/mask')
    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        gt_img = Image.open(self.img_path_list[idx])
        gt_img = self.img_transfrom(gt_img.convert('RGB'))
        
        mask=None
        if self.custom_mask :
            mask = Image.open(self.mask_path_list[idx])
            mask = self.mask_transform(mask.convert('L'))
            mask = (mask==0).to(torch.float).squeeze(0)
            mask = torch.stack([mask,mask,mask])
            
        else:
            mask = Image.open(self.mask_path_list[torch.randint(0,12000,[1,1])])
            mask = self.mask_transform(mask.convert('RGB'))
            mask = 1-mask
        mask = mask//1
        
        return gt_img*mask, mask, gt_img

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, path, img_size, seq_len, img_transfrom, mask_transform,period,device, mask_path='./data/mask'):
        super().__init__()
        self.period=period
        self.data = HoleDataset(path,img_transfrom,mask_transform, mask_path)
        self.seq_len = seq_len
        self.img_height = img_size[0]
        self.img_weight = img_size[1]
        self.device = device
    def __len__(self):
        #return 1
        return self.data.__len__()-(self.seq_len+(self.seq_len-1)*(self.period-1))+1
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        masked_img=torch.Tensor(self.seq_len, 3, self.img_height, self.img_weight).to(self.device)
        mask=torch.zeros_like(masked_img)
        gt_img=torch.zeros_like(masked_img)
        for i in range(self.seq_len):
            masked_img_, mask_, gt_img_ = self.data[idx+i*self.period]
            masked_img[i] = masked_img_
            mask[i] = mask_
            gt_img[i] = gt_img_
        return masked_img, mask, gt_img
    
    
def SequenceDataloader(split,img_size,seq_len,transform,period,batch_size,shuffle,device):
    folder_list = glob(f'./data/{split}/*')
    sds=EmptyDataset()
    for folder in folder_list:
        sd=None
        if(split=='test'):
            sd = SequenceDataset(f'{folder}/img',img_size,seq_len,transform,transform,period,device,f'{folder}/mask')
        else:
            sd = SequenceDataset(folder,img_size,seq_len,transform,transform,period,device)
        sds = ConcatDataset([sds,sd])
    
    return DataLoader(sds,batch_size,shuffle,drop_last=True)


if __name__ == '__main__':
    seq_len = 5
    batch_size = 1
    img_size=[256,256]
    epochs = 10000
    period=3
    soft=0.3
    
    transform = transforms.Compose(
        [transforms.Resize(size=[256,256]), 
        transforms.ToTensor()]
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    data_loaders,_ = SequenceDataloader('test',img_size,seq_len,transform,period,batch_size,device)
    
    model = FastDVDnet(soft).to(device)
    for epoch in range(epochs):
        for data_loader in tqdm(data_loaders):
            for batch in data_loader:
                input_img, input_mask, gt_img = batch
                output_img,output_mask = model(input_img, input_mask)