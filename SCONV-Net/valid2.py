from glob import glob
from net import FastDVDnet
import torch
from HoleDataset import SequenceDataset
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


if __name__ == '__main__':
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    now = datetime.now()
    writer = SummaryWriter(f'[test]runs-{now.day}_{now.hour}_{now.minute}_{now.second}')
    print(f'[test]runs-{now.day}_{now.hour}_{now.minute}_{now.second}')

    transform = transforms.Compose(
        [transforms.Resize(size=[256,256]), 
        transforms.ToTensor()]
    )
    sequencedatasets=[]
    folder_list = glob("./data/test/*")
    batch_num=0
    for folder in folder_list:
        sd = SequenceDataset(f'{folder}/img',[256,256],5,transform,transform,7,device,f'{folder}/mask')
        sequencedatasets.append(sd)
      
    model0 = FastDVDnet(False,0.3,True,'leaky_relu').to(device)
    model0.load_state_dict(torch.load('0')['model_state_dict'])
    
    model1 = FastDVDnet(False,0.1,True,'leaky_relu').to(device)
    model1.load_state_dict(torch.load('1')['model_state_dict'])

    model2 = FastDVDnet(False,0.01,True,'leaky_relu').to(device)
    model2.load_state_dict(torch.load('2')['model_state_dict'])

    model3 = FastDVDnet(False,1e-8,True,'leaky_relu').to(device)
    model3.load_state_dict(torch.load('3')['model_state_dict'])
    
    
    with torch.no_grad():
        model0.eval()
        model1.eval()
        model2.eval()
        model3.eval()
        for i,sd in enumerate(tqdm(sequencedatasets)):
            input_img, input_mask, gt_img = sd[0]
            input_img = input_img.unsqueeze(0)
            input_mask = input_mask.unsqueeze(0)
            
            
            output_img0, output_mask0 = model0(input_img,input_mask)
            output_img1, output_mask1 = model1(input_img,input_mask)
            output_img2, output_mask2 = model2(input_img,input_mask)
            output_img3, output_mask3 = model3(input_img,input_mask)
            
            _input_img = input_img[0,2,:,:,:].cpu().detach()
            
            _output_img0 = output_img0[0,:,:,:].cpu().detach()
            _output_img1 = output_img1[0,:,:,:].cpu().detach()
            _output_img2 = output_img2[0,:,:,:].cpu().detach()
            _output_img3 = output_img3[0,:,:,:].cpu().detach()
            
            _gt_img = gt_img[2,:,:,:].cpu().detach()
            _input_mask = input_mask[0,2,:,:,:].cpu().detach()
            
            images = input_img[0,:,:,:,:].cpu().detach()
            images = torch.cat([images,torch.stack([_gt_img*_input_mask + _output_img0*(1-_input_mask),
                                                    _gt_img*_input_mask + _output_img1*(1-_input_mask),
                                                    _gt_img*_input_mask + _output_img2*(1-_input_mask),
                                                    _gt_img*_input_mask + _output_img3*(1-_input_mask),
                                                    _gt_img])],dim=0)
            images = torch.clamp(images,0,1)
            img_grid = torchvision.utils.make_grid(images,5)
            writer.add_image(f'image:{i}',img_grid)