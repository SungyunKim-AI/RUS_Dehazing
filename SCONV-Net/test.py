from glob import glob
from net import FastDVDnet
import torch
from HoleDataset import SequenceDataset
import torchvision
from tqdm import tqdm
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


if __name__ == '__main__':
    load = torch.load('model')
    
    """
    ## model args ##
    batch_size = load['batch_size']
    seq_len = load['seq_len']
    img_size = load['img_size']
    period = load['period']
    isPartial = load['isPartial']
    soft = load['soft']
    bn = load['bn']
    activation = load['activation']
    
    ## model result ##
    epoch = load['epoch']
    train_loss = load['train_loss']
    valid_loss = load['valid_loss']
    """
    
    batch_size = 8
    seq_len = 5
    img_size = [256,256]
    period = 1
    isPartial = True
    soft = 0.3
    bn = True
    activation ='leaky_relu'
    
    epoch = 53
    train_loss = 0.2879
    valid_loss = 0.3299
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    now = datetime.now()
    writer = SummaryWriter(f'[test]runs-{now.day}_{now.hour}_{now.minute}_{now.second}')
    print(f'[test]runs-{now.day}_{now.hour}_{now.minute}_{now.second}')

    writer.add_text('model args',
                    f'batch_size={batch_size},\
                    seq_len={seq_len},\
                    img_size={img_size},\
                    period={period},\
                    isPartial={isPartial},\
                    soft={soft},\
                    bn={bn},\
                    activation={activation}')
    
    writer.add_text('model result',
                    f'epoch={epoch},\
                    train_loss={train_loss},\
                    valid_loss={valid_loss}') 
    
    transform = transforms.Compose(
        [transforms.Resize(size=[256,256]), 
        transforms.ToTensor()]
    )
    
    sequencedatasets=[]
    folder_list = glob("./data/test/*")
    batch_num=0
    for folder in folder_list:
        sd = SequenceDataset(f'{folder}/img',img_size,seq_len,transform,transform,period,device,f'{folder}/mask')
        sequencedatasets.append(sd)
      
    model = FastDVDnet(isPartial,soft,bn,activation).to(device)
    #model.load_state_dict(load['model_state_dict'])
    model.load_state_dict(load)

    with torch.no_grad():
        model.eval()
        for i,sd in enumerate(tqdm(sequencedatasets)):
            input_img, input_mask, gt_img = sd[0]
            input_img = input_img.unsqueeze(0)
            input_mask = input_mask.unsqueeze(0)
            
            
            output_img, output_mask = model(input_img,input_mask)
            
            _input_img = input_img[0,2,:,:,:].cpu().detach()
            _output_img = output_img[0,:,:,:].cpu().detach()
            _gt_img = gt_img[2,:,:,:].cpu().detach()
            _input_mask = input_mask[0,2,:,:,:].cpu().detach()
            
            images = input_img[0,:,:,:,:].cpu().detach()
            images = torch.cat([images,torch.stack([_input_img,_output_img,_gt_img*_input_mask + _output_img*(1-_input_mask),_gt_img])],dim=0)
            images = torch.clamp(images,0,1)
            img_grid = torchvision.utils.make_grid(images,5)
            writer.add_image(f'image:{i}',img_grid)