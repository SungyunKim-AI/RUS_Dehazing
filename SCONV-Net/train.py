from torch import tensor
from net import InpaintingLoss, FastDVDnet
from glob import glob
from HoleDataset import SequenceDataset, SequenceDataloader
from torchvision import transforms
import torchvision
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v=='True':
        return True
    elif v=='False':
        return False
    else:
        raise argparse.ArgumentTypeError('True or False')

if __name__ == '__main__':
    ## model args ##
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,required=False,default=60)
    parser.add_argument('--batch_size',type=int,required=False,default=8)
    parser.add_argument('--seq_len',type=int,required=False,default=5)
    parser.add_argument('--img_size',type=tuple,required=False,default=[256,256])
    parser.add_argument('--period', type=int,required=False,default=1)
    parser.add_argument('--isPartial',type=str2bool,required=True)
    parser.add_argument('--soft',type=float, required=False,default=0.1)
    parser.add_argument('--bn', type=bool, required=False, default=True)
    parser.add_argument('--activation', type=str,required=False,default='leaky_relu')
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    now = datetime.now()
    writer = SummaryWriter(f'[train]runs-{now.day}_{now.hour}_{now.minute}_{now.second}')
    print(f'[train]runs-{now.day}_{now.hour}_{now.minute}_{now.second}')

    transform = transforms.Compose(
        [transforms.Resize(size=[256,256]), 
        transforms.ToTensor()]
    )
    dataloader = SequenceDataloader('train',args.img_size,args.seq_len,transform,args.period,args.batch_size,True,device)
    batch_num = len(dataloader)
    valid_dataloader = SequenceDataloader('valid',args.img_size,args.seq_len,transform,args.period,args.batch_size,False,device)
    valid_batch_num = len(valid_dataloader)
    
    testsd = SequenceDataset("./data/train/swing",args.img_size,args.seq_len,transform,transform,args.period,device)    
    
    model = FastDVDnet(args.isPartial,args.soft,args.bn,args.activation).to(device)
    writer.add_text('model args',
                    f'batch_size={args.batch_size},\
                    seq_len={args.seq_len},\
                    img_size={args.img_size},\
                    period={args.period},\
                    isPartial={args.isPartial},\
                    soft={args.soft},\
                    bn={args.bn},\
                    activation={args.activation}')
    
    #model.load_state_dict(torch.load('model')['model_state_dict'])
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    min_loss = 1000
    lossfn = InpaintingLoss(device)
    
    
    for epoch in range(args.epochs):
        print(f'epoch : {epoch+1}/{args.epochs}')
        loss_per_epoch=0
        model.train()
        for batch in tqdm(dataloader):
            input_img, input_mask, gt_img = batch
            optimizer.zero_grad()
            output_img, output_mask = model(input_img,input_mask)
            output_img = output_img * output_mask
            
            loss_hole, loss_valid,loss_style = lossfn(output_img,input_mask[:,2,:,:,:],gt_img[:,2,:,:,:])
            loss = 6 * loss_hole + loss_valid +  120*loss_style 
            #loss = 20 * loss_hole + 120 * loss_valid
            loss_per_epoch+=loss.cpu().detach()
            loss.backward()
            optimizer.step()

        loss_per_epoch/=batch_num
        writer.add_scalar('Loss per epoch',loss_per_epoch,epoch)

        
        with torch.no_grad():
            model.eval()
            valid_loss_per_epoch=0
            for batch in tqdm(valid_dataloader):
                input_img, input_mask, gt_img = batch
                output_img, output_mask = model(input_img,input_mask)
                output_img = output_img * output_mask
                
                loss_hole, loss_valid, loss_style = lossfn(output_img, input_mask[:,2,:,:,:],gt_img[:,2,:,:,:])
                loss = 6 * loss_hole + loss_valid +  120*loss_style 
                #loss = 20 * loss_hole + 120 * loss_valid
                valid_loss_per_epoch+=loss
            valid_loss_per_epoch/=valid_batch_num
            writer.add_scalar('valid_loss_per_epoch',valid_loss_per_epoch,epoch)
            
            ## show img ##
            input_img, input_mask, gt_img = testsd[0]
            input_img = input_img.unsqueeze(0)
            input_mask = input_mask.unsqueeze(0)

            output_img,output_mask = model(input_img,input_mask)
            
            _input_img = input_img[0,2,:,:,:].cpu().detach()
            _output_img = output_img[0,:,:,:].cpu().detach()
            _gt_img = gt_img[2,:,:,:].cpu().detach()
            _input_mask = input_mask[0,2,:,:,:].cpu().detach()

            images = input_img[0,:,:,:,:].cpu().detach()
            images = torch.cat([images,torch.stack([_input_img,_output_img,_gt_img*_input_mask + _output_img*(1-_input_mask),_gt_img])],dim=0)
            images = torch.clamp(images,0,1)
            img_grid = torchvision.utils.make_grid(images,5)
            writer.add_image(f'image-epoch:{epoch}',img_grid)
                
            if valid_loss_per_epoch < min_loss:
                min_loss = valid_loss_per_epoch
                writer.add_text('model result',
                                f'train_loss={loss_per_epoch},\
                                valid_loss={valid_loss_per_epoch}',
                                epoch) 
                
                torch.save({'model_state_dict':model.state_dict(),
                            'batch_size':args.batch_size,
                            'seq_len':args.seq_len,
                            'img_size':args.img_size,
                            'period':args.period,
                            'isPartial':args.isPartial,
                            'soft':args.soft,
                            'bn':args.bn,
                            'activation':args.activation,
                            'epoch':epoch,
                            'train_loss':loss_per_epoch,
                            'valid_loss':valid_loss_per_epoch,
                            },f'[train]runs-{now.day}_{now.hour}_{now.minute}_{now.second}/model')