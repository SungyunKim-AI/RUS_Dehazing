import wandb
from data \
    import HazeDataset.RESIDE_Beta_Dataset, HazeDataset.O_Haze_Dataset, HazeDataset.NH_Haze_Dataset
import torch
from metrics import psnr,ssim
from net import ConvUNet, GConvUNet
import math
from torch.backends import cudnn
from torch import optim
from torch.utils.data import DataLoader
import torch,warnings
from torch import nn
warnings.filterwarnings('ignore')
from torchvision.models import vgg16
from tqdm import tqdm
from loss import LossNetwork as PerLoss
	
def lr_schedule_cosdecay(t,T,init_lr):
	lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
	return lr
         
def train(model,train_loader,optim,criterion, epochs, init_lr, device):
    batch_num = len(train_loader)
    steps = batch_num * epochs
    
    losses = []
    i = 0
    mse_epoch = 0.0
    ssim_epoch = 0.0
    psnr_epoch = 0.0
    unet_epoch = 0.0
    
    step = 1 
    model.train()
    for epoch in range(epochs):
        i=0
        mse_epoch = 0.0
        ssim_epoch = 0.0
        psnr_epoch = 0.0
        for batch in tqdm(train_loader):
            optim.zero_grad()
            lr = lr_schedule_cosdecay(step, steps, init_lr)
            for param_group in optim.param_groups:
                param_group["lr"] = lr
            hazy_images, clear_images = batch
            hazy_images = hazy_images.to(device)
            clear_images = clear_images.to(device)
            
            outputs = model(hazy_images) # for conv
            
            loss = criterion[0](outputs, clear_images)
            loss2 = criterion[1](outputs, clear_images)
            loss = loss + 0.04*loss2
            loss.backward()
            
            optim.step()
            losses.append(loss.item())
            
            MSELoss = nn.MSELoss()
            mse_ = MSELoss(outputs, clear_images)
            ssim_ = ssim(outputs, clear_images)
            psnr_ = psnr(outputs, clear_images)
            
            mse_epoch += mse_.cpu().item()
            ssim_epoch += ssim_.cpu().item()
            psnr_epoch += psnr_
            i += 1
            
            step+=1
            
        mse_epoch /= i
        ssim_epoch /= i
        psnr_epoch /= i
        print("mse: + "+str(mse_epoch) + " | ssim: "+ str(ssim_epoch) + " | psnr:"+str(psnr_epoch))
        print()
        
        wandb.log({"MSE" : mse_epoch,
                   "SSIM": ssim_epoch,
                   "PSNR" : psnr_epoch,
                   "global_step" : epoch+1})
        
        path_of_weight = f'weights/weight_{epoch+1:03}.pth'  #path for storing the weights of genertaor
        torch.save(model.state_dict(), path_of_weight)
        
if __name__ == "__main__":
    # ============== Config Init ==============
    lr = 0.0001
    epochs = 15
    soft = 0.1
    img_size = (256,256)
    batch_size = 16
    dataset_train=RESIDE_Beta_Dataset('D:/data/RESIDE-beta/train',[0],img_size)
    #dataset_train = O_Haze_Dataset('D:/data/O-Haze/train',img_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config_defaults = {
        'model_name' : 'GConvUNet_New',
        'init_lr' : lr,
        'epochs' : epochs,
        'soft' : soft,
        'dataset' : 'RESIDE-beta(0.85_0.04)',
        'batch_size': batch_size,
        'image_size': img_size}
    
    wandb.init(config=config_defaults, project='Dehazing', entity='rus')
    wandb.run.name = config_defaults['model_name']
    config = wandb.config
    
    # ============== Data Load =============
    loader_train = DataLoader(
                dataset=dataset_train,
                batch_size=batch_size,
                num_workers=0,
                drop_last=True,
                shuffle=True)
    
    net=GConvUNet(soft=soft)
    net=net.to(device)
    if device=='cuda':
        #net=torch.nn.DataParallel(net)
        cudnn.benchmark=True
    
    criterion = []
    criterion.append(nn.L1Loss().to(device))
    vgg_model = vgg16(pretrained=True).features[:16]
    vgg_model = vgg_model.to(device)
    for param in vgg_model.parameters():
        param.requires_grad = False
    criterion.append(PerLoss(vgg_model).to(device))
        
    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()),lr=lr, betas = (0.9, 0.999), eps=1e-08)
    train(net,loader_train,optimizer,criterion,epochs,lr,device)
    