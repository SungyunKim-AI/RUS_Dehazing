# User warnings ignore
import warnings
warnings.filterwarnings("ignore")

# Python, Utils packages
import argparse, os, random, cv2
from tqdm import tqdm

# Pytorch
import torch
from torch import nn
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import torch.nn.parallel

# Self-made Models, Utils
from misc import *
import models.dehaze22  as net
from myutils.metrics import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=False, default='pix2pix',  help='')
    parser.add_argument('--dataroot', required=False, default='', help='path to trn dataset')
    parser.add_argument('--outputRoot', type=str, default='output/', help='Model test output folder path')
    parser.add_argument('--mode', type=str, default='B2A', help='B2A: facade, A2B: edges2shoes')
    parser.add_argument('--manualSeed', type=int, default=101, help='B2A: facade, A2B: edges2shoes')
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--originalSize', type=int, default=286, help='the height / width of the original input image')
    parser.add_argument('--imageSize', type=int, default=512, help='the height / width of the cropped input image to network')
    parser.add_argument('--inputChannelSize', type=int, default=3, help='size of the input channels')
    parser.add_argument('--outputChannelSize', type=int, default=3, help='size of the output channels')
    parser.add_argument('--sizePatchGAN', type=int, default=62)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--lambdaGAN', type=float, default=0.35, help='lambdaGAN')
    parser.add_argument('--lambdaIMG', type=float, default=1, help='lambdaIMG')
    parser.add_argument('--poolSize', type=int, default=50, help='Buffer size for storing previously generated samples from G')
    parser.add_argument('--wd', type=float, default=0.0000, help='weight decay in D')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--exp', default='sample', help='folder to output images and model checkpoints')
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    return parser.parse_args()

def save_GT(root, input, target, trans, ato, imgname):
    for category in ['input', 'clear', 'tran', 'atm']:
        gtPath = os.path.join(root+'GT', category)
        if not os.path.exists(gtPath):
            os.makedirs(gtPath)
        
        if category == 'input':
            vutils.save_image(input[0], os.path.join(gtPath, (imgname[0] + '_input.png')))
        elif category == 'clear':
            vutils.save_image(target[0], os.path.join(gtPath, (imgname[0] + '_clear.png')),
                            normalize=True, scale_each=False)
        elif category == 'tran':
            vutils.save_image(trans[0], os.path.join(gtPath, (imgname[0] + '_tran.png')), 
                            normalize=True, scale_each=False)
        elif category == 'atm':
            vutils.save_image(ato[0], os.path.join(gtPath, (imgname[0] + '_atm.png')), 
                            normalize=True, scale_each=False)
    

def test(opt, dataloader, netG):
    if not os.path.exists(f'{opt.outputRoot}/{opt.epoch}'):
        os.makedirs(f'{opt.outputRoot}/{opt.epoch}')
    
    val_target = torch.FloatTensor(opt.batchSize, opt.outputChannelSize, opt.imageSize, opt.imageSize).to(opt.device)
    val_input = torch.FloatTensor(opt.batchSize, opt.inputChannelSize, opt.imageSize, opt.imageSize).to(opt.device)
    val_batch_output = torch.FloatTensor(4, opt.inputChannelSize, opt.imageSize, opt.imageSize).fill_(0)

    netG.eval()
    with torch.no_grad():
        for data in tqdm(dataloader, desc='Testing'):
            val_input_cpu, val_target_cpu, val_tran_cpu, val_ato_cpu, imgname = data
        
            val_target_cpu, val_input_cpu = val_target_cpu.float().to(opt.device), val_input_cpu.float().to(opt.device)
            val_tran_cpu, val_ato_cpu = val_tran_cpu.float().to(opt.device), val_ato_cpu.float().to(opt.device)
            
            val_target.resize_as_(val_target_cpu).copy_(val_target_cpu)
            val_input.resize_as_(val_input_cpu).copy_(val_input_cpu)
            
            val_inputv = Variable(val_input, volatile=True)
            x_hat_val, x_hat_val2, x_hat_val3, dehaze21 = netG(val_inputv)
            val_batch_output[0].unsqueeze(0).copy_(val_input.data)
            val_batch_output[1].unsqueeze(0).copy_(dehaze21.data)
            val_batch_output[2].unsqueeze(0).copy_(x_hat_val2.data)
            val_batch_output[3].unsqueeze(0).copy_(x_hat_val3.data)
            
            vutils.save_image(val_batch_output, f'{opt.outputRoot}/{opt.epoch}/input_dehazed_tran_atp_{imgname[0]}.png', normalize=False, scale_each=False)
            

if __name__=="__main__":
    cudnn.benchmark = True
    cudnn.fastest = True
      
    opt = get_args()
    
    # opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)
    print("Random Seed: ", opt.manualSeed)
    
    # get dataloader    
    opt.dataset='pix2pix_val2'
    dataloader = getLoader(opt,
                           mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                           split='val', shuffle=False)
    

    # get models
    netG = net.dehaze(opt.inputChannelSize, opt.outputChannelSize, opt.ngf)
    netG.apply(weights_init)
    if opt.netG != '':
        opt.epoch = int(opt.netG.split('/')[-1][11:14])        # sample/netG_epoch_010.pth
        netG.load_state_dict(torch.load(opt.netG))
    netG.to(opt.device)
    
    print(opt)
    test(opt, dataloader, netG)
    
    