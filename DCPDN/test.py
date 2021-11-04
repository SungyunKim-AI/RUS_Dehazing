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
    parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the cropped input image to network')
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
        # elif category == 'clear':
        #     vutils.save_image(target[0], os.path.join(gtPath, (imgname[0] + '_clear.png')),
        #                     normalize=True, scale_each=False)
        # elif category == 'tran':
        #     vutils.save_image(trans[0], os.path.join(gtPath, (imgname[0] + '_tran.png')), 
        #                     normalize=True, scale_each=False)
        # elif category == 'atm':
        #     vutils.save_image(ato[0], os.path.join(gtPath, (imgname[0] + '_atm.png')), 
        #                     normalize=True, scale_each=False)
    

def test(opt, dataloader, netG, criterionCAE):
    loss_ato, i = 0.0, 0
    img_ssim, img_psnr = 0.0, 0.0
    tran_ssim, tran_psnr = 0.0, 0.0
    
    netG.eval()
    with torch.no_grad():
        for data in tqdm(dataloader, desc='Testing'):
            i += 1
            input, target, tran, ato, imgname = data
            input, target, tran, ato = input.to(opt.device).float(), target.to(opt.device).float(), tran.to(opt.device).float(), ato.to(opt.device).float()
            
            # save_GT(opt.outputRoot, input, target, trans, ato, imgname)
                        
            x_hat, tran_hat, atp_hat, dehaze21 = netG(input)
            
            dehazed = x_hat.detach().cpu()
            tran_map = tran_hat.detach().cpu()
            airlight = atp_hat.detach().cpu()
            
            target = target.detach().cpu()
            tran = tran.detach().cpu()
            
            loss_ato += criterionCAE(atp_hat, ato).item()
            img_ssim += ssim(dehazed, target).item()
            img_psnr += psnr(dehazed, target)
            tran_ssim += ssim(tran_map, tran).item()
            tran_psnr += psnr(tran_map, tran)
            
            dehazed = (dehazed.squeeze() * 0.5 + 0.5).numpy().transpose(1,2,0)
            # dehazed = (x_hat[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
            # tran = (tran_hat[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
            # atm = (atp_hat[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
            
            dehazed = cv2.cvtColor(dehazed,cv2.COLOR_BGR2RGB)
            # tran = cv2.cvtColor(tran,cv2.COLOR_BGR2RGB)
            
            # print(psnr(x_hat, target))
            # print(psnr(tran_hat, trans))
            cv2.imshow("HAZY", dehazed)
            cv2.waitKey(0)
            
            
            
            # for category in ['dehazed', 'tran', 'atm']:
            #     outputPath = os.path.join(opt.outputRoot+'hat', category)
            #     if not os.path.exists(outputPath):
            #         os.makedirs(outputPath)
                                
            #     if category == 'dehazed':
            #         vutils.save_image(x_hat[0], os.path.join(outputPath, (imgname[0] + '_dehazed.png')))
            #     elif category == 'tran':
            #         vutils.save_image(tran_hat[0], os.path.join(outputPath, (imgname[0] + '_tran.png')))
            #     elif category == 'atm':
            #         vutils.save_image(atp_hat[0], os.path.join(outputPath, (imgname[0] + '_atm.png')))
         
    loss_ato /= i
    img_ssim /= i
    img_psnr /= i 
    tran_ssim /= i
    tran_psnr /= i
    
    return {'loss_ato':loss_ato, 'img_ssim':img_ssim, 'img_psnr':img_psnr, 'tran_ssim':tran_ssim, 'tran_psnr':tran_psnr}

if __name__=="__main__":
    cudnn.benchmark = True
    cudnn.fastest = True
      
    opt = get_args()
    print(opt)
    
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
        netG.load_state_dict(torch.load(opt.netG))
    netG.to(opt.device)

    # init Loss
    criterionCAE = nn.L1Loss().to(opt.device)
    
    loss_val = test(opt, dataloader, netG, criterionCAE)
    print(loss_val)
    