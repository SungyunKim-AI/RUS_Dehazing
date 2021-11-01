# User warnings ignore
import warnings
warnings.filterwarnings("ignore")

# Python, Utils packages
import argparse, os, sys, random
import wandb
from tqdm import tqdm

# Pytorch
import torch
from torch import nn, optim
import torch.backends.cudnn as cudnn
import torch.nn.parallel

# Self-made Models, Utils
from misc import *
import models.dehaze22  as net
from myutils import utils
from myutils.vgg16 import Vgg16
from myutils.metrics import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=False, default='pix2pix',  help='')
    parser.add_argument('--dataroot', required=False, default='', help='path to trn dataset')
    parser.add_argument('--valDataroot', required=False, default='', help='path to val dataset')
    parser.add_argument('--mode', type=str, default='B2A', help='B2A: facade, A2B: edges2shoes')
    parser.add_argument('--seed', type=int, default=101, help='B2A: facade, A2B: edges2shoes')
    parser.add_argument('--batchSize', type=int, default=6, help='input batch size')
    parser.add_argument('--valBatchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--originalSize', type=int, default=286, help='the height / width of the original input image')
    parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the cropped input image to network')
    parser.add_argument('--inputChannelSize', type=int, default=3, help='size of the input channels')
    parser.add_argument('--outputChannelSize', type=int, default=3, help='size of the output channels')
    parser.add_argument('--sizePatchGAN', type=int, default=62)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=400, help='number of epochs to train for')
    parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--annealStart', type=int, default=0, help='annealing learning rate start to')
    parser.add_argument('--annealEvery', type=int, default=400, help='epoch to reaching at learning rate of 0')
    parser.add_argument('--lambdaGAN', type=float, default=0.35, help='lambdaGAN')
    parser.add_argument('--lambdaIMG', type=float, default=1, help='lambdaIMG')
    parser.add_argument('--poolSize', type=int, default=50, help='Buffer size for storing previously generated samples from G')
    parser.add_argument('--wd', type=float, default=0.0000, help='weight decay in D')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--exp', default='sample', help='folder to output images and model checkpoints')
    parser.add_argument('--display', type=int, default=5, help='interval for displaying train-logs')
    parser.add_argument('--evalIter', type=int, default=5, help='interval for evauating(generating) images from valDataroot')
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    return parser.parse_args()

# Two directional gradient loss function
def gradient(y):
    gradient_h=torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])
    gradient_y=torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])
    
    return gradient_h, gradient_y
  
def train_one_epoch(opt, dataloader, vgg, netG, netD, optimizerD, optimizerG, imagePool, epoch):
    lossesD_real, lossesD_fake, lossesG = [], [], []
    losses_trans, losses_content, losses_content1 = [], [], []
    netG.train()
    netD.train()
    for data in tqdm(dataloader):
        # optimizerD.zero_grad()
            
        for param in netD.parameters():
            param.requires_grad = True    # gradients compute
        
        netD.zero_grad()
        
        input, target, trans, ato, imgname = data
        input, target, trans, ato = input.to(opt.device), target.to(opt.device), trans.to(opt.device), ato.to(opt.device)
        x_hat, tran_hat, atp_hat, dehaze21 = netG(input)
        
        for param in netD.parameters():
            param.requires_grad = True    # gradients compute
        
        netD.zero_grad()
        
        # NOTE: compute L_cGAN in eq.(2)
        label_d = torch.full((opt.batchSize, 1, opt.sizePatchGAN, opt.sizePatchGAN), 1)
        output = netD(torch.cat([trans, target], 1)) # conditional
        errD_real = criterionBCE(output, label_d)
        errD_real.backward()
        # D_x = output.data.mean()
        # torch.new_tensor(output, requires_grad=True)
        
        fake = imagePool.query(x_hat)
        fake_trans = imagePool.query(fake_trans)

        label_d.fill_(0)
        output = netD(torch.cat([fake_trans, fake], 1)) # conditional
        errD_fake = criterionBCE(output, label_d)
        errD_fake.backward()
        # D_G_z1 = output.data.mean()
        # errD = errD_real + errD_fake
        
        optimizerD.step() # update parameters
        
        lossesD_real.append(errD_real.item())
        lossesD_fake.append(errD_fake.item())

        
        # prevent computing gradients of weights in Discriminator
        # optimizerG.zero_grad()
        for param in netD.parameters():
            param.requires_grad = False
        
        netG.zero_grad()    # start to update G

        # compute L_L1 (eq.(4) in the paper
        L_img = criterionCAE(x_hat, target) * opt.lambdaIMG
        if opt.lambdaIMG != 0:
            L_img.backward(retain_graph=True)
        
        # NOTE compute L1 for transamission map
        L_tran_ = criterionCAE(tran_hat, trans)

        # NOTE compute gradient loss for transamission map
        gradie_h_est, gradie_v_est = gradient(tran_hat)
        gradie_h_gt, gradie_v_gt = gradient(trans)

        L_tran_h = criterionCAE(gradie_h_est, gradie_h_gt)
        L_tran_v = criterionCAE(gradie_v_est, gradie_v_gt)

        L_tran =  opt.lambdaIMG * (L_tran_+ (2*L_tran_h) + (2*L_tran_v))
        if opt.lambdaIMG != 0:
            L_tran.backward(retain_graph=True)
        losses_trans.append(L_tran)

        # NOTE feature loss for transmission map
        features_content = vgg(trans)
        f_xc_c = features_content[1].detach().requires_grad_(False)
        
        features_y = vgg(tran_hat)
        content_loss =  0.8 * opt.lambdaIMG * criterionCAE(features_y[1], f_xc_c)
        content_loss.backward(retain_graph=True)
        losses_content.append(L_tran)

        # Edge Loss 2
        features_content = vgg(trans)
        f_xc_c = features_content[0].detach().requires_grad_(False)

        features_y = vgg(tran_hat)
        content_loss1 =  0.8 * opt.lambdaIMG * criterionCAE(features_y[0], f_xc_c)
        content_loss1.backward(retain_graph=True)
        losses_content1.append(L_tran)


        # NOTE compute L1 for atop-map
        L_ato_ = criterionCAE(atp_hat, ato)
        L_ato =  opt.lambdaIMG * L_ato_
        if opt.lambdaIMG != 0:
            L_ato_.backward(retain_graph=True)
        
        
        # compute  gan_loss for the joint discriminator
        label_d.fill_(1)
        output = netD(torch.cat([tran_hat, x_hat], 1))
        errG = criterionBCE(output, label_d) * opt.lambdaGAN

        if opt.lambdaGAN != 0:
            errG.backward()
        lossesG.append(errG)
        
        optimizerG.step()
    
    wandb.log({"lossesD_real" : lossesD_real, "lossesD_fake": lossesD_fake, "lossesG" : lossesG,
               "losses_trans":losses_trans, "losses_content":losses_content, "losses_content1":losses_content1,
               "global_step" : epoch})
    
    schedulerD.step()
    schedulerG.step()

def validate(opt, valDataloader, netG, epoch):
    img_ssim_epoch, img_psnr_epoch = 0.0, 0.0
    tran_ssim_epoch, tran_psnr_epoch = 0.0, 0.0
    
    netG.eval()
    with torch.no_grad():
        for data in tqdm(valDataloader):
            input, target, trans, ato, imgname = data
            input, target, trans, ato = input.to(opt.device), target.to(opt.device), trans.to(opt.device), ato.to(opt.device)
            
            x_hat, tran_hat, atp_hat, dehaze21 = netG(input)
            
            img_ssim, img_psnr = ssim(x_hat, target), psnr(x_hat, target)
            tran_ssim, tran_psnr = ssim(tran_hat, trans), psnr(tran_hat, trans)
            
            img_ssim_epoch += img_ssim.cpu().item()
            img_psnr_epoch += img_psnr.cpu().item()
            tran_ssim_epoch += tran_ssim.cpu().item()
            tran_psnr_epoch += tran_psnr.cpu().item()
            
            directory = os.path.join('output', f'{epoch:03d}')
            if not os.path.exists(directory):
                os.makedirs(directory)
                
            for i in range(opt.valBatchSize):
                utils.save_image(x_hat[i,:,:,:], os.path.join(directory, (imgname[i] + '_dehazed.png')), 
                                 normalize=True, scale_each=False)
                utils.save_image(tran_hat[i,:,:,:], os.path.join(directory, (imgname[i] + '_tran.png')), 
                                 normalize=True, scale_each=False)
                utils.save_image(atp_hat[i,:,:,:], os.path.join(directory, (imgname[i] + '_atm.png')), 
                                 normalize=True, scale_each=False)
                
    img_ssim_epoch /= i
    img_psnr_epoch /= i
    tran_ssim_epoch /= i
    tran_psnr_epoch /= i
    
    wandb.log({"IMAGE_SSIM" : img_ssim_epoch, "IMAGE_PSNR" : img_psnr_epoch,
               "TRAN_SSIM" : tran_ssim_epoch, "TRAN_SSIM" : tran_psnr_epoch,
               "global_step" : epoch})
    

if __name__=='__main__':
    cudnn.benchmark = True
    cudnn.fastest = True
      
    opt = get_args()
    print(opt)
    
    create_exp_dir(opt.exp)
    
    # opt.seed = random.randint(1, 10000)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    print("Random Seed: ", opt.seed)
  
    wandb.init(project='Dehazing', entity='rus')
    wandb.config.update(opt)
    wandb.run.name = 'DCPDN'
    
    # get dataloader
    dataloader = getLoader(opt, 
                           mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                           split='train', shuffle=True)
    
    opt.dataset='pix2pix_val2'
    valDataloader = getLoader(opt, 
                              mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                              split='val', shuffle=False)
    

    # get models
    netG = net.dehaze(opt.inputChannelSize, opt.outputChannelSize, opt.ngf)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
        print(netG)
    netG.to(opt.device)
        
    netD = net.D(opt.inputChannelSize + opt.outputChannelSize, opt.ndf)
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    netD.to(opt.device)

    # init Loss, Optimizer, LR_Scheduler
    criterionBCE = nn.BCELoss()
    criterionCAE = nn.L1Loss()
    
    optimizerD = optim.Adam(netD.parameters(), lr = opt.lrD, betas = (opt.beta1, 0.999), weight_decay=opt.wd)
    optimizerG = optim.Adam(netG.parameters(), lr = opt.lrG, betas = (opt.beta1, 0.999), weight_decay=0.00005)
    
    schedulerD = optim.lr_scheduler.StepLR(optimizerD, step_size=30, gamma=0.1)
    schedulerG = optim.lr_scheduler.StepLR(optimizerG, step_size=30, gamma=0.1)
    
    # image pool storing previously generated samples from G
    imagePool = ImagePool(opt.poolSize)
    
    # Initialize VGG-16
    vgg = Vgg16()
    utils.init_vgg16('models/')
    vgg.load_state_dict(torch.load(os.path.join('models/', "vgg16.weight")))
    vgg.to(opt.device)
    

    # NOTE training loop
    for epoch in range(1, opt.niter):
        train_one_epoch(opt, dataloader, 
                        vgg, netG, netD, 
                        optimizerD, optimizerG, 
                        criterionBCE, criterionCAE, 
                        schedulerD, schedulerG, 
                        imagePool, epoch)
        
        
        if epoch % opt.evalIter == 0:
            validate(opt, valDataloader, 
                     vgg, netG, netD,
                     imagePool, epoch)
            
            torch.save(netG.state_dict(), f'{opt.exp}/netG_epoch_{epoch:03d}.pth')
            torch.save(netD.state_dict(), f'{opt.exp}/netD_epoch_{epoch:03d}.pth')


