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
from torchvision.models import vgg16

# Self-made Models, Utils
from misc import *
import models.dehaze22  as net
from myutils import utils

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=False, default='pix2pix',  help='')
    parser.add_argument('--dataroot', required=False, default='', help='path to trn dataset')
    parser.add_argument('--valDataroot', required=False, default='', help='path to val dataset')
    parser.add_argument('--mode', type=str, default='B2A', help='B2A: facade, A2B: edges2shoes')
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
  
def train_one_epoch(opt, netG, netD, dataloader, optim):
    netG.train()
    netD.train()
    
    for i, data in enumerate(tqdm(dataloader, 0)):
        optim.zero_grad()
        input, target, trans, ato, imgname = data
        
        x_hat, tran_hat, atp_hat, dehaze21 = netG(input)
        

        # NOTE: compute L_cGAN in eq.(2)
        label_d.resize_((opt.batchSize, 1, opt.sizePatchGAN, opt.sizePatchGAN)).fill_(real_label)
        output = netD(torch.cat([trans, target], 1)) # conditional
        errD_real = criterionBCE(output, label_d)
        errD_real.backward()
        D_x = output.data.mean()

        fake = x_hat.detach()
        fake = Variable(imagePool.query(fake.data))

        fake_trans = tran_hat.detach()
        fake_trans = Variable(imagePool.query(fake_trans.data))

        label_d.data.fill_(fake_label)
        output = netD(torch.cat([fake_trans, fake], 1)) # conditional
        errD_fake = criterionBCE(output, label_d)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step() # update parameters

        # prevent computing gradients of weights in Discriminator
        for p in netD.parameters():
          p.requires_grad = False


        netG.zero_grad() # start to update G



        # compute L_L1 (eq.(4) in the paper
        L_img_ = criterionCAE(x_hat, target)
        L_img = lambdaIMG * L_img_
        if lambdaIMG != 0:
          L_img.backward(retain_graph=True)



        # NOTE compute L1 for transamission map
        L_tran_ = criterionCAE(tran_hat, trans)

        # NOTE compute gradient loss for transamission map
        gradie_h_est, gradie_v_est=gradient(tran_hat)
        gradie_h_gt, gradie_v_gt=gradient(trans)

        L_tran_h = criterionCAE(gradie_h_est, gradie_h_gt)
        L_tran_v = criterionCAE(gradie_v_est, gradie_v_gt)

        L_tran =  lambdaIMG * (L_tran_+ 2*L_tran_h+ 2* L_tran_v)

        if lambdaIMG != 0:
            L_tran.backward(retain_graph=True)

        # NOTE feature loss for transmission map
        features_content = vgg(trans)
        f_xc_c = Variable(features_content[0].data, requires_grad=False)

        features_y = vgg(tran_hat)
        content_loss =  0.8*lambdaIMG* criterionCAE(features_y[0], f_xc_c)
        content_loss.backward(retain_graph=True)

        # Edge Loss 2
        features_content = vgg(trans)
        f_xc_c = Variable(features_content[0].data, requires_grad=False)

        features_y = vgg(tran_hat)
        content_loss1 =  0.8*lambdaIMG* criterionCAE(features_y[0], f_xc_c)
        content_loss1.backward(retain_graph=True)


        # NOTE compute L1 for atop-map
        L_ato_ = criterionCAE(atp_hat, ato)
        L_ato =  lambdaIMG * L_ato_
        if lambdaIMG != 0:
            L_ato_.backward(retain_graph=True)





        # compute  gan_loss for the joint discriminator
        label_d.data.fill_(real_label)
        output = netD(torch.cat([tran_hat, x_hat], 1))
        errG_ = criterionBCE(output, label_d)
        errG = lambdaGAN * errG_

        if lambdaGAN != 0:
            (errG).backward()

        optimizerG.step()

if __name__=='__main__':
    cudnn.benchmark = True
    cudnn.fastest = True
      
    opt = get_args()
    print(opt)
    
    create_exp_dir(opt.exp)
    
    opt.manualSeed = random.randint(1, 10000)
    # opt.manualSeed = 101
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)
    print("Random Seed: ", opt.manualSeed)
    
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

    # init Loss, Optimizer
    criterionBCE = nn.BCELoss()
    criterionCAE = nn.L1Loss()
    
    optimizerD = optim.Adam(netD.parameters(), lr = opt.lrD, betas = (opt.beta1, 0.999), weight_decay=opt.wd)
    optimizerG = optim.Adam(netG.parameters(), lr = opt.lrG, betas = (opt.beta1, 0.999), weight_decay=0.00005)
    
    # NOTE: size of 2D output maps in the discriminator
    sizePatchGAN = 30
    real_label = 1
    fake_label = 0
    
    # image pool storing previously generated samples from G
    imagePool = ImagePool(opt.poolSize)
    
    # NOTE weight for L_cGAN and L_L1 (i.e. Eq.(4) in the paper)
    lambdaGAN = opt.lambdaGAN
    lambdaIMG = opt.lambdaIMG
    
    # Initialize VGG-16
    vgg = vgg16(pretrained=True)
    vgg.to(opt.device)
    

    # NOTE training loop
    for epoch in range(1, opt.niter):
        if epoch > opt.annealStart:
            adjust_learning_rate(optimizerD, opt.lrD, epoch, None, opt.annealEvery)
            adjust_learning_rate(optimizerG, opt.lrG, epoch, None, opt.annealEvery)
        
        train_one_epoch(netG, netD, dataloader)
        
        # get paired data
        target.resize_as_(target_cpu).copy_(target_cpu)
        input.resize_as_(input_cpu).copy_(input_cpu)
        trans.resize_as_(trans_cpu).copy_(trans_cpu)
        ato.resize_as_(ato_cpu).copy_(ato_cpu)
 
        # target_cpu, input_cpu = target_cpu.float().to(opt.device), input_cpu.float().to(opt.device)
        # # NOTE paired samples
        # target.resize_as_(target_cpu).copy_(target_cpu)
        # input.resize_as_(input_cpu).copy_(input_cpu)
        # trans.resize_as_(trans_cpu).copy_(trans_cpu)


        
        # if (ganIterations % opt.display) == 0:
        #   print(f'[{epoch}/{opt.niter}][{i}/{len(dataloader)}] L_D: {L_tran_.item()} L_img: {L_tran_.item()} L_G: {L_img.item()} D(x): {L_img.item()} D(G(z)): {L_img.item()} / {L_img.item()}')
        #   sys.stdout.flush()
        #   trainLogger.write(f'{i}\t{L_img.item()}\t{L_img.item()}\t{L_img.item()}\t{L_img.item()}\t{L_img.item()}\t{L_img.item()}\n')
        #   trainLogger.flush()
        if epoch % opt.evalIter == 0:
          val_batch_output = torch.FloatTensor(val_input.size()).fill_(0)
          for idx in range(val_input.size(0)):
            single_img = val_input[idx,:,:,:].unsqueeze(0)
            val_inputv = Variable(single_img, volatile=True)
            x_hat_val, x_hat_val2, x_hat_val3, dehaze21 = netG(val_inputv)
            val_batch_output[idx,:,:,:].copy_(dehaze21.data.squeeze(0))
          torch.utils.save_image(val_batch_output,  
                            f'{opt.exp}/generated_epoch_{epoch:08d}_iter{ganIterations:08d}.png', 
                            normalize=False, scale_each=False)

      if epoch % 2 == 0:
            torch.save(netG.state_dict(), f'{opt.exp}/netG_epoch_{epoch}.pth')
            torch.save(netD.state_dict(), f'{opt.exp}/netD_epoch_{epoch}.pth')
    trainLogger.close()
