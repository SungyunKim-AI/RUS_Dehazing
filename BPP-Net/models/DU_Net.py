import torch
import torch.nn as nn
import torch.optim as optim

from lossFn import *
from .UNet import UNet
from metrics import *


class DU_Net(nn.Module):

    def __init__(self, unet_input, unet_output, discriminator_input):
        super().__init__()
        unet = UNet(in_channels=unet_input ,out_channels=unet_output)
        #unet = nn.DataParallel(unet, device_ids=[0, 1])
        unet = unet.cuda()

        discriminator = Discriminator(in_channels=discriminator_input , use_sigmoid=True)
        #discriminator = nn.DataParallel(discriminator, device_ids=[0, 1])
        discriminator = discriminator.cuda()

        criterion = nn.MSELoss()
        adversarial_loss = AdversarialLoss(type='hinge')
        l1_loss = nn.L1Loss()
        content_loss = ContentLoss()
        ssim = SSIM(window_size = 11)
        bce = nn.BCELoss()

        self.add_module('unet', unet)
        self.add_module('discriminator', discriminator)

        self.add_module('criterion', criterion)
        self.add_module('adversarial_loss', adversarial_loss)
        self.add_module('l1_loss', l1_loss)
        self.add_module('content_loss', content_loss)
        self.add_module('ssim_loss', ssim)
        self.add_module('bce_loss', bce)
        

        self.unet_optimizer = optim.Adam(
            unet.parameters(), 
            lr = float(0.00001),
            betas=(0.9, 0.999)
            )

        self.dis_optimizer = optim.Adam(
             params=discriminator.parameters(),
             lr=float(0.00001),
             betas=(0.9, 0.999)
             )

        self.unet_input = unet_input
        self.unet_output = unet_output
        self.discriminator_input = discriminator_input


    def load(self, path_unet, path_discriminator):
        weight_unet = torch.load(path_unet)
        weight_discriminator = torch.load(path_discriminator)
        self.unet.load_state_dict(weight_unet)
        self.discriminator.load_state_dict(weight_discriminator)

    def save_weight(self, path_unet, path_dis):
        torch.save(self.unet.state_dict(), path_unet)
        torch.save(self.discriminator.state_dict(), path_dis)

    def process(self, haze_images, dehaze_images):

        # zero optimizers
        self.unet_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()


        # find output and initialize loss to zero
        unet_loss = 0
        dis_loss = 0

        outputs = self.unet(haze_images.cuda())


        # discriminator loss
        dis_real, dis_real_feat = self.discriminator(dehaze_images.cuda())        
        dis_fake, dis_fake_feat = self.discriminator(outputs.detach().cuda())       
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2


        # unet loss
        unet_fake, unet_fake_feat = self.discriminator(outputs.cuda())        
        unet_gan_loss = self.adversarial_loss(unet_fake, True, False) * 0.7
        unet_loss += unet_gan_loss

        unet_criterion = self.criterion(outputs.cuda(), dehaze_images.cuda())
        unet_loss += unet_criterion


        gen_content_loss = self.content_loss(outputs.cuda(), dehaze_images.cuda())
        gen_content_loss = (gen_content_loss * 0.7).cuda()
        unet_loss += gen_content_loss.cuda()
        
        
        ssim_loss =  self.ssim_loss(outputs.cuda(), dehaze_images.cuda())
        ssim_loss = (1-ssim_loss)*2
        unet_loss += ssim_loss.cuda()
        
        psnr = self.psnr(outputs.cuda(), dehaze_images.cuda())
        
        return unet_loss, dis_loss, unet_criterion, 1-ssim_loss/2, psnr, outputs

    def backward(self, unet_loss, dis_loss):
        dis_loss.backward(retain_graph = True)
        self.dis_optimizer.step()

        unet_loss.backward()
        self.unet_optimizer.step()
        

    def predict(self, haze_images):
      predict_mask = self.unet(haze_images.cuda())
      return predict_mask
  

    def psnr(self, pred, gt):
        pred = pred.clamp(0,1).cpu().detach().numpy()
        gt = gt.clamp(0,1).cpu().detach().numpy()
        
        imdff = pred - gt
        rmse = math.sqrt(np.mean(imdff ** 2))
        if rmse == 0:
            return 100
        
        return 20 * math.log10(1.0 / rmse)