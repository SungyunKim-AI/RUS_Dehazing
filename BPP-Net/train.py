import torch
import random, os, glob
import cv2
import numpy as np
import PIL
from PIL import Image

from models import *
from metrics import *
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader

class Dataset(torch.utils.data.Dataset):
    def __init__(self, haze_list, dehaze_list, augment=False):
        super().__init__()
        self.augment = augment
        self.haze_list = haze_list
        self.dehaze_list = dehaze_list
        
    def __len__(self):
        return 210

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.haze_list[index])
            item = self.load_item(0)
            

        return item


    def load_item(self, index):
        val = 1024*2           #crop size i.e hight and width
        size_data = 25         #depends on the no. of training images in the dataset
        height_data = 4657     #heigth of the training images
        width_data = 2833      #width of the training images

        numx = random.randint(0, height_data-val)
        numy = random.randint(0, width_data-val)

        haze_image = cv2.imread(self.haze_list[index%size_data])
        dehaze_image = cv2.imread(self.dehaze_list[index%size_data])
        haze_image = Image.fromarray(haze_image)
        dehaze_image = Image.fromarray(dehaze_image)

        haze_crop=haze_image.crop((numx, numy, numx+val, numy+val))
        dehaze_crop=dehaze_image.crop((numx, numy, numx+val, numy+val))
 
        haze_crop = haze_crop.resize((512,512), resample=PIL.Image.BICUBIC)
        dehaze_crop = dehaze_crop.resize((512,512), resample=PIL.Image.BICUBIC)

        haze_crop = np.array(haze_crop)
        dehaze_crop = np.array(dehaze_crop)
        haze_crop = cv2.cvtColor(haze_crop, cv2.COLOR_BGR2YCrCb)
        dehaze_crop = cv2.cvtColor(dehaze_crop, cv2.COLOR_BGR2YCrCb)
        haze_crop = self.to_tensor(haze_crop).cuda()
        dehaze_crop = self.to_tensor(dehaze_crop).cuda()
        
        return haze_crop.cuda(), dehaze_crop.cuda()
    
    def to_tensor(self, img):
        img_t = F.to_tensor(img).float()
        return img_t


    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item

def train(max_epochs, model):
    for epoch in range(max_epochs):
        i=1
        mse_epoch = 0.0
        ssim_epoch = 0.0
        unet_epoch = 0.0
        for haze_images, dehaze_images, in train_loader:
            unet_loss, dis_loss, mse, ssim = model.process(haze_images.cuda(), dehaze_images.cuda())
            model.backward(unet_loss.cuda(), dis_loss.cuda())
            print('Epoch: '+str(epoch+1)+ ' || Batch: '+str(i)+ " || unet loss: "+str(unet_loss.cpu().item()) + " || dis loss: "+str(dis_loss.cpu().item()) + " || mse: "+str(mse.cpu().item()) + " | ssim:" + str(ssim.cpu().item()) )
            mse_epoch =  mse_epoch + mse.cpu().item() 
            ssim_epoch = ssim_epoch + ssim.cpu().item()
            unet_epoch = unet_epoch + unet_loss.cpu().item()
            i=i+1
        
        print()
        mse_epoch = mse_epoch/i
        ssim_epoch = ssim_epoch/i
        unet_epoch = unet_epoch/i
        graph_gloss.append(ssim_epoch)
        print("mse: + "+str(mse_epoch) + " | ssim: "+ str(ssim_epoch)+ " | unet:"+str(unet_epoch))
        print()
        

if __name__ == '__main__':
    # ================ DataLoader ================
    path_of_train_hazy_images = 'train/haze/*.png'
    path_of_train_gt_images = 'train/gt/*.png'

    images_paths_train_gt=glob.glob(path_of_train_gt_images)
    image_paths_train_hazy=glob.glob(path_of_train_hazy_images)

    train_dataset = Dataset(image_paths_train_hazy, images_paths_train_gt, augment=False)

    train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=2,
                num_workers=0,
                drop_last=True,
                shuffle=False
            )
    
    # ================ Creating the model ================
    graph_gloss = []
    input_unet_channel = 3
    output_unet_channel = 3
    input_dis_channel = 3
    max_epochs = 100
    DUNet = DU_Net(input_unet_channel ,output_unet_channel ,input_dis_channel).cuda()
    
    # ================ Training ================
    epochs = 150
    train(epochs, DUNet)
    
    
    # ================ Saving weights ================
    path_of_generator_weight = 'weight/generator.pth'  #path for storing the weights of genertaor
    path_of_discriminator_weight = 'weight/discriminator.pth'  #path for storing the weights of discriminator
    DUNet.save_weight(path_of_generator_weight,path_of_discriminator_weight)
    
    path_of_generator_weight = 'weight/generator.pth'  #path where the weights of genertaor are stored
    path_of_discriminator_weight = 'weight/discriminator.pth'  #path where the weights of discriminator are stored
    DUNet.load('weights/new/in all/u_' + str(21) + '.pth','weights/new/in all/d_' + str(21) + '.pth')