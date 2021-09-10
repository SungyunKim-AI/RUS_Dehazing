import torch
from models import DU_Net
from HazeDataset import O_Haze_Train_Dataset, RESIDE_Beta_Train_Dataset
from torch.utils.data import DataLoader
import cv2
import numpy as np

if __name__=='__main__':
    input_unet_channel = 3
    output_unet_channel = 3
    input_dis_channel = 3
    max_epochs = 100
    model = DU_Net(input_unet_channel ,output_unet_channel ,input_dis_channel).cuda()
    chk_path = 'weight/015'
    model.load(chk_path+'/generator.pth',chk_path+'/discriminator.pth')
    
    test_dataset = O_Haze_Train_Dataset('D:/data/BEDDE/train')
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        num_workers=0,
        drop_last=True,
        shuffle=False
    )
    print(test_dataset.__len__())
    model.eval()
    with torch.no_grad():
        for one_batch in test_loader:
            haze_images, dehaze_images = one_batch
            unet_loss, dis_loss, mse, ssim, psnr, outputs = model.process(haze_images.cuda(), dehaze_images.cuda())
            print(" || unet loss: "+str(unet_loss.cpu().item()) + " || dis loss: "+str(dis_loss.cpu().item()) + " || mse: "+str(mse.cpu().item()) + " | ssim:" + str(ssim.cpu().item()) + " | psnr:" + str(psnr))
            output = outputs[0].cpu().detach().numpy()
            output = output.transpose(1,2,0)
            
            hazy_image = haze_images[0].cpu().detach().numpy()
            hazy_image = hazy_image.transpose(1,2,0)
            
            dehaze_image = dehaze_images[0].cpu().detach().numpy()
            dehaze_image = dehaze_image.transpose(1,2,0)
            
            
            print(hazy_image.shape)
            cv2.imshow("input",hazy_image)
            cv2.imshow("output",output)
            cv2.imshow("gt",dehaze_image)
            cv2.waitKey(0)
            
