import torch
from models import DU_Net
from HazeDataset import O_Haze_Dataset, RESIDE_Beta_Dataset
from torch.utils.data import DataLoader
import cv2
import numpy as np

if __name__=='__main__':
    input_unet_channel = 3
    output_unet_channel = 3
    input_dis_channel = 3
    model = DU_Net(input_unet_channel ,output_unet_channel ,input_dis_channel).cuda()
    weight_path = 'weight/015'
    model.load(weight_path+'/generator.pth',weight_path+'/discriminator.pth')
    
    test_dataset = O_Haze_Dataset('D:/data/BEDDE/train')
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
            hazy_images, clear_images = one_batch
            unet_loss, dis_loss, mse, ssim, psnr, outputs = model.process(hazy_images.cuda(), clear_images.cuda())
            print(" || unet loss: "+str(unet_loss.cpu().item()) + " || dis loss: "+str(dis_loss.cpu().item()) + " || mse: "+str(mse.cpu().item()) + " | ssim:" + str(ssim.cpu().item()) + " | psnr:" + str(psnr))
            output = outputs[0].cpu().detach().numpy()
            output = output.transpose(1,2,0)
            
            hazy_image = hazy_images[0].cpu().detach().numpy()
            hazy_image = hazy_image.transpose(1,2,0)
            
            clear_image = clear_images[0].cpu().detach().numpy()
            clear_image = clear_image.transpose(1,2,0)
            
            
            print(hazy_image.shape)
            cv2.imshow("input",hazy_image)
            cv2.imshow("output",output)
            cv2.imshow("gt",clear_image)
            cv2.waitKey(0)
            
