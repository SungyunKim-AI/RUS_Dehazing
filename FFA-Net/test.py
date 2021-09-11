import torch
from models import FFA
from HazeDataset import O_Haze_Dataset, RESIDE_Beta_Dataset
from torch.utils.data import DataLoader, dataset
import cv2
from torch import nn
from metrics import psnr, ssim

if __name__=='__main__':
    gps = 3
    blocks = 20
    img_size = [512,512]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FFA(gps,blocks).to(device)
    weight_path = 'weights/weight_015.pth'
    weight = torch.load(weight_path)
    model.load_state_dict(weight)
    
    #test_dataset = RESIDE_Beta_Dataset('D:/data/RESIDE-beta/train',[0])
    test_dataset = O_Haze_Dataset('D:/data/O-Haze/train',img_size)
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
            
            hazy_images = hazy_images.to(device)
            clear_images = clear_images.to(device)
            outputs = model(hazy_images)
            
            MSELoss = nn.MSELoss()
            mse_ = MSELoss(outputs, clear_images)
            ssim_ = ssim(outputs, clear_images)
            psnr_ = psnr(outputs, clear_images)
            
            print("mse: "+str(mse_.cpu().item()) + " | ssim:" + str(ssim_.cpu().item()) + " | psnr:" + str(psnr_))
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
            
