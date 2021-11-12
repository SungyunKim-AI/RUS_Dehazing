import torch
import cv2
import numpy as np

from torch.utils.data import DataLoader
from metrics import psnr,ssim
from dpt.models import DPTDepthModel
from HazeDataset import Dataset, NH_Haze_Dataset, NTIRE_Dataset, NYU_Dataset, NYU_Dataset_with_Notation, O_Haze_Dataset, RESIDE_Beta_Dataset, RESIDE_Beta_Dataset_With_Notation, BeDDE_Dataset, Dense_Haze_Dataset
e = 1e-8

def calc_airlight(hazy,clear,trans):
    airlight_nh= (hazy-clear*trans)/(1-trans+e)
    
    airlight_blue = np.mean(airlight_nh[:,:,0])
    airlight_green = np.mean(airlight_nh[:,:,1])
    airlight_red = np.mean(airlight_nh[:,:,2])
    
    airlight = np.array([airlight_blue,airlight_green,airlight_red])
    airlight = np.clip(airlight,0,1)
    airlight = np.reshape(airlight, (1,1,-1))
    
    size = hazy.shape[:2]
    airlight = cv2.resize(airlight,[size[1],size[0]])
    return airlight, airlight_nh

def calc_trans(hazy,clear,airlight):
    trans = (hazy-airlight)/(clear-airlight+e)
    #trans = np.clip(trans,0,1)
    #channel_size = airlight.shape[2]
    #trans = np.mean(trans,2)
    #trans = np.stack((trans,)*channel_size, axis=-1)
    return trans

def test2(model, test_loader, device):
    model.eval()
    for batch in test_loader:
        hazy_images, clear_images, airlight_post, _, airlight_input, _ = batch
        with torch.no_grad():
            hazy_images = hazy_images.to(device)
            clear_images = clear_images.to(device)
            airlight_post = airlight_post.to(device)
            _, hazy_trans = model.forward(hazy_images)
        

        hazy_trans = hazy_trans[0].unsqueeze(2).detach().cpu().numpy()
        hazy = hazy_images[0].detach().cpu().numpy().transpose(1,2,0)
        clear = clear_images[0].detach().cpu().numpy().transpose(1,2,0)
        airlight_post = airlight_post[0].detach().cpu().numpy().transpose(1,2,0)
        airlight_input = airlight_input[0].detach().cpu().numpy().transpose(1,2,0)

        prediction = (hazy-airlight_post)/(hazy_trans+e) + airlight_post

        trans_calc = calc_trans(hazy,clear,airlight_post)
        trans_gt = calc_trans(hazy,clear,airlight_input)

        print(trans_calc.shape)
        print(hazy.shape)
        pred_calc = (hazy-airlight_post)/(trans_calc+e) + airlight_post
        pred_gt = (hazy-airlight_input)/(trans_gt+e) + airlight_input

        
        ssim_ = ssim(torch.Tensor(prediction).unsqueeze(0), torch.Tensor(clear).unsqueeze(0))
        psnr_ = psnr(torch.Tensor(prediction).unsqueeze(0), torch.Tensor(clear).unsqueeze(0))
        
        print(ssim_)
        print(psnr_)
        
        hazy = cv2.cvtColor(hazy,cv2.COLOR_BGR2RGB)
        clear = cv2.cvtColor(clear,cv2.COLOR_BGR2RGB)

        prediction = cv2.cvtColor(prediction,cv2.COLOR_BGR2RGB)
        pred_calc = cv2.cvtColor(pred_calc,cv2.COLOR_BGR2RGB)
        pred_gt = cv2.cvtColor(pred_gt,cv2.COLOR_BGR2RGB)

        airlight_post = cv2.cvtColor(airlight_post,cv2.COLOR_BGR2RGB)
        airlight_input = cv2.cvtColor(airlight_input,cv2.COLOR_BGR2RGB)

        trans_calc = cv2.cvtColor(trans_calc,cv2.COLOR_BGR2RGB)
        trans_gt = cv2.cvtColor(trans_gt,cv2.COLOR_BGR2RGB)

        cv2.imshow("hazy",hazy)
        cv2.imshow("clear",clear)
        
        cv2.imshow("hazy_trans", hazy_trans)
        cv2.imshow("calc_trans", trans_calc)
        cv2.imshow("gt_trans", trans_gt)
        
        cv2.imshow("airlight_post",airlight_post)
        cv2.imshow("airlight_input",airlight_input)
        
        cv2.imshow("clear_prediction",prediction)
        cv2.imshow("calc_predction",pred_calc)
        cv2.imshow("gt_predction",pred_gt)


        cv2.waitKey(0)
            


def test(model, test_loader, device):
    model.eval()
    for batch in test_loader:
        hazy_images, clear_images, airlight_images = batch
        with torch.no_grad():
            hazy_images = hazy_images.to(device)
            clear_images = clear_images.to(device)
            airlight_images = airlight_images.to(device)
            _, hazy_trans = model.forward(hazy_images)
        
        #pred
        hazy_trans = hazy_trans[0].unsqueeze(2).detach().cpu().numpy()
        hazy = (hazy_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        clear = (clear_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        airlight = (airlight_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        #hazy_trans = np.clip(hazy_trans, 0,1)
        prediction = (hazy-airlight)/(hazy_trans+e) + airlight
        prediction = (np.clip(prediction,0,1)*255).astype('uint8')

        #calc
        trans_calc = calc_trans(hazy,clear,airlight)
        #trans_calc = np.clip(trans_calc,0,1)
        pred_calc = (hazy-airlight)/(trans_calc+e) + airlight
        pred_calc = (np.clip(pred_calc,0,1)*255).astype('uint8')
        
        #gray
        hazy_gray = cv2.cvtColor(hazy,cv2.COLOR_RGB2GRAY)
        clear_gray = cv2.cvtColor(clear,cv2.COLOR_RGB2GRAY)
        airlight_gray = cv2.cvtColor(airlight,cv2.COLOR_RGB2GRAY)

        hazy_gray = np.expand_dims(hazy_gray,2)
        clear_gray = np.expand_dims(clear_gray,2)
        airlight_gray = np.expand_dims(airlight_gray,2)

        trans_calc_gray = calc_trans(hazy_gray,clear_gray,airlight_gray)
        #trans_calc_gray = np.clip(trans_calc_gray,0,1)
        pred_calc_gray = (hazy-airlight)/(trans_calc_gray+e) + airlight
        pred_calc_gray = (np.clip(pred_calc_gray,0,1)*255).astype('uint8')
        
        ssim_ = ssim(torch.Tensor(prediction).unsqueeze(0), torch.Tensor(clear).unsqueeze(0))
        psnr_ = psnr(torch.Tensor(prediction).unsqueeze(0), torch.Tensor(clear).unsqueeze(0))
        
        print(ssim_)
        print(psnr_)
        
        hazy = cv2.cvtColor(hazy,cv2.COLOR_RGB2BGR)
        clear = cv2.cvtColor(clear,cv2.COLOR_RGB2BGR)
        prediction = cv2.cvtColor(prediction,cv2.COLOR_RGB2BGR)
        pred_calc = cv2.cvtColor(pred_calc,cv2.COLOR_RGB2BGR)
        airlight = cv2.cvtColor(airlight,cv2.COLOR_RGB2BGR)
        trans_calc = cv2.cvtColor(trans_calc,cv2.COLOR_RGB2BGR)
        pred_calc_gray = cv2.cvtColor(pred_calc_gray,cv2.COLOR_RGB2BGR)

        cv2.imshow("hazy",hazy)
        cv2.imshow("clear",clear)
        
        cv2.imshow("hazy_trans", hazy_trans)
        cv2.imshow("calc_trans", trans_calc)
        
        cv2.imshow("airlight",airlight)
        
        cv2.imshow("clear_prediction",prediction)
        cv2.imshow("calc_predction",pred_calc)

        cv2.imshow("calc_trans_gray", trans_calc_gray)
        cv2.imshow("calc_predction_gray",pred_calc_gray)
        cv2.waitKey(0)
            
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    epochs = 100
    net_w = 256
    net_h = 256
    input_path = 'input'
    output_path = 'output_dehazed'
    
    model = DPTDepthModel(
        path = 'weights/211109/dpt_hybrid-midas-501f0c75_trans_020.pt',
        scale=0.00006016,
        shift=0.00579,
        invert=True,
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
    )
    
    model = model.to(memory_format=torch.channels_last)
    #model = model.half()
    
    model.to(device)
    
    dataset_test=NTIRE_Dataset('D:/data',[net_w,net_h],flag='train',verbose=False)
    #dataset_test = RESIDE_Beta_Dataset_With_Notation('D:/data/RESIDE_Beta/train',[net_w,net_h])
    loader_test = DataLoader(
                dataset=dataset_test,
                batch_size=1,
                num_workers=0,
                drop_last=True,
                shuffle=False)
    
    test(model,loader_test,device)