from math import nan
import torch
import cv2
import numpy as np

from torch.utils.data import DataLoader

from dpt.models import DPTDepthModel
from HazeDataset import Dataset, NH_Haze_Dataset, NYU_Dataset, NYU_Dataset_with_Notation, O_Haze_Dataset, RESIDE_Beta_Dataset, RESIDE_Beta_Dataset_With_Notation, BeDDE_Dataset, Dense_Haze_Dataset
from metrics import ssim,psnr

def calc_beta(hazy, clear, airlight, depth):
    print(hazy.shape, clear.shape, airlight.shape, depth.shape)
    c = clear-airlight
    h = hazy-airlight
    beta = np.log(c/h)
    beta = beta/(depth+1e-8)
    beta = np.mean(beta)
    if beta is nan:
        return 0.001
    return beta

def calc_airlight(hazy,clear,trans):
    airlight_nh= (hazy-clear*trans)/(1-trans+1e-8)
    
    airlight_blue = np.mean(airlight_nh[:,:,0])
    airlight_green = np.mean(airlight_nh[:,:,1])
    airlight_red = np.mean(airlight_nh[:,:,2])
    
    airlight = np.array([airlight_blue,airlight_green,airlight_red])
    airlight = np.clip(airlight,0,1)
    airlight = np.reshape(airlight, (1,1,-1))
    return airlight, airlight_nh

def test3(model, test_loader, device):
    stage = 5
    model.eval()
    for batch in test_loader:
        hazy_images, clear_images= batch
        
        airlight = None
        one_shot_beta = 0.5
        init_beta = 0.01
        
        print(f'init_beta = {init_beta}')
        
        
        with torch.no_grad():
            hazy_images = hazy_images.to(device)
            clear_images = clear_images.to(device)
            _, hazy_depth = model.forward(hazy_images)
            
        hazy = (hazy_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        clear = (clear_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        
        hazy_depth = torch.clamp(hazy_depth,0,20).detach().cpu().numpy().transpose(1,2,0)/8
        hazy_trans = np.exp(hazy_depth * one_shot_beta * -1)
        airlight, airlight_nh = calc_airlight(hazy,clear,hazy_trans)
        prediction = (hazy-airlight)/(hazy_trans+1e-8) + airlight
        
        print(psnr(hazy, prediction),ssim(hazy,prediction))
        
        hazy = cv2.cvtColor(hazy,cv2.COLOR_BGR2RGB)
        clear = cv2.cvtColor(clear,cv2.COLOR_BGR2RGB)
        airlight_img = cv2.resize(airlight,[hazy.shape[1],hazy.shape[0]])
        airlight_img = cv2.cvtColor(airlight_img,cv2.COLOR_BGR2RGB)
        airlight_nh = cv2.cvtColor(airlight_nh,cv2.COLOR_BGR2RGB)
        prediction = np.clip(prediction,0,1)
        prediction = cv2.cvtColor((prediction*255).astype(np.uint8),cv2.COLOR_BGR2RGB)
        
        
        cv2.imshow("hazy",hazy)
        cv2.imshow("clear",clear)
        cv2.imshow("airlight",airlight_img)
        cv2.imshow("airlight_nh",airlight_nh)
        cv2.imshow('one-shot',prediction)
        
        beta = init_beta
        for i in range(1,stage+1):
            with torch.no_grad():
                hazy_images = hazy_images.to(device)
                _, hazy_depth = model.forward(hazy_images)
                            
            hazy = (hazy_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
            
            hazy_depth = torch.clamp(hazy_depth,0,20).detach().cpu().numpy().transpose(1,2,0)/8
            hazy_trans = np.exp(hazy_depth * beta * -1)
            
            airlight, _ = calc_airlight(hazy,clear,hazy_trans)
            prediction = (hazy-airlight)/(hazy_trans+1e-8) + airlight
            hazy_images = torch.Tensor(((prediction-0.5)/0.5).transpose(2,0,1)).unsqueeze(0)
            beta = calc_beta(hazy, prediction, airlight,hazy_depth) * 0.5
            print(beta)
            
            prediction = np.clip(hazy,0,1)
            prediction = cv2.cvtColor((prediction*255).astype(np.uint8),cv2.COLOR_BGR2RGB)
            
            cv2.imshow(f'{i}-stage_depth', hazy_depth/10)
            cv2.imshow(f'{i}-stage_prediction',prediction)
            print(f'{i}-stage complete')
        cv2.waitKey(0)

def test2(model, test_loader, device):
    stage = 20
    depth_threshold = 30
    model.eval()
    psnr_sum, ssim_sum = 0.0, 0.0
    for batch in test_loader:
        hazy_images, clear_images, airlight_images= batch
        beta = 0.8
        beta_per_stage = beta/stage
        
        print(f'beta = {beta}, beta_per_stage = {beta_per_stage}')
        
        
        with torch.no_grad():
            hazy_images = hazy_images.to(device)
            clear_images = clear_images.to(device)
            airlight_images = airlight_images.to(device)
            _, hazy_depth = model.forward(hazy_images)
            
        hazy = (hazy_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        clear = (clear_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        airlight = (airlight_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        
        hazy_depth = torch.clamp(hazy_depth,0,depth_threshold)
        hazy_depth = hazy_depth.detach().cpu().numpy().transpose(1,2,0)/8
        hazy_trans = np.exp(hazy_depth * beta * -1)
        prediction = (hazy-airlight)/(hazy_trans+1e-8) + airlight
        
        print(f'one-shot: psnr = {psnr(clear,prediction)}, ssim = {ssim(clear,prediction).item()}')
        
        prediction_img = np.clip(prediction,0,1)
        hazy_img = cv2.cvtColor(hazy,cv2.COLOR_BGR2RGB)
        clear_img = cv2.cvtColor(clear,cv2.COLOR_BGR2RGB)
        airlight_img = cv2.cvtColor(airlight,cv2.COLOR_BGR2RGB)
        prediction_img = cv2.cvtColor((prediction_img*255).astype(np.uint8),cv2.COLOR_BGR2RGB)
        
        
        cv2.imshow("hazy",hazy_img)
        cv2.imshow("clear",clear_img)
        cv2.imshow("airlight",airlight_img)
        cv2.imshow("depth", hazy_depth/10)
        cv2.imshow('one-shot',prediction_img)
        
        for i in range(1,stage+1):
            
            before_hazy_depth = hazy_depth.copy()
            with torch.no_grad():
                hazy_images = hazy_images.to(device)
                _, hazy_depth = model.forward(hazy_images)
                            
            hazy = (hazy_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
            
            hazy_depth = torch.clamp(hazy_depth,0,depth_threshold)
            hazy_depth = hazy_depth.detach().cpu().numpy().transpose(1,2,0)/8
            hazy_depth = np.minimum(hazy_depth,before_hazy_depth)
            hazy_trans = np.exp(hazy_depth * beta_per_stage * -1)
            #airlight, _ = calc_airlight(hazy,clear,hazy_trans)
            
            prediction = (hazy-airlight)/(hazy_trans+1e-8) + airlight
            hazy_images = torch.Tensor(((prediction-0.5)/0.5).transpose(2,0,1)).unsqueeze(0)
            
            
            if i%(stage/5)!=0:
                continue
            prediction_img = np.clip(hazy,0,1)
            prediction_img = cv2.cvtColor((prediction_img*255).astype(np.uint8),cv2.COLOR_BGR2RGB)
            
            cv2.imshow(f'{i}-stage_depth', hazy_depth/10)
            cv2.imshow(f'{i}-stage_prediction',prediction_img)
            #print(f'{i}-stage complete')
        
        final_psnr = psnr(clear,prediction)
        final_ssim = ssim(clear,prediction).item()
        print(f'{i}-stage: psnr = {final_psnr}, ssim = {final_ssim}')
        
        psnr_sum+=final_psnr
        ssim_sum+=final_ssim
        cv2.waitKey(0)
    
    batch_num = len(test_loader)
    print(f'mean_psnt = {psnr_sum/batch_num}, mean_ssim = {ssim_sum/batch_num}')

def test(model, test_loader, device):
    stage = 5
    model.eval()
    for batch in test_loader:
        hazy_images, clear_images, depth_images,airlight, beta = batch
        
        airlight = airlight.type(torch.FloatTensor).to(device)
        beta = beta.type(torch.FloatTensor).to(device)
        beta_per_stage = beta/5.0
        
        print(f'beta = {beta}, beta_per_stage = {beta_per_stage}')
        
        hazy = (hazy_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        clear = (clear_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        depth = depth_images.detach().cpu().numpy().transpose(1,2,0).astype(np.float32)
        
        hazy = cv2.cvtColor(hazy,cv2.COLOR_BGR2RGB)
        clear = cv2.cvtColor(clear,cv2.COLOR_BGR2RGB)
        
        with torch.no_grad():
            hazy_images = hazy_images.to(device)
            _, hazy_depth = model.forward(hazy_images)
            trans = torch.exp(hazy_depth/8 * beta * -1)
            prediction = (hazy_images-airlight)/(trans+1e-8) + airlight
            
        prediction = (hazy_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        prediction = np.clip(prediction,0,1)
        prediction = cv2.cvtColor((prediction*255).astype(np.uint8),cv2.COLOR_BGR2RGB)
        
        cv2.imshow("hazy",hazy)
        cv2.imshow("clear",clear)
        cv2.imshow('one-shot',prediction)
        
        for i in range(stage):
            with torch.no_grad():
                hazy_images = hazy_images.to(device)
                _, hazy_depth = model.forward(hazy_images)
                hazy_depth = torch.clamp(hazy_depth,0,20)/8
                trans = torch.exp(hazy_depth * beta_per_stage * -1)
                hazy_images = (hazy_images-airlight)/(trans+1e-8) + airlight
            
            hazy = (hazy_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
            hazy = np.clip(hazy,0,1)
            hazy = cv2.cvtColor((hazy*255).astype(np.uint8),cv2.COLOR_BGR2RGB)
            cv2.imshow(f'{i}-stage',hazy)
            print(f'{i}-stage complete')
        cv2.waitKey(0)
            
            
        

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    epochs = 100
    net_w = 320
    net_h = 240
    input_path = 'input'
    output_path = 'output_dehazed'
    
    model = DPTDepthModel(
        path = 'weights/dpt_hybrid-midas-501f0c75.pt',
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
    
    dataset_test= Dense_Haze_Dataset('D:/data/Dense_Haze/train',[net_w,net_h],printName=True)
    loader_test = DataLoader(
                dataset=dataset_test,
                batch_size=1,
                num_workers=0,
                drop_last=True,
                shuffle=True)
    
    test2(model,loader_test,device)