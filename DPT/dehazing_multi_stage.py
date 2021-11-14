from math import nan
import torch
import cv2
import numpy as np
import csv
from torch.nn.functional import l1_loss
from tqdm import tqdm
from torch.utils.data import DataLoader
from airlight_estimation import Airlight_Module, show_plt
import matplotlib.pyplot as plt

from dpt.models import DPTDepthModel
from HazeDataset import *
from entropy_module import Entropy_Module
from metrics import ssim,psnr

e=1e-8


def show_histogram(img,index):
    
    val, cnt = np.unique(img, return_counts=True)
    img_size = img.shape[0] * img.shape[1]
    #prob = cnt / img_size    # PMF
    prob = cnt
    
    l = np.zeros((np.max(val)+1))
    for i, v in enumerate(val):
        l[v] = prob[i]
    show_plt(range((np.max(val)+1)), l,index)
    
def test_with_D_Hazy_NYU_notation(model, test_loader, device):
    entropy_module = Entropy_Module()
    stage = 100
    model.eval()
    psnr_sum, ssim_sum = 0.0, 0.0
    step_beta = 0.05
    
    a = 0.0018
    b = 3.95
    
    a_sum, b_sum= 0,0
    for cnt, batch in tqdm(enumerate(test_loader)):
        
        last_etp = 0
        hazy_images, clear_images, airlight_images, depth_images = batch
        
        print(f'beta_per_stage = {step_beta}')
        
        
        with torch.no_grad():
            hazy_images = hazy_images.to(device)
            clear_images = clear_images.to(device)
            airlight_images = airlight_images.to(device)
            _, init_depth = model.forward(hazy_images)
            _, init_clear_depth = model.forward(clear_images)
            
        init_hazy = (hazy_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        init_clear = (clear_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        init_airlight = (airlight_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        depth_gt = (depth_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        
        init_depth = init_depth.detach().cpu().numpy().transpose(1,2,0)
        init_depth = a*init_depth+b
        #init_depth = depth_gt.copy()
        
        init_clear_depth = init_clear_depth.detach().cpu().numpy().transpose(1,2,0)
        init_clear_depth = a*init_clear_depth+b
        
        _psnr = 0
        _ssim = 0
        step=0
        
        depth = init_depth.copy()
        prediction = None
        last_depth = None
        last_prediction = None
        
        
        for i in range(1,stage):
            
            last_depth = depth.copy()
            last_prediction = prediction
            last_psnr = _psnr
            last_ssim = _ssim
            
            
            with torch.no_grad():
                hazy_images = hazy_images.to(device)
                _, depth = model.forward(hazy_images)
                            
            hazy = (hazy_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
            depth = depth.detach().cpu().numpy().transpose(1,2,0)
            
            
            '''
            print(np.min(depth))
            print(np.max(depth))
            
            a = np.sqrt(np.var(depth_gt)/np.var(depth))
            print(f'{a}')
            depth_gt_mean = np.mean(depth_gt)
            depth_mean = np.mean(depth)
            b = depth_gt_mean - a * depth_mean
            print(f'{b}')
            
            a_sum += a
            b_sum += b
            
            print(f'a_mean = {a_sum/(cnt+1)}, b_mean = {b_sum/(cnt+1)}')
            
            #show_histogram((depth).astype(np.int32),131)
            #show_histogram(((a*depth+b)*255).astype(np.int32),132)
            #show_histogram((depth_gt*255).astype(np.int32),133)
            #plt.show()
            break
            '''
            
            
            
            depth = a*depth+b
            depth = np.minimum(depth,last_depth)
            #depth = depth_gt.copy()
            
            trans = np.exp(depth * step_beta * -1)
            
            prediction = (hazy-init_airlight)/(trans+e) + init_airlight
            prediction = np.clip(prediction,0,1)
            hazy_images = torch.Tensor(((prediction-0.5)/0.5).transpose(2,0,1)).unsqueeze(0)
            
            cur_etp = entropy_module.get_entropy((prediction*255).astype(np.uint8))
            diff_etp = cur_etp - last_etp
            
            #print(f'{i}-stage')
            #print(f'cur_etp = {cur_etp}')
            #print(f'last_etp = {last_etp}')
            #print(f'etp_diff = {diff_etp}')
            
            _psnr = psnr(init_clear,prediction)
            _ssim = ssim(init_clear,prediction).item()
            #print(f'psnr = {_psnr}, ssim = {_ssim}')
            #print()
            
            if diff_etp<0 or i==stage-1:
                step = i-1
                psnr_sum +=last_psnr
                ssim_sum +=last_ssim
                print(f'last_stage = {step}')
                print(f'last_psnr  = {last_psnr}')
                print(f'last_ssim  = {last_ssim}')
                print(f'last_etp   = {last_etp}')
                last_etp = cur_etp
                break
            
            last_etp = cur_etp
        
        #continue
        
        trans = np.exp(init_depth * (step_beta*step) * -1)
        one_shot_prediction = (init_hazy-init_airlight)/(trans+e) + init_airlight
        one_shot_prediction = np.clip(one_shot_prediction,0,1)
        _psnr = psnr(init_clear,one_shot_prediction)
        _ssim = ssim(init_clear,one_shot_prediction).item()
        clear_etp = entropy_module.get_entropy((init_clear*255).astype(np.uint8))
        print(f'clear_etp  = {clear_etp}')
        print(f'one-shot: beta = {step_beta*step}, psnr = {_psnr}, ssim={_ssim}')
        
        last_prediction = cv2.cvtColor(last_prediction.astype(np.float32),cv2.COLOR_BGR2RGB)
        init_hazy_img = cv2.cvtColor(init_hazy,cv2.COLOR_RGB2BGR)
        init_clear_img = cv2.cvtColor(init_clear,cv2.COLOR_RGB2BGR)
        init_airlight_img = cv2.cvtColor(init_airlight,cv2.COLOR_RGB2BGR)
        one_shot_prediction = cv2.cvtColor(one_shot_prediction.astype(np.float32),cv2.COLOR_BGR2RGB)
        
        cv2.imshow('final depth', depth/10)
        cv2.imshow('last stage prediction',last_prediction)
        cv2.imshow("init_hazy",init_hazy_img)
        cv2.imshow("init_clear",init_clear_img)
        cv2.imshow("init_airlight",init_airlight_img)
        cv2.imshow("init_depth", init_depth/10)
        cv2.imshow("init_clear_depth", init_clear_depth/10)
        cv2.imshow("depth_gt",depth_gt)
        cv2.imshow('one_shot_prediction',one_shot_prediction)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    

def test_with_RESIDE_notation(model, test_loader, device):
    entropy_module = Entropy_Module()
    stage = 100
    model.eval()
    psnr_sum, ssim_sum = 0.0, 0.0
    step_beta = 0.005
    
    a = 0.0018
    b = 3.95
    
    a_sum, b_sum= 0,0
    for cnt, batch in tqdm(enumerate(test_loader)):
        
        last_etp = 0
        hazy_images, clear_images, airlight_images, depth_images, airlight_gt, beta_gt= batch
        
        print(f'beta_per_stage = {step_beta}')
        
        
        with torch.no_grad():
            hazy_images = hazy_images.to(device)
            clear_images = clear_images.to(device)
            airlight_images = airlight_images.to(device)
            _, init_depth = model.forward(hazy_images)
            _, init_clear_depth = model.forward(clear_images)
            
        init_hazy = (hazy_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        init_clear = (clear_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        init_airlight = (airlight_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        depth_gt = depth_images.detach().cpu().numpy().transpose(1,2,0).astype(np.float32)
        
        init_depth = init_depth.detach().cpu().numpy().transpose(1,2,0)
        init_depth = a*init_depth+b
        #init_depth = depth_gt
        
        init_clear_depth = init_clear_depth.detach().cpu().numpy().transpose(1,2,0)
        init_clear_depth = a*init_clear_depth+b
        
        _psnr = 0
        _ssim = 0
        step=0
        
        depth = init_depth.copy()
        prediction = None
        last_depth = None
        last_prediction = None
        
        
        for i in range(1,stage):
            
            last_depth = depth.copy()
            last_prediction = prediction
            last_psnr = _psnr
            last_ssim = _ssim
            
            
            with torch.no_grad():
                hazy_images = hazy_images.to(device)
                _, depth = model.forward(hazy_images)
                            
            hazy = (hazy_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
            depth = depth.detach().cpu().numpy().transpose(1,2,0)
            
            
            print(np.min(depth))
            print(np.max(depth))
            
            a = np.sqrt(np.var(depth_gt)/np.var(depth))
            print(f'{a}')
            depth_gt_mean = np.mean(depth_gt)
            depth_mean = np.mean(depth)
            b = depth_gt_mean - a * depth_mean
            print(f'{b}')
            
            a_sum += a
            b_sum += b
            
            print(f'a_mean = {a_sum/(cnt+1)}, b_mean = {b_sum/(cnt+1)}')
            
            #show_histogram((depth).astype(np.int32),131)
            #show_histogram(((a*depth+b)*255).astype(np.int32),132)
            #show_histogram((depth_gt*255).astype(np.int32),133)
            #plt.show()
            break
            
            
            
            
            depth = a*depth+b
            depth = np.minimum(depth,last_depth)
            #depth = depth_gt
            
            trans = np.exp(depth * step_beta * -1)
            
            prediction = (hazy-init_airlight)/(trans+e) + init_airlight
            prediction = np.clip(prediction,0,1)
            hazy_images = torch.Tensor(((prediction-0.5)/0.5).transpose(2,0,1)).unsqueeze(0)
            
            cur_etp = entropy_module.get_entropy((prediction*255).astype(np.uint8))
            diff_etp = cur_etp - last_etp
            
            #print(f'{i}-stage')
            #print(f'cur_etp = {cur_etp}')
            #print(f'last_etp = {last_etp}')
            #print(f'etp_diff = {diff_etp}')
            
            _psnr = psnr(init_clear,prediction)
            _ssim = ssim(init_clear,prediction).item()
            #print(f'psnr = {_psnr}, ssim = {_ssim}')
            #print()
            
            if diff_etp<0 or i==stage-1:
                step = i-1
                psnr_sum +=last_psnr
                ssim_sum +=last_ssim
                print(f'last_stage = {step}')
                print(f'last_psnr  = {last_psnr}')
                print(f'last_ssim  = {last_ssim}')
                print(f'last_etp   = {last_etp}')
                last_etp = cur_etp
                break
            
            last_etp = cur_etp
        
        continue
        
        trans = np.exp(init_depth * (step_beta*step) * -1)
        one_shot_prediction = (init_hazy-init_airlight)/(trans+e) + init_airlight
        one_shot_prediction = np.clip(one_shot_prediction,0,1)
        _psnr = psnr(init_clear,one_shot_prediction)
        _ssim = ssim(init_clear,one_shot_prediction).item()
        clear_etp = entropy_module.get_entropy((init_clear*255).astype(np.uint8))
        print(f'clear_etp  = {clear_etp}')
        print(f'one-shot: beta = {step_beta*step}, psnr = {_psnr}, ssim={_ssim}')
        
        last_prediction = cv2.cvtColor(last_prediction.astype(np.float32),cv2.COLOR_BGR2RGB)
        init_hazy_img = cv2.cvtColor(init_hazy,cv2.COLOR_RGB2BGR)
        init_clear_img = cv2.cvtColor(init_clear,cv2.COLOR_RGB2BGR)
        init_airlight_img = cv2.cvtColor(init_airlight,cv2.COLOR_RGB2BGR)
        one_shot_prediction = cv2.cvtColor(one_shot_prediction.astype(np.float32),cv2.COLOR_BGR2RGB)
        
        cv2.imshow('final depth', depth/10)
        cv2.imshow('last stage prediction',last_prediction)
        cv2.imshow("init_hazy",init_hazy_img)
        cv2.imshow("init_clear",init_clear_img)
        cv2.imshow("init_airlight",init_airlight_img)
        cv2.imshow("init_depth", init_depth/10)
        cv2.imshow("init_clear_depth", init_clear_depth/10)
        cv2.imshow("depth_gt",depth_gt/10)
        cv2.imshow('one_shot_prediction',one_shot_prediction)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def test(model, test_loader, device, show=False, write_csv=False):
    entropy_module = Entropy_Module()
    stage = 100
    model.eval()
    psnr_sum, ssim_sum = 0.0, 0.0
    step_beta = 0.01
    print(f'beta_per_stage = {step_beta}')
    a = 0.0018
    b = 3.95
    for batch in tqdm(test_loader):
        
        last_etp = 0
        hazy_images, clear_images, airlight_images, input_name= batch
        
        csv_wr = None
        csv_file = None
        if write_csv:
            csv_name = input_name[0][:-3]+'csv'
            csv_file = open(csv_name,'w',newline='')
            csv_wr = csv.writer(csv_file)
            csv_wr.writerow(['stage','step_beta','cur_etp','diff_etp','psnr','ssim'])
        
        
        with torch.no_grad():
            hazy_images = hazy_images.to(device)
            clear_images = clear_images.to(device)
            airlight_images = airlight_images.to(device)
            _, init_depth = model.forward(hazy_images)
            _, init_clear_depth = model.forward(clear_images)
            
        init_hazy = (hazy_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        init_clear = (clear_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        init_airlight = (airlight_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        
        init_depth = init_depth.detach().cpu().numpy().transpose(1,2,0)
        init_depth = a*init_depth+b
        
        init_clear_depth = init_clear_depth.detach().cpu().numpy().transpose(1,2,0)
        init_clear_depth = a*init_clear_depth+b
        
        _psnr = 0
        _ssim = 0
        step=0
        
        depth = init_depth.copy()
        prediction = None
        last_depth = None
        last_prediction = None
        finding_max=True
        
        for i in range(1,stage+1):
            
            last_depth = depth.copy()
            last_prediction = prediction
            last_psnr = _psnr
            last_ssim = _ssim
            
            
            with torch.no_grad():
                hazy_images = hazy_images.to(device)
                _, depth = model.forward(hazy_images)
                            
            hazy = (hazy_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
            
            depth = depth.detach().cpu().numpy().transpose(1,2,0)
            depth = a*depth+b
            depth = np.minimum(depth,last_depth)
            
            trans = np.exp(depth * step_beta * -1)
            
            prediction = (hazy-init_airlight)/(trans+e) + init_airlight
            prediction = np.clip(prediction,0,1)
            hazy_images = torch.Tensor(((prediction-0.5)/0.5).transpose(2,0,1)).unsqueeze(0)
            
            cur_etp = entropy_module.get_entropy((prediction*255).astype(np.uint8))
            diff_etp = cur_etp - last_etp
            
            _psnr = psnr(init_clear,prediction)
            _ssim = ssim(init_clear,prediction).item()
            
            if (diff_etp<0 or i==stage) and finding_max:
                step = i-1
                psnr_sum +=last_psnr
                ssim_sum +=last_ssim
                print(f'last_stage = {step}')
                print(f'last_psnr= {last_psnr}')
                print(f'last_ssim= {last_ssim}')
                print(f'last_etp   = {last_etp}')
                finding_max=False
                if write_csv==False:
                    last_etp = cur_etp
                    break
                
            last_etp = cur_etp
            
            if write_csv:
                csv_wr.writerow([i,step_beta,cur_etp,diff_etp,_psnr,_ssim])
                
        if write_csv:
            csv_file.close()
        
        trans = np.exp(init_depth * (step_beta*step) * -1)
        one_shot_prediction = (init_hazy-init_airlight)/(trans+e) + init_airlight
        one_shot_prediction = np.clip(one_shot_prediction,0,1)
        _psnr = psnr(init_clear,one_shot_prediction)
        _ssim = ssim(init_clear,one_shot_prediction).item()
        clear_etp = entropy_module.get_entropy((init_clear*255).astype(np.uint8))
        print(f'clear_etp  = {clear_etp}')
        print(f'one-shot: beta = {step_beta*step}, psnr = {_psnr}, ssim={_ssim}')
        
        last_prediction = cv2.cvtColor(last_prediction.astype(np.float32),cv2.COLOR_BGR2RGB)
        init_hazy_img = cv2.cvtColor(init_hazy,cv2.COLOR_RGB2BGR)
        init_clear_img = cv2.cvtColor(init_clear,cv2.COLOR_RGB2BGR)
        init_airlight_img = cv2.cvtColor(init_airlight,cv2.COLOR_RGB2BGR)
        one_shot_prediction = cv2.cvtColor(one_shot_prediction.astype(np.float32),cv2.COLOR_BGR2RGB)
        
        if show:
            cv2.imshow('final depth', depth/10)
            cv2.imshow('last stage prediction',last_prediction)
            cv2.imshow("init_hazy",init_hazy_img)
            cv2.imshow("init_clear",init_clear_img)
            cv2.imshow("init_airlight",init_airlight_img)
            cv2.imshow("init_depth", init_depth/10)
            cv2.imshow("init_clear_depth", init_clear_depth/10)
            cv2.imshow('one_shot_prediction',one_shot_prediction)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    batch_num = len(test_loader)
    print(f'mean_psnr = {psnr_sum/batch_num}, mean_ssim = {ssim_sum/batch_num}')

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    epochs = 100
    net_w = 256
    net_h = 256
    input_path = 'input'
    output_path = 'output_dehazed'
    
    model = DPTDepthModel(
        path = 'weights/dpt_hybrid-midas-501f0c75.pt',
        scale=1,
        shift=0,
        invert=False,
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
    )
    
    model = model.to(memory_format=torch.channels_last)
    #model = model.half()
    
    model.to(device)
    
    
    dataset_test = D_Hazy_NYU_Dataset_With_Notation('D:/data/D_Hazy_NYU/train',[net_w,net_h],printName=False)
    loader_test = DataLoader(
                dataset=dataset_test,
                batch_size=1,
                num_workers=0,
                drop_last=False,
                shuffle=False)
    
    test_with_D_Hazy_NYU_notation(model,loader_test,device)
    #test(model,loader_test,device,show=False,write_csv=False)