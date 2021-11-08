import torch
import cv2
import numpy as np

from torch.utils.data import DataLoader

from dpt.models import DPTDepthModel
from HazeDataset import Dataset, NH_Haze_Dataset, NYU_Dataset, NYU_Dataset_with_Notation, O_Haze_Dataset, RESIDE_Beta_Dataset, RESIDE_Beta_Dataset_With_Notation, BeDDE_Dataset, Dense_Haze_Dataset

def calc_airlight(hazy,clear,trans):
    airlight_nh= (hazy-clear*trans)/(1-trans+1e-8)
    
    airlight_blue = np.mean(airlight_nh[:,:,0])
    airlight_green = np.mean(airlight_nh[:,:,1])
    airlight_red = np.mean(airlight_nh[:,:,2])
    
    airlight = np.array([airlight_blue,airlight_green,airlight_red])
    airlight = np.clip(airlight,0,1)
    airlight = np.reshape(airlight, (1,1,-1))
    return airlight, airlight_nh

def test1(model, test_loader, device):
    model.eval()
    for batch in test_loader:
        hazy_images, clear_images = batch
        with torch.no_grad():
            hazy_images = hazy_images.to(device)
            clear_images = clear_images.to(device)
            _, hazy_depth = model.forward(hazy_images)
            _, clear_depth = model.forward(clear_images)
        
        hazy = (hazy_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        clear = (clear_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        
        airlight = 0.85
        beta = 0.3
        
        #hazy_depth = torch.clamp(hazy_depth,0,20)
        hazy_depth = hazy_depth.detach().cpu().numpy().transpose(1,2,0)/8
        
        #clear_depth = torch.clamp(clear_depth,0,20)
        clear_depth = clear_depth.detach().cpu().numpy().transpose(1,2,0)/8

        hazy_trans = np.exp(hazy_depth * beta * -1)
        
        airlight, airlight_nh = calc_airlight(hazy,clear,hazy_trans)
        
        print(f'air = {airlight}, beta = {beta}')
        
        prediction = (hazy-airlight)/(hazy_trans+1e-8) + airlight
        
        hazy = cv2.cvtColor(hazy,cv2.COLOR_BGR2RGB)
        clear = cv2.cvtColor(clear,cv2.COLOR_BGR2RGB)
        prediction = cv2.cvtColor(prediction,cv2.COLOR_BGR2RGB)
        airlight = cv2.resize(airlight,[hazy.shape[1],hazy.shape[0]])
        airlight = cv2.cvtColor(airlight,cv2.COLOR_BGR2RGB)
        airlight_nh = cv2.cvtColor(airlight_nh,cv2.COLOR_BGR2RGB)
        
        #hazy_depth = cv2.applyColorMap((hazy_depth/5*255).astype(np.uint8),cv2.COLORMAP_JET)
        
        cv2.imshow("hazy",hazy)
        cv2.imshow("clear",clear)
        
        cv2.imshow("hazy_depth",hazy_depth/10)
        cv2.imshow("clear_depth",clear_depth/10)
        
        cv2.imshow("hazy_trans", hazy_trans/5)
        
        cv2.imshow("airlight",airlight)
        cv2.imshow("airlight_nh",airlight_nh)
        
        cv2.imshow("clear_prediction",prediction)
        cv2.waitKey(0)
        
def test2(model,test_loader,device):
    
    model.eval()
    for batch in test_loader:
        hazy_images, clear_images, depth_images, airlight, beta = batch
        with torch.no_grad():
            hazy_images = hazy_images.to(device)
            clear_images = clear_images.to(device)
            depth_images = depth_images.to(device)
            _, hazy_depth = model.forward(hazy_images)
        
        hazy = (hazy_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        clear = (clear_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        depth = depth_images.detach().cpu().numpy().transpose(1,2,0).astype(np.float32)
        
        airlight = airlight.detach().numpy().astype(np.float32)
        beta = beta.detach().numpy().astype(np.float32)
        print(f'air = {airlight}, beta = {beta}')
        hazy_depth = torch.clamp(hazy_depth,0,20).detach().cpu().numpy().transpose(1,2,0)/8
        #hazy_depth =hazy_depth.detach().cpu().numpy().transpose(1,2,0)/8
        #print(np.max(hazy_depth,0))
        
        trans = np.exp(depth * beta * -1)
        hazy_trans = np.exp(hazy_depth * beta * -1)
        
        clear_notation_prediction = (hazy-airlight)/(trans+1e-8) + airlight
        clear_prediction = (hazy-airlight)/(hazy_trans+1e-8) + airlight
        

        
        hazy = cv2.cvtColor(hazy,cv2.COLOR_BGR2RGB)
        clear = cv2.cvtColor(clear,cv2.COLOR_BGR2RGB)
        clear_notation_prediction = cv2.cvtColor(clear_notation_prediction,cv2.COLOR_BGR2RGB)
        clear_prediction = cv2.cvtColor(clear_prediction,cv2.COLOR_BGR2RGB)
        
        #depth = cv2.applyColorMap((depth/20*255).astype(np.uint8),cv2.COLORMAP_JET)
        #hazy_depth = cv2.applyColorMap((hazy_depth/20*255).astype(np.uint8),cv2.COLORMAP_JET)
        
        cv2.imshow("hazy",hazy)
        cv2.imshow("clear",clear)
        
        cv2.imshow("depth",depth/10)
        cv2.imshow("hazy_depth",hazy_depth/10)
        
        cv2.imshow("trans", trans/5)
        cv2.imshow("hazy_trans", hazy_trans/5)
        
        cv2.imshow("clear_prediction",clear_prediction)
        cv2.imshow("clear_notation_prediction",clear_notation_prediction)
        cv2.waitKey(0)
            
def test3(model,test_loader,device):
    
    model.eval()
    for batch in test_loader:
        hazy_images, clear_images, airlight_images, trans_images = batch
        with torch.no_grad():
            hazy_images = hazy_images.to(device)
            clear_images = clear_images.to(device)
            _, hazy_depth = model.forward(hazy_images)
        
        hazy = (hazy_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        clear = (clear_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        airlight = (airlight_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        trans = (trans_images[0] * 0.5 + 0.5).detach().cpu().numpy().transpose(1,2,0)
        
        beta = 0.5
        print(f'air = {airlight[0,0,:]}, beta = {beta}')
        #hazy_depth = torch.clamp(hazy_depth,0,30).detach().cpu().numpy().transpose(1,2,0)/8
        hazy_depth =hazy_depth.detach().cpu().numpy().transpose(1,2,0)/8
        #print(np.max(hazy_depth,0))
        
        hazy_trans = np.exp(hazy_depth * beta * -1)
        
        clear_notation_prediction = (hazy-airlight)/(trans+1e-8) + airlight
        clear_prediction = (hazy-airlight)/(hazy_trans+1e-8) + airlight
        

        
        hazy = cv2.cvtColor(hazy,cv2.COLOR_BGR2RGB)
        clear = cv2.cvtColor(clear,cv2.COLOR_BGR2RGB)
        clear_notation_prediction = cv2.cvtColor(clear_notation_prediction,cv2.COLOR_BGR2RGB)
        clear_prediction = cv2.cvtColor(clear_prediction,cv2.COLOR_BGR2RGB)
        
        #depth = cv2.applyColorMap((depth/20*255).astype(np.uint8),cv2.COLORMAP_JET)
        #hazy_depth = cv2.applyColorMap((hazy_depth/20*255).astype(np.uint8),cv2.COLORMAP_JET)
        
        cv2.imshow("hazy",hazy)
        cv2.imshow("clear",clear)
        
        cv2.imshow("hazy_depth",hazy_depth/10)
        
        cv2.imshow("trans", trans/5)
        cv2.imshow("hazy_trans", hazy_trans/5)
        cv2.imshow("airlight", airlight)
        
        cv2.imshow("clear_prediction",clear_prediction)
        cv2.imshow("clear_notation_prediction",clear_notation_prediction)
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
    
    dataset_test=Dense_Haze_Dataset('D:/data/Dense_Haze/train',[net_w,net_h],printName=True)
    loader_test = DataLoader(
                dataset=dataset_test,
                batch_size=1,
                num_workers=0,
                drop_last=True,
                shuffle=False)
    
    test1(model,loader_test,device)
    #test(model,loader_test,device)
    #test3(model,loader_test,device)