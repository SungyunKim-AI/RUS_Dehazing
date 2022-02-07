import argparse
from turtle import clear
import cv2
import numpy as np
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import torch
from dataset import *
from utils.entropy_module import Entropy_Module
from utils.airlight_module import Airlight_Module
from utils import util
from utils.metrics import get_ssim, get_psnr
import os
import csv
from densedepth import *
    
def predict(model, module, images):
    pred = torch.clamp(1000/model.forward(images), 0, 1000)/1000
    pred_y_flip = torch.clamp(1000/model.forward(torch.fliplr(images)),0,1000)/1000

    depth_images = module(0.5 * pred + 0.5*(torch.fliplr(pred_y_flip)))
    return depth_images



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--betaStep', type=float, default=0.005, help='beta step')
    parser.add_argument('--norm', action='store_true',  help='Image Normalize flag')
    # NYU
    parser.add_argument('--dataset', required=False, default='KITTI',  help='dataset name')
    parser.add_argument('--dataRoot', type=str, default='D:/data/KITTI',  help='data file path')
    return parser.parse_args()

def print_score(score):
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = score
    print(f'{abs_rel:.2f} {sq_rel:.2f} {rmse:.2f} {rmse_log:.2f} | {a1:.2f} {a2:.2f} {a3:.2f}')

def run(opt, model, loader, airlight_module, entropy_module):
    up_module = torch.nn.Upsample(scale_factor=(2,2)).to('cuda').eval()
    
    output_folder = 'output/DenseDenpth_depth_' + opt.dataset
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for batch in tqdm(loader):
        # hazy_input, clear_input, GT_depth, GT_airlight, GT_beta, haze
        hazy_images, clear_images, depth_images, gt_airlight, gt_beta, input_names = batch
        # print(input_names)
        with torch.no_grad():
            clear_images = clear_images.to('cuda')
            gt_depth_median = torch.median(depth_images)
            depth_images = predict(model, up_module, clear_images)
            init_ratio = gt_depth_median / torch.median(depth_images).item()
            depth_images *= init_ratio
            #depth_images = 1/up_module(model.forward(clear_images))*90

            trans = torch.exp(depth_images*gt_beta.item()*-1)
            gt_airlight = util.air_denorm(opt.dataset, opt.norm, gt_airlight)[0][0]
            hazy_images = clear_images*trans + gt_airlight*(1-trans)
            cur_hazy = hazy_images

            init_depth = predict(model, up_module, cur_hazy)*init_ratio

            
        output_name = output_folder + '/' + input_names[0][:-4] + '/' + input_names[0][:-4] + '.csv'
        if not os.path.exists(f'{output_folder}/{input_names[0][:-4]}'):
            os.makedirs(f'{output_folder}/{input_names[0][:-4]}')
        f = open(output_name,'w', newline='')
        wr = csv.writer(f)
        
        cur_depth = None
        sum_depth = torch.zeros_like(init_depth).to('cuda')

        airlight = airlight_module.get_airlight(cur_hazy, opt.norm)
        airlight = util.air_denorm(opt.dataset, opt.norm, airlight)


        # airlight = util.air_denorm(opt.dataset, opt.norm, airlight).item()
        # airlight = util.air_denorm(opt.dataset, opt.norm, gt_airlight).item()

        # print('airlight = ', airlight, 'gt_airlight = ', util.air_denorm(opt.dataset, opt.norm, gt_airlight).item())

        steps = int((gt_beta*2) / opt.betaStep)
        dehaze = None
        for step in range(0,steps):
            # if step == int(gt_beta/opt.betaStep)-1:
            #     print('gt_step')
            with torch.no_grad():
                cur_depth = predict(model, up_module, cur_hazy) * init_ratio
            
            diff_depth = cur_depth*step - sum_depth
            cur_hazy = util.denormalize(cur_hazy,opt.norm)
            trans = torch.exp((diff_depth+cur_depth)*opt.betaStep*-1)
            sum_depth = cur_depth * (step+1) 
            prediction = (cur_hazy - airlight) / (trans + 1e-12) + airlight
            prediction = torch.clamp(prediction.float(),0,1)
            
            entropy, _, _ = entropy_module.get_cur(cur_hazy[0].detach().cpu().numpy().transpose(1,2,0))
            
            
            ratio = np.median(depth_images[0].detach().cpu().numpy()) / np.median(cur_depth[0].detach().cpu().numpy())
            multi_score = util.compute_errors(cur_depth[0].detach().cpu().numpy() * ratio, depth_images[0].detach().cpu().numpy())
            wr.writerow([step]+multi_score+[entropy])
            

            # ##viz haze##
            # init_haze_viz = (util.denormalize(hazy_images, opt.norm)[0].detach().cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)
            # cur_haze_viz = (cur_hazy[0].detach().cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)
            # init_clear_viz = (util.denormalize(clear_images, opt.norm)[0].detach().cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)
            # haze_set= cv2.cvtColor(np.concatenate([init_haze_viz, cur_haze_viz, init_clear_viz], axis = 0), cv2.COLOR_RGB2BGR)
            # ############

            # ##viz depth##
            # init_depth_viz = util.visualize_depth(init_depth[0])
            # cur_depth_viz = util.visualize_depth(cur_depth[0])
            # gt_depth_viz = util.visualize_depth(depth_images[0])
            # depth_set_1 = np.concatenate([init_depth_viz, cur_depth_viz, gt_depth_viz],axis=0)
            # #############

            # ##viz depth##
            # init_depth_viz = util.visualize_depth_inverse(init_depth[0])
            # cur_depth_viz = util.visualize_depth_inverse(cur_depth[0])
            # gt_depth_viz = util.visualize_depth_inverse(depth_images[0])
            # depth_set_2 = np.concatenate([init_depth_viz, cur_depth_viz, gt_depth_viz],axis=0)
            # #############

            # ##viz depth##
            # init_depth_viz = util.visualize_depth_gray(init_depth[0])
            # cur_depth_viz = util.visualize_depth_gray(cur_depth[0])
            # gt_depth_viz = util.visualize_depth_gray(depth_images[0])
            # depth_set_3 = np.concatenate([init_depth_viz, cur_depth_viz, gt_depth_viz],axis=0)
            # #############
            
            # ##viz depth##
            # init_depth_viz = util.visualize_depth_inverse_gray(init_depth[0])
            # cur_depth_viz = util.visualize_depth_inverse_gray(cur_depth[0])
            # gt_depth_viz = util.visualize_depth_inverse_gray(depth_images[0])
            # depth_set_4 = np.concatenate([init_depth_viz, cur_depth_viz, gt_depth_viz],axis=0)
            # #############


            # save_set = np.concatenate([haze_set, depth_set_1, depth_set_2, depth_set_3, depth_set_4], axis=1)
            # cv2.imwrite(f'{output_folder}/{input_names[0][:-4]}/{step:03}.jpg', save_set)
                       
            # cv2.imshow('depth', cv2.resize(save_set,(2000,1000)))
            # cv2.waitKey(0)    

            cur_hazy = util.normalize(prediction[0].detach().cpu().numpy().transpose(1,2,0).astype(np.float32),opt.norm).unsqueeze(0).to('cuda')
        #init_psnr = get_psnr(init_depth[0].detach().cpu().numpy(), depth_images[0].detach().cpu().numpy())
        #multi_psnr = get_psnr(cur_depth[0].detach().cpu().numpy(), depth_images[0].detach().cpu().numpy())
        # print(init_psnr, multi_psnr)

        f.close()
    

if __name__ == '__main__':
    opt = get_args()
    opt.norm = False
    
    # init model
    model = DenseDepth()
    model.eval()
    model.to('cuda')
    
    
    # init dataset
    if opt.dataset=='KITTI':
        weight = torch.load('weights/depth_weights/densedepth_kitti.pt')
        model.load_state_dict(weight)
        width = 1216
        height = 352
        val_set   = KITTI_Dataset(opt.dataRoot + '/val',  img_size=[width,height], norm=opt.norm)
    elif opt.dataset == 'NYU':
        weight = torch.load('weights/depth_weights/densedepth_nyu.pt')
        model.load_state_dict(weight)
        width = 640
        height = 480
    
    loader_args = dict(batch_size=1, num_workers=1, drop_last=False, shuffle=False)
    val_loader = DataLoader(dataset=val_set, **loader_args)

    airlight_module = Airlight_Module()
    entropy_module = Entropy_Module()

    run(opt, model, val_loader, airlight_module, entropy_module)
    
    