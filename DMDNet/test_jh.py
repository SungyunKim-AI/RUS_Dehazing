from os import kill
import monodepth.networks as networks
from torch.utils.data import DataLoader
import torch
from dataset import *
from tqdm import tqdm
import cv2

def main():
    device = torch.device('cuda')
    
    # init encoder
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load('monodepth/models/mono+stereo_1024x320/encoder.pth')
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()
    
    # init decoder
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))
    loaded_dict = torch.load('monodepth/models/mono+stereo_1024x320/depth.pth', map_location=device)
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(device)
    depth_decoder.eval()
    
    # init dataset
    val_set   = KITTI_Dataset('D:/data/KITTI' + '/val',  img_size=[1024,320], norm=False)
    loader_args = dict(batch_size=1, num_workers=1, drop_last=False, shuffle=False)
    val_loader = DataLoader(dataset=val_set, **loader_args)
    
    for batch in tqdm(val_loader):
        hazy_images, _, _, _, _, _ = batch
        with torch.no_grad():
            cur_hazy = hazy_images.to(device)
            output = depth_decoder(encoder(cur_hazy))[("disp", 0)]
        print(output.shape)
        output = output[0,0].detach().cpu().numpy()
        cv2.imshow("depth",output)
        cv2.waitKey(0)
    
    

if __name__=='__main__':
    main()