from tqdm import tqdm
import torch
from dpt.models import DPTDepthModel
from dpt.discriminator import Discriminator

from Module_Metrics.metrics import get_ssim_batch, get_psnr_batch
from util import misc, save_log, utils

def validation(opt, model, netD, dataloader):
    model.eval()
    netD.eval()
    
    for batch in tqdm(dataloader, desc="Validate"):
        csv_log = [[] for _ in range(opt.batchSize)]   
        # Data Init
        hazy_image, clear_image, airlight, GT_depth, input_name = batch
        
        clear_image_ = clear_image.clone() if opt.saveORshow == 'save' else None
        clear_image = clear_image.to(opt.device)
        airlight = airlight.to(opt.device)
        cur_hazy = hazy_image.clone().to(opt.device)
        
        # Multi-Step Depth Estimation and Dehazing
        beta = opt.betaStep
        beta_list = [0 for _ in range(opt.batchSize)]
        step_list = [0 for _ in range(opt.batchSize)]
        
        netD_psnr = torch.zeros((opt.batchSize), dtype=torch.float32)
        netD_ssim = torch.zeros((opt.batchSize), dtype=torch.float32)
        
        preds = torch.Tensor(opt.batchSize, 3, opt.imageSize_H, opt.imageSize_W).to(opt.device)
        step_flag = []
        
        for step in range(1, opt.stepLimit + 1):
            # Depth Estimation
            with torch.no_grad():
                cur_hazy = cur_hazy.to(opt.device)
                _, cur_depth = model.forward(cur_hazy)
            cur_depth = cur_depth.unsqueeze(1)
            
            # Transmission Map
            trans = torch.exp(cur_depth * -beta)
            trans = torch.add(trans, opt.eps)
            
            # Dehazing
            prediction = (cur_hazy - airlight) / trans + airlight
            prediction = torch.clamp(prediction, -1, 1)
            
            # Calculate Metrics
            with torch.no_grad():
                output = netD(prediction).view(-1).detach().cpu()
            
            for i in range(output.shape[0]):                
                if i in step_flag:
                    continue
                else:
                    if output[i] == 1:
                        preds[i] = prediction[i].clone()
                        step_flag.append(i)
                        step_list[i] = step
                    else:
                        # Discriminator가 종료 못시켰을 때
                        gt_beta = utils.get_GT_beta(input_name[i])
                        if beta_list[i] > (gt_beta + 0.1):
                            netD_psnr[i] = 0.0
                            netD_ssim[i] = 0.0
                            step_flag.append(i)
                            step_list[i] = step
            
            
            if len(step_flag) == opt.batchSize:
                netD_psnr = get_psnr_batch(preds, clear_image).detach().cpu()
                netD_ssim = get_ssim_batch(preds, clear_image).detach().cpu()
                break   # Stop Multi Step
            else:
                beta += opt.betaStep    # Set Next Step
        
        for i in range(opt.batchSize):
            if opt.save_log:
                csv_log[i].append([step_list[i], beta_list[i], netD_psnr[i], netD_ssim[i]])
                save_log.write_csv_depth_err(opt.dataRoot, input_name[i], csv_log[i])
                
            if opt.saveORshow == 'save':
                misc.results_save_tensor(opt.dataRoot, input_name[i],
                                            clear_image_[i], hazy_image[i], preds[i])
        
        if opt.verbose:
            print(f'\n netD_psnr = {netD_psnr}')
            print(f' netD_ssim = {netD_ssim}')
            
if __name__=='__main__':
    pass
    # Load saved model
    # netD = Discriminator()
    # checkpoint = torch.load(PATH)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizerD.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']
    