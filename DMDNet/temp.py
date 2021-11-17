def test_with_D_Hazy_NYU_notation(model, test_loader, device):
    entropy_module = Entropy_Module()
    stage = 100
    model.eval()
    psnr_sum, ssim_sum = 0.0, 0.0
    step_beta = 0.01
    
    a = 0.0012
    b = 3.7
    
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
