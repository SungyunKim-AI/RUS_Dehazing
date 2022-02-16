import cv2
import math
import os
import csv
import numpy as np;
from glob import glob
from metrics import psnr
from tqdm import tqdm
import torch
from entropy_module import Entropy_Module

def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz);
    imvec = im.reshape(imsz,3);

    indices = darkvec.argsort();
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A

def TransmissionEstimate(im,A,sz):
    omega = 0.95;
    im3 = np.empty(im.shape,im.dtype);

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz);
    return transmission

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r));
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r));
    cov_Ip = mean_Ip - mean_I*mean_p;

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r));
    var_I   = mean_II - mean_I*mean_I;

    a = cov_Ip/(var_I + eps);
    b = mean_p - a*mean_I;

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r));

    q = mean_a*im + mean_b;
    return q;

def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray)/255;
    r = 60;
    eps = 0.0001;
    t = Guidedfilter(gray,et,r,eps);

    return t;

def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype);
    t = cv2.max(t,tx);

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res

if __name__ == '__main__':
    
    hazy_imgs = glob('D:/data/RESIDE_V0_outdoor/val/hazy/*/*.jpg')
    clear_img_dir = 'D:/data/RESIDE_V0_outdoor/val/clear'
    
    output_dir = 'D:/data/output_dehaze/pred_DCP_SOTS'
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    f = open('D:/data/output_dehaze/pred_DCP_SOTS/DCP_SOTS.csv', 'w', newline='')
    wr = csv.writer(f)
    
    entropy_module = Entropy_Module()
    
    for hazy_img in tqdm(hazy_imgs):
        file_name = os.path.basename(hazy_img);
        token = file_name.split('_')
        clear_img = clear_img_dir+'/'+token[0]+'.png'
        beta = token[-1][:-4]

        haze = cv2.imread(hazy_img)
        I = haze.astype(np.float64)/255
        clear = cv2.imread(clear_img).astype(np.float64)/255

    
        dark = DarkChannel(I,15);
        A = AtmLight(I,dark);
        te = TransmissionEstimate(I,A,15);
        t = TransmissionRefine(haze,te);
        J = Recover(I,t,A,0.1);

        pred_psnr = psnr(torch.Tensor(J), torch.Tensor(clear))
        pred_entropy, _, _ = entropy_module.get_cur(J)
        
        wr.writerow([file_name, beta, pred_psnr, pred_entropy])
        
        cv2.imwrite(output_dir+'/'+file_name, (np.clip(J,0,1)*255).astype(np.uint8))
        # cv2.imshow("pred", J)
        # cv2.imshow("clear", clear)
        # cv2.waitKey();
        
    f.close()
        
