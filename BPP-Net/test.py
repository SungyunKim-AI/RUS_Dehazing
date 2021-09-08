import glob
import cv2
import numpy as np
import PIL
from PIL import Image

from models import *
from metrics import *
import torchvision.transforms.functional as F


def to_tensor(img):
    img_t = F.to_tensor(img).float()
    return img_t

def postprocess(img):
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()
    
path_of_test_hazy_images = 'test/haze/*.png'
path_for_resultant_dehaze_images = 'test/result/'
image_paths_test_hazy=glob.glob(path_of_test_hazy_images)

for i in range(len(image_paths_test_hazy)):
    haze_image = cv2.imread(image_paths_test_hazy[i])
    haze_image = Image.fromarray(haze_image)
    haze_image = haze_image.resize((512,512), resample=PIL.Image.BICUBIC)
    haze_image = np.array(haze_image)
    haze_image = cv2.cvtColor(haze_image, cv2.COLOR_BGR2YCrCb)
    haze_image = to_tensor(haze_image).cuda()
    haze_image = haze_image.reshape(1,3,512,512)

    dehaze_image = DU_Net.predict(haze_image) 
    
    dehaze_image = postprocess(dehaze_image)[0]
    dehaze_image = dehaze_image.cpu().detach().numpy()
    dehaze_image = dehaze_image.astype('uint8')
    dehaze_image = dehaze_image.reshape(512,512,3)
    dehaze_image = cv2.cvtColor(dehaze_image, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(path_for_resultant_dehaze_images+str(i+50)+'.png', dehaze_image)
    
    
ssim = SSIM(window_size = 11)
psnr = PSNR()
psnr_val = 0
psnr_val = 0.0
final_ssim = 0
final_psnr = 0

path_of_test_hazy_images = 'test/haze/*.png'
path_of_test_gt_images = 'test/gt/*.png'
path_for_resultant_dehaze_images = 'test/result/'

image_paths_test_hazy=glob.glob(path_of_test_hazy_images)
image_paths_test_gt=glob.glob(path_of_test_gt_images)

for i in range(len(image_paths_test_hazy)):
    im1 = cv2.imread(image_paths_GT[i])
    im1 = Image.fromarray(im1)
    im1 = im1.resize((512,512), resample=PIL.Image.BICUBIC)
    im1 = np.array(im1)
    im2 = cv2.imread('Results/experiment yrcrcb/new/outdoor/' + str(i+50)+'.png')

    im1 = to_tensor(im1).reshape(1,3,512,512) 
    im2 = to_tensor(im2).reshape(1,3,512,512)
    
    psnr_val = psnr(im1, im2)
    final_psnr = final_psnr + 10*np.log10((psnr_val))
    final_ssim = final_ssim + ssim(im1, im2)

print(final_ssim/5.0, final_psnr/5.0)
