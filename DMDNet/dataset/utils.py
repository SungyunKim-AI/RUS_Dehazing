import cv2
import torchvision.transforms.functional as F
from torchvision.transforms import Compose
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
from util import io

def to_tensor(img):
    img_t = F.to_tensor(img).float()
    return img_t
    
def load_item(haze, clear, img_size):
        hazy_image = cv2.imread(haze)
        hazy_image = cv2.resize(hazy_image,img_size)/255
        
        clear_image = cv2.imread(clear)
        clear_image = cv2.resize(clear_image,img_size)/255
        
        hazy_resize = to_tensor(hazy_image)
        clear_resize = to_tensor(clear_image)
        
        return hazy_resize, clear_resize
    
def load_item_2(haze, clear, transform):
    haze = io.read_image(haze)
    clear = io.read_image(clear)
    
    haze_input  = transform({"image": haze})["image"]
    clear_input = transform({"image": clear})["image"]

    return haze_input, clear_input

def load_item_3(haze, clear, airlight, transform):
    haze = io.read_image(haze)
    clear = io.read_image(clear)
    airlight = io.read_image(airlight)
    
    haze_input  = transform({"image": haze})["image"]
    clear_input = transform({"image": clear})["image"]
    airlight_input = transform({"image": airlight})["image"]

    return haze_input, clear_input, airlight_input

def make_transform(img_size, norm=False, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    
    resize = Resize(img_size[0],
                    img_size[1],
                    resize_target=None,
                    keep_aspect_ratio=False,
                    ensure_multiple_of=32,
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC)
    
    # resize = Resize(img_size[0],
    #                 img_size[1],
    #                 resize_target=None,
    #                 keep_aspect_ratio=False,
    #                 ensure_multiple_of=32,
    #                 resize_method="",
    #                 image_interpolation_method=cv2.INTER_AREA)
    
    if norm == True:
        transform = Compose([resize, 
                             NormalizeImage(mean=mean, std=std),
                             PrepareForNet()])
    else:
        transform = Compose([resize,
                             PrepareForNet()])

    return transform
