import os
import numpy as np
from PIL import Image

import torch
from torch.autograd import Variable
from torchvision.models import vgg16
from myutils.vgg16 import Vgg16


# Two directional gradient loss function
def gradient(y):
    gradient_h=torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])
    gradient_y=torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])
    
    return gradient_h, gradient_y


def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
	img = Image.open(filename).convert('RGB')
	if size is not None:
		if keep_asp:
			size2 = int(size * 1.0 / img.size[0] * img.size[1])
			img = img.resize((size, size2), Image.ANTIALIAS)
		else:
			img = img.resize((size, size), Image.ANTIALIAS)

	elif scale is not None:
		img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
	img = np.array(img).transpose(2, 0, 1)
	img = torch.from_numpy(img).float()
	return img


def tensor_save_rgbimage(tensor, filename, cuda=False):
	if cuda:
		img = tensor.clone().cpu().clamp(0, 255).numpy()
	else:
		img = tensor.clone().clamp(0, 255).numpy()
	img = img.transpose(1, 2, 0).astype('uint8')
	img = Image.fromarray(img)
	img.save(filename)


def tensor_save_bgrimage(tensor, filename, cuda=False):
	(b, g, r) = torch.chunk(tensor, 3)
	tensor = torch.cat((r, g, b))
	tensor_save_rgbimage(tensor, filename, cuda)


def gram_matrix(y):
	(b, ch, h, w) = y.size()
	features = y.view(b, ch, w * h)
	features_t = features.transpose(1, 2)
	gram = features.bmm(features_t) / (ch * h * w)
	return gram


def subtract_imagenet_mean_batch(batch):
	"""Subtract ImageNet mean pixel-wise from a BGR image."""
	tensortype = type(batch.data)
	mean = tensortype(batch.data.size())
	mean[:, 0, :, :] = 103.939
	mean[:, 1, :, :] = 116.779
	mean[:, 2, :, :] = 123.680
	return batch - Variable(mean)


def add_imagenet_mean_batch(batch):
	"""Add ImageNet mean pixel-wise from a BGR image."""
	tensortype = type(batch.data)
	mean = tensortype(batch.data.size())
	mean[:, 0, :, :] = 103.939
	mean[:, 1, :, :] = 116.779
	mean[:, 2, :, :] = 123.680
	return batch + Variable(mean)

def imagenet_clamp_batch(batch, low, high):
	batch[:,0,:,:].data.clamp_(low-103.939, high-103.939)
	batch[:,1,:,:].data.clamp_(low-116.779, high-116.779)
	batch[:,2,:,:].data.clamp_(low-123.680, high-123.680)


def preprocess_batch(batch):
	batch = batch.transpose(0, 1)
	(r, g, b) = torch.chunk(batch, 3)
	batch = torch.cat((b, g, r))
	batch = batch.transpose(0, 1)
	return batch


def init_vgg16(vgg, model_folder):
    if not os.path.exists(os.path.join(model_folder, 'vgg16.weight')):
        pretrained_vgg = vgg16(pretrained=True)
        
        conv_list = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']
        fetures_list = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
        for flag in ['weight', 'bias']:
            for i in range(13):
                vgg_param = getattr(getattr(getattr(vgg, conv_list[i]), flag), "data")
                pretrained_vgg_param = getattr(getattr(getattr(pretrained_vgg, "features")[fetures_list[i]], flag), "data")
                vgg_param.copy_(pretrained_vgg_param)
                
        torch.save(vgg.state_dict(), os.path.join(model_folder, 'vgg16.weight'))	 