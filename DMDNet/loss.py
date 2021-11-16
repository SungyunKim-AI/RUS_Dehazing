import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, dehaze, gt):
        loss = []
        dehaze_features = self.output_features(dehaze)
        gt_features = self.output_features(gt)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss.append(F.mse_loss(dehaze_feature, gt_feature))

        return sum(loss)/len(loss)

def gram_matrix(feat):
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]
    
    
class InpaintingLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.device=device
        self.extractor = VGG16FeatureExtractor().to(device)

    def forward(self, output,mask, gt):
        loss_hole = self.l1(output*(1-mask),gt*(1-mask))
        loss_valid = self.l1(output*mask,gt*mask)

        feat_output = self.extractor(output)
        feat_gt = self.extractor(gt)

        loss_style = 0.0
        for i in range(3):
            loss_style += self.l1(gram_matrix(feat_output[i]),gram_matrix(feat_gt[i]))
        
        #return 6 * loss_hole + 120 * loss_style
        return loss_hole,loss_valid, loss_style
