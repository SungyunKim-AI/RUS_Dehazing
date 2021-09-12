import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
from torchinfo import summary

class SoftConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, padding_mode='zeros', bias='True', isPartial=False, soft=0.3, activation='leaky_relu'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.isPartial = isPartial
        self.soft = soft
        
        self.input_conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride, padding=padding, padding_mode=padding_mode, bias=bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels,kernel_size, stride, padding=padding, padding_mode=padding_mode, bias=False)
        
        
        nn.init.kaiming_normal_(self.input_conv.weight, a=0, mode='fan_in',nonlinearity=activation)
        nn.init.constant_(self.mask_conv.weight, 1.0)

        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input_img, input_mask):
        slide_winsize = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        mask_ratio = None
        output_mask = None

        with torch.no_grad():
            output_mask = self.mask_conv(input_mask)
            mask_ratio = slide_winsize / (output_mask.masked_fill(output_mask == 0,1.0))
            #mask_ratio = slide_winsize/(output_mask+1e-8)
            
            if self.isPartial:
                output_mask = torch.clamp(output_mask, 0, 1)
            else:
                output_mask =torch.clamp((output_mask/self.soft)//slide_winsize,0,1)
            mask_ratio = output_mask * mask_ratio

        output = self.input_conv(input_img * input_mask)
        if self.bias:
            output_bias = self.input_conv.bias.view(1, self.out_channels, 1, 1)
            output = (output - output_bias) * mask_ratio + output_bias
        else:
            output = output * mask_ratio
        return output, output_mask
    
class SCBlock(nn.Module):
    def __init__(self, in_ch, out_ch,isPartial,soft,bn,activation):
        super().__init__()
        self.sconv1 = SoftConv2d(in_ch, out_ch, [3,3], padding=[1,1], bias=False,isPartial=isPartial,soft=soft,activation=activation)
        self.bn1=None
        self.bn2=None
        if bn:
            self.bn1 = nn.BatchNorm2d(out_ch)
            self.bn2 = nn.BatchNorm2d(out_ch)
        
        self.actfun=None
        if activation=='relu':
            self.actfun = nn.ReLU()
        elif activation == 'leaky_relu':
            self.actfun = nn.LeakyReLU()
        
        self.sconv2 = SoftConv2d(out_ch, out_ch, [3,3], padding=[1,1], bias=False,isPartial=isPartial,soft=soft,activation=activation)
    def forward(self, x, x_mask):
        x, x_mask = self.sconv1(x,x_mask)
        if self.bn1:
            x = self.bn1(x)
        x = self.actfun(x)
        x, x_mask = self.sconv2(x,x_mask)
        if self.bn2:
            x = self.bn2(x)
        x = self.actfun(x)
        return x,x_mask

class InputSCBlock(nn.Module): #3-frame 
    def __init__(self, out_ch,isPartial,soft,bn,activation):
        super().__init__()
        self.sconv1 = SoftConv2d(3,30,[3,3],padding=[1,1],bias=False,isPartial=isPartial,soft=soft,activation=activation)
        self.bn1=None
        self.bn2=None
        if bn:
            self.bn1 = nn.BatchNorm2d(3 * 30)
            self.bn2 = nn.BatchNorm2d(out_ch)
        
        self.actfun=None
        if activation=='relu':
            self.actfun = nn.ReLU()
        elif activation=='leaky_relu':
            self.actfun = nn.LeakyReLU()
        
        self.sconv2 = SoftConv2d(3*30,out_ch,[3,3],padding=[1,1],bias=False,isPartial=isPartial,soft=soft,activation=activation)
    def forward(self,x,x_mask): # x:(B,T=3,C=3,W,H)
        x0,x0_mask = self.sconv1(x[:,0,:,:,:],x_mask[:,0,:,:,:])
        x1,x1_mask = self.sconv1(x[:,1,:,:,:],x_mask[:,1,:,:,:])
        x2,x2_mask = self.sconv1(x[:,2,:,:,:],x_mask[:,2,:,:,:])

        x = torch.cat([x0,x1,x2],dim=1)
        x_mask = torch.cat([x0_mask,x1_mask,x2_mask],dim=1)
        # x:(B,C=90,W,H)
        if self.bn1:
            x=self.bn1(x)
        x=self.actfun(x)
        x,x_mask = self.sconv2(x,x_mask)
        if self.bn2:
            x=self.bn2(x)
        x=self.actfun(x)
        return x,x_mask

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch,isPartial,soft,bn,activation):
        super().__init__()
        self.sconv=SoftConv2d(in_ch,out_ch,[3,3],[2,2],[1,1],bias=False,isPartial=isPartial,soft=soft,activation=activation)
        
        self.bn=None
        if bn:
            self.bn=nn.BatchNorm2d(out_ch)
        
        self.actfun=None
        if activation=='relu':
            self.actfun=nn.ReLU()
        elif activation=='leaky_relu':
            self.actfun=nn.LeakyReLU()
        self.scblock=SCBlock(out_ch,out_ch,isPartial,soft,bn,activation)

    def forward(self,x,x_mask):
        x,x_mask = self.sconv(x,x_mask)
        if self.bn:
            x = self.bn(x)
        x = self.actfun(x)
        x,x_mask = self.scblock(x,x_mask)
        return x,x_mask

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch,isPartial,soft,bn,activation):
        super().__init__()
        self.scblock=SCBlock(in_ch,in_ch,isPartial,soft,bn,activation)
        self.sconv=SoftConv2d(in_ch,out_ch*4,[3,3],padding=[1,1],bias=False,isPartial=isPartial,soft=soft,activation=activation)
        self.ps=nn.PixelShuffle(2)

    def forward(self,x,x_mask):
        x,x_mask = self.scblock(x,x_mask)
        x,x_mask = self.sconv(x,x_mask)
        x = self.ps(x)
        x_mask = self.ps(x_mask)
        return x,x_mask

class OutputSCBlock(nn.Module):
    def __init__(self,in_ch,out_ch,isPartial,soft,bn,activation):
        super().__init__()
        self.sconv1=SoftConv2d(in_ch,in_ch,[3,3],padding=[1,1],bias=False,isPartial=isPartial,soft=soft,activation=activation)
        self.bn=None
        if bn:
            self.bn=nn.BatchNorm2d(in_ch)
            
        self.actfun=None
        if activation=='relu':
            self.actfun=nn.ReLU()
        elif activation=='leaky_relu':
            self.actfun=nn.LeakyReLU()
            
        self.sconv2=SoftConv2d(in_ch,out_ch,[3,3],padding=[1,1],bias=False,isPartial=isPartial,soft=soft,activation=activation)

    def forward(self,x,x_mask):
        x,x_mask=self.sconv1(x,x_mask)
        if self.bn:
            x = self.bn(x)
        x = self.actfun(x)
        x,x_mask=self.sconv2(x,x_mask)
        return x,x_mask

class InpBlock(nn.Module):
    def __init__(self,isPartial,soft,bn,activation):
        super().__init__()
        self.inb=InputSCBlock(32,isPartial,soft,bn,activation)
        self.downb0=DownBlock(32,64,isPartial,soft,bn,activation)
        self.downb1=DownBlock(64,128,isPartial,soft,bn,activation)
        self.downb2=DownBlock(128,256,isPartial,soft,bn,activation)
        self.downb3=DownBlock(256,512,isPartial,soft,bn,activation)
        self.downb4=DownBlock(512,1024,isPartial,soft,bn,activation)
        self.upb5=UpBlock(1024,512,isPartial,soft,bn,activation)
        self.upb4=UpBlock(512,256,isPartial,soft,bn,activation)
        self.upb3=UpBlock(256,128,isPartial,soft,bn,activation)
        self.upb2=UpBlock(128,64,isPartial,soft,bn,activation)
        self.upb1=UpBlock(64,32,isPartial,soft,bn,activation)
        self.outb=OutputSCBlock(32,3,isPartial,soft,bn,activation)

    def forward(self,x,x_mask):# x:(B,T=3,C=3,W,H)
        x0, x0_mask = self.inb(x,x_mask)
        x1, x1_mask = self.downb0(x0,x0_mask)
        x2, x2_mask = self.downb1(x1,x1_mask)
        x3, x3_mask = self.downb2(x2,x2_mask)
        x4, x4_mask = self.downb3(x3,x3_mask)
        x5, x5_mask = self.downb4(x4,x4_mask)

        x5, x5_mask = self.upb5(x5,x5_mask)
        x45_mask = x4_mask+x5_mask
        x4, x4_mask = self.upb4((x4*x4_mask + x5*x5_mask)/x45_mask.masked_fill(x45_mask==0,1.0),(x4_mask+x5_mask >= 1).to(torch.float))
        x34_mask = x3_mask+x4_mask
        x3, x3_mask = self.upb3((x3*x3_mask + x4*x4_mask)/x34_mask.masked_fill(x34_mask==0,1.0),(x3_mask+x4_mask >= 1).to(torch.float))
        x23_mask = x2_mask+x3_mask
        x2, x2_mask = self.upb2((x2*x2_mask + x3*x3_mask)/x23_mask.masked_fill(x23_mask==0,1.0),(x2_mask+x3_mask >= 1).to(torch.float))
        x12_mask = x1_mask+x2_mask
        x1, x1_mask = self.upb1((x1*x1_mask + x2*x2_mask)/x12_mask.masked_fill(x12_mask==0,1.0),(x1_mask+x2_mask >= 1).to(torch.float))
        x01_mask = x0_mask+x1_mask
        xx, xx_mask=self.outb((x0*x0_mask+x1*x1_mask)/x01_mask.masked_fill(x01_mask==0,1.0),(x0_mask+x1_mask >=1).to(torch.float))
        
        x_mask = x_mask[:,2,:,:,:]
        x = x[:,2,:,:,:]
        
        xxx_mask = x_mask+xx_mask
        xx = (x * x_mask + xx * xx_mask)/xxx_mask.masked_fill(xxx_mask==0,1.0)
        xx_mask = (x_mask + xx_mask >=1).to(torch.float)

        return xx, xx_mask

class FastDVDnet(nn.Module):
    def __init__(self,isPartial=False,soft=0.1,bn=True,activation='relu'):
        super().__init__()
        self.iblock1=InpBlock(isPartial,soft,bn,activation)
        self.iblock2=InpBlock(isPartial,soft,bn,activation)
        
        
        self.sconv1= SoftConv2d(3,12,[3,3],padding=[1,1],isPartial=isPartial,soft=soft,activation=activation)
        self.sconv2= SoftConv2d(12,3,[3,3],padding=[1,1],isPartial=isPartial,soft=soft,activation=activation)
        
        """"
        self.sconv1=SoftConv2d(3,8,[3,3],padding=[1,1],isPartial=isPartial,soft=soft,activation=activation)
        self.sconv2=SoftConv2d(8,16,[3,3],padding=[1,1],isPartial=isPartial,soft=soft,activation=activation)
        self.sconv3=SoftConv2d(16,8,[3,3],padding=[1,1],isPartial=isPartial,soft=soft,activation=activation)
        self.sconv4=SoftConv2d(8,3,[3,3],padding=[1,1],isPartial=isPartial,soft=soft,activation=activation)
        """
        
        self.actfun=None
        if activation=='relu':
            self.actfun=nn.ReLU()
        if activation=='leaky_relu':
            self.actfun=nn.LeakyReLU()
    
    def forward(self,x,x_mask): # x:(B,T,C,W,H)
        
        (x02,x13,x24) = tuple(x[:,m:m+3,:,:,:] for m in range(3))
        (x02_mask,x13_mask,x24_mask) = tuple(x_mask[:,m:m+3,:,:,:] for m in range(3))
        x02,x02_mask = self.iblock1(x02,x02_mask)
        x13,x13_mask = self.iblock1(x13,x13_mask)
        x24,x24_mask = self.iblock1(x24,x24_mask)


        out = torch.stack([x02,x13,x24]).transpose(0,1)
        out_mask = torch.stack([x02_mask,x13_mask,x24_mask]).transpose(0,1)
        out,out_mask = self.iblock2(out,out_mask)
        
        out, out_mask = self.sconv1(out,out_mask)
        out, out_mask = self.sconv2(out,out_mask)
        
        """
        x_mask = x_mask[:,2,:,:,:]
        x = x[:,2,:,:,:]
        

        out = x*x_mask + out*(1-x_mask)
        
        out,out_mask = self.sconv1(out,out_mask)
        out = self.actfun(out)
        
        out,out_mask = self.sconv2(out,out_mask)
        out = self.actfun(out)
        
        out,out_mask = self.sconv3(out,out_mask)
        out = self.actfun(out)
        
        out,out_mask = self.sconv4(out,out_mask)
        out = self.actfun(out)
        
        out = x * x_mask + out * (1-x_mask)
        """
        
        
        return out,out_mask
        



class SoftConv3d(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, padding, padding_mode, soft):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size =kernel_size
        self.padding = padding
        self.padding_mode = padding_mode
        self.soft = soft

        self.input_conv = nn.Conv3d(self.in_channels, self.out_channels, self.kernel_size, padding=self.padding, padding_mode=self.padding_mode )
        self.mask_conv = nn.Conv3d(self.in_channels, self.out_channels, self.kernel_size, padding=self.padding, padding_mode=self.padding_mode, bias=False )

        nn.init.kaiming_normal_(self.input_conv.weight, a=0, mode='fan_in')
        nn.init.constant_(self.mask_conv.weight, 1.0)

        for param in self.mask_conv.parameters():
            param.requires_grad = False

        
    def forward(self, input_img, input_mask):
        slide_winsize = self.in_channels * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        mask_ratio =None
        output_mask = None

        with torch.no_grad():
            output_mask = self.mask_conv(input_mask)
            mask_ratio = slide_winsize/(output_mask+1e-8)
            output_mask =torch.clamp((output_mask/self.soft)//slide_winsize,0,1)
            mask_ratio = output_mask * mask_ratio

        output = self.input_conv(input_img * input_mask)
        output_bias = self.input_conv.bias.view(1, self.out_channels, 1, 1, 1)
        output = (output - output_bias) * mask_ratio + output_bias

        return output, output_mask

class SConvNet(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, padding=0, padding_mode='zeros',soft=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.padding_mode = padding_mode
        self.soft = soft


        self.conv1 = SoftConv3d(self.input_dim, self.input_dim*2, self.kernel_size, self.padding, self.padding_mode,self.soft)
        self.conv2 = SoftConv3d(self.input_dim*2, self.input_dim*4, self.kernel_size, self.padding, self.padding_mode,self.soft)
        self.conv3 = SoftConv3d(self.input_dim*4, self.input_dim*8, self.kernel_size, self.padding, self.padding_mode,self.soft)
        self.conv4 = SoftConv3d(self.input_dim*8, self.input_dim*8, self.kernel_size, self.padding, self.padding_mode,self.soft)
        self.conv5 = SoftConv3d(self.input_dim*8, self.input_dim*8, self.kernel_size, self.padding, self.padding_mode,self.soft)
        self.conv6 = SoftConv3d(self.input_dim*8, self.input_dim*8, self.kernel_size, self.padding, self.padding_mode,self.soft)
        self.conv7 = SoftConv3d(self.input_dim*8, self.input_dim*4, self.kernel_size, self.padding, self.padding_mode,self.soft)
        self.conv8 = SoftConv3d(self.input_dim*4, self.input_dim*2, [3,3,3], self.padding, self.padding_mode,self.soft)
        self.conv9 = SoftConv3d(self.input_dim*2, self.output_dim, [3,3,3], self.padding, self.padding_mode,self.soft)


        self.activation_ = nn.LeakyReLU()

    def forward(self, input_tensor, input_mask):
        x,x_mask = self.conv1(input_tensor, input_mask)
        x = self.activation_(x)
        x,x_mask = self.conv2(x,x_mask)
        x = self.activation_(x)
        x,x_mask = self.conv3(x,x_mask)
        x = self.activation_(x)
        x,x_mask = self.conv4(x,x_mask)
        x = self.activation_(x)
        x,x_mask = self.conv5(x,x_mask)
        x = self.activation_(x)
        x,x_mask = self.conv6(x,x_mask)
        x = self.activation_(x)
        x,x_mask = self.conv7(x,x_mask)
        x = self.activation_(x)
        x,x_mask = self.conv8(x,x_mask)
        x = self.activation_(x)
        x,x_mask = self.conv9(x,x_mask)


        return x,x_mask

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

if __name__ == '__main__':
    print("main in net.py")
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    model = FastDVDnet(True,soft=0.1,bn=False,activation='leaky_relu').to(device)
    #summary(model,[[1,5,3,256,256],[1,5,3,256,256]])
    input_img = torch.rand([1,5,3,256,256]).to(device)
    input_mask = torch.rand([1,5,3,256,256]).to(device)
    
    output_img, output_mask= model(input_img,input_mask)
    print(output_img.shape)
    print(output_mask.shape)
    """
    
    sconv = SoftConv2d(3,3,[3,3],isPartial=False,padding=1,soft=0.4)
    input_img = torch.rand([1,3,128,128])
    
    from PIL import Image
    from torchvision import transforms
    import matplotlib.pyplot as plt
    
    input_mask = Image.open('data/mask/00160.png')
    mask_transform = transforms.Compose(
        [transforms.Resize(size=[128,128]), 
        transforms.ToTensor()]
    )
    input_mask = mask_transform(input_mask.convert('RGB'))
    input_mask = 1-input_mask
    input_mask = input_mask//1
    input_mask = input_mask.unsqueeze(0)
    
    
    fig = plt.figure()
    cols = 420
    j=2
    ax = fig.add_subplot(2,4,1)
    ax.imshow(input_mask.squeeze(0).transpose(0,1).transpose(1,2))
    ax.set_title(0)
    
    out, out_mask = sconv(input_img,input_mask)
    for i in range(cols):
        if (i+1)%60 != 0:
            continue
        ax = fig.add_subplot(2,4,j)
        j+=1
        ax.imshow(out_mask.squeeze(0).transpose(0,1).transpose(1,2))
        out, out_mask = sconv(out,out_mask)
        ax.set_title(f'{i+1}')
    #fig.tight_layout()
    plt.show()
    
    #plt.imshow(out_mask.squeeze(0).transpose(0,1).transpose(1,2))
    #plt.show()