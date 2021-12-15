import torch
import torch.nn as nn

def blockUNet(in_c, out_c, transposed=False, bn=False, relu=True, dropout=False):
    block = nn.Sequential()
    if relu:
        block.add_module('relu', nn.ReLU(inplace=True))
    else:
        block.add_module('leakyrelu', nn.LeakyReLU(0.2, inplace=True))
    if not transposed:
        block.add_module('conv', nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
    else:
        block.add_module('t_conv', nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))
    if bn:
        block.add_module('bn', nn.BatchNorm2d(out_c))
    if dropout:
        block.add_module('dropout', nn.Dropout2d(0.5, inplace=True))
    
    return block

class Air_UNet(nn.Module):
    def __init__(self, input_nc, output_nc, nf):
        super(Air_UNet, self).__init__()
        # input is 256 x 256
        self.layer1 = nn.Conv2d(input_nc, nf, 4, 2, 1, bias=False)
        # input is 128 x 128
        self.layer2 = blockUNet(nf, nf*2, transposed=False, bn=True, relu=False, dropout=False)
        # input is 64 x 64
        self.layer3 = blockUNet(nf*2, nf*4, transposed=False, bn=True, relu=False, dropout=False)
        # input is 32
        self.layer4 = blockUNet(nf*4, nf*8, transposed=False, bn=True, relu=False, dropout=False)
        # input is 16
        self.layer5 = blockUNet(nf*8, nf*8, transposed=False, bn=True, relu=False, dropout=False)
        # input is 8
        self.layer6 = blockUNet(nf*8, nf*8, transposed=False, bn=True, relu=False, dropout=False)
        # input is 4
        self.layer7 = blockUNet(nf*8, nf*8, transposed=False, bn=True, relu=False, dropout=False)
        # input is 2 x  2
        self.layer8 = blockUNet(nf*8, nf*8, transposed=False, bn=True, relu=False, dropout=False)

        ## NOTE: decoder
        # input is 1
        d_inc = nf*8
        self.dlayer8 = blockUNet(d_inc, nf*8, transposed=True, bn=False, relu=True, dropout=True)
        # input is 2
        d_inc = nf*8*2
        self.dlayer7 = blockUNet(d_inc, nf*8, transposed=True, bn=True, relu=True, dropout=True)
        # input is 4
        d_inc = nf*8*2
        self.dlayer6 = blockUNet(d_inc, nf*8, transposed=True, bn=True, relu=True, dropout=True)
        # input is 8
        d_inc = nf*8*2
        self.dlayer5 = blockUNet(d_inc, nf*8, transposed=True, bn=True, relu=True, dropout=False)
        # input is 16
        d_inc = nf*8*2
        self.dlayer4 = blockUNet(d_inc, nf*4, transposed=True, bn=True, relu=True, dropout=False)
        # input is 32
        d_inc = nf*4*2
        self.dlayer3 = blockUNet(d_inc, nf*2, transposed=True, bn=True, relu=True, dropout=False)
        # input is 64
        d_inc = nf*2*2
        self.dlayer2 = blockUNet(d_inc, nf, transposed=True, bn=True, relu=True, dropout=False)
        # input is 128
        d_inc = nf*2
        self.dlayer1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d_inc, output_nc, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.pooling = nn.AdaptiveAvgPool2d((1,1))
    
    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        
        dout8 = self.dlayer8(out8)
        dout8_out7 = torch.cat([dout8, out7], 1)
        dout7 = self.dlayer7(dout8_out7)
        dout7_out6 = torch.cat([dout7, out6], 1)
        dout6 = self.dlayer6(dout7_out6)
        dout6_out5 = torch.cat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2 = torch.cat([dout3, out2], 1)
        dout2 = self.dlayer2(dout3_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)
        
        # out = self.pooling(dout1)
        
        return dout1