import torch
import torch.nn as nn
from collections import OrderedDict

class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        self.conv_1 = nn.Conv2d(
            in_channels=128, out_channels=out_channels, kernel_size=1
        )

        self.pyramid_3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, padding=1),
            nn.InstanceNorm2d(16, track_running_stats=False),
            nn.ReLU(True)
        )

        self.pyramid_5 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=5, padding=2),
            nn.InstanceNorm2d(16, track_running_stats=False),
            nn.ReLU(True)
        )

        self.pyramid_7 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=7, padding=3),
            nn.InstanceNorm2d(16, track_running_stats=False),
            nn.ReLU(True)
        )

        self.pyramid_11 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=11, padding=5),
            nn.InstanceNorm2d(16, track_running_stats=False),
            nn.ReLU(True)
        )

        self.pyramid_17 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=17, padding=8),
            nn.InstanceNorm2d(16, track_running_stats=False),
            nn.ReLU(True)
        )

        self.pyramid_25 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=25, padding=12),
            nn.InstanceNorm2d(16, track_running_stats=False),
            nn.ReLU(True)
        )

        self.pyramid_35 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=35, padding=17),
            nn.InstanceNorm2d(16, track_running_stats=False),
            nn.ReLU(True)
        )

        self.pyramid_45 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=45, padding=22),
            nn.InstanceNorm2d(16, track_running_stats=False),
            nn.ReLU(True)
        )

    def forward(self, x):

        #block1
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        coonv1_1 = self.conv(dec1)

        
        #block2 
        enc1_2 = self.encoder1(coonv1_1)
        enc2_2 = self.encoder2(self.pool1(enc1_2))
        enc3_2 = self.encoder3(self.pool2(enc2_2))
        enc4_2 = self.encoder4(self.pool3(enc3_2))

        bottleneck_2 = self.bottleneck(self.pool4(enc4_2))

        dec4_2 = self.upconv4(bottleneck_2)
        dec4_2 = torch.cat((dec4_2, enc4_2), dim=1)
        dec4_2 = self.decoder4(dec4_2)
        dec3_2 = self.upconv3(dec4_2)
        dec3_2 = torch.cat((dec3_2, enc3_2), dim=1)
        dec3_2 = self.decoder3(dec3_2)
        dec2_2 = self.upconv2(dec3_2)
        dec2_2 = torch.cat((dec2_2, enc2_2), dim=1)
        dec2_2 = self.decoder2(dec2_2)
        dec1_2 = self.upconv1(dec2_2)
        dec1_2 = torch.cat((dec1_2, enc1_2), dim=1)
        dec1_2 = self.decoder1(dec1_2)
        coonv1_2 = self.conv(dec1_2)


        
        #block3
        enc1_3 = self.encoder1(coonv1_2)
        enc2_3 = self.encoder2(self.pool1(enc1_3))
        enc3_3 = self.encoder3(self.pool2(enc2_3))
        enc4_3 = self.encoder4(self.pool3(enc3_3))

        bottleneck_3 = self.bottleneck(self.pool4(enc4_3))

        dec4_3 = self.upconv4(bottleneck_3)
        dec4_3 = torch.cat((dec4_3, enc4_3), dim=1)
        dec4_3 = self.decoder4(dec4_3)
        dec3_3 = self.upconv3(dec4_3)
        dec3_3 = torch.cat((dec3_3, enc3_3), dim=1)
        dec3_3 = self.decoder3(dec3_3)
        dec2_3 = self.upconv2(dec3_3)
        dec2_3 = torch.cat((dec2_3, enc2_3), dim=1)
        dec2_3 = self.decoder2(dec2_3)
        dec1_3 = self.upconv1(dec2_3)
        dec1_3 = torch.cat((dec1_3, enc1_3), dim=1)
        dec1_3 = self.decoder1(dec1_3)
        coonv1_3 = self.conv(dec1_3)


        #block4
        enc1_4 = self.encoder1(coonv1_3)
        enc2_4 = self.encoder2(self.pool1(enc1_4))
        enc3_4 = self.encoder3(self.pool2(enc2_4))
        enc4_4 = self.encoder4(self.pool3(enc3_4))

        bottleneck_4 = self.bottleneck(self.pool4(enc4_4))

        dec4_4 = self.upconv4(bottleneck_4)
        dec4_4 = torch.cat((dec4_4, enc4_4), dim=1)
        dec4_4 = self.decoder4(dec4_4)
        dec3_4 = self.upconv3(dec4_4)
        dec3_4 = torch.cat((dec3_4, enc3_4), dim=1)
        dec3_4 = self.decoder3(dec3_4)
        dec2_4 = self.upconv2(dec3_4)
        dec2_4 = torch.cat((dec2_4, enc2_4), dim=1)
        dec2_4 = self.decoder2(dec2_4)
        dec1_4 = self.upconv1(dec2_4)
        dec1_4 = torch.cat((dec1_4, enc1_4), dim=1)
        dec1_4 = self.decoder1(dec1_4)
        coonv1_4 = self.conv(dec1_4)

        #concatenation of different UNet feature maps
        concat = torch.cat((coonv1_1, coonv1_2, coonv1_3, coonv1_4  ), dim=1)


        #pyramid convolution
        conv_pyramid_3 =  self.pyramid_3(concat)
        conv_pyramid_5 =  self.pyramid_5(concat)
        conv_pyramid_7 =  self.pyramid_7(concat)
        conv_pyramid_11 =  self.pyramid_11(concat)
        conv_pyramid_17 =  self.pyramid_17(concat)
        conv_pyramid_25 =  self.pyramid_25(concat)
        conv_pyramid_35 =  self.pyramid_35(concat)
        conv_pyramid_45 =  self.pyramid_45(concat)

        #concatenation of feature maps corresponding different convolution filters present in Pyramid convolution layer
        concat_py = torch.cat(( conv_pyramid_3, conv_pyramid_5, conv_pyramid_7, conv_pyramid_11, conv_pyramid_17, conv_pyramid_25, conv_pyramid_35, conv_pyramid_45), dim=1)
      
        return torch.sigmoid(self.conv_1(concat_py))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )