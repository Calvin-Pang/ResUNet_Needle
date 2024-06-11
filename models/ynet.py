import torch
import torch.nn as nn
import math

class ResConvBlock(nn.Module):
    '''
    Basic residual convolutional block
    '''
    def __init__(self, in_channels, out_channels, res = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.res = res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        if self.res:
            if self.in_channels == self.out_channels:
                out = x + x2 
            else:
                out = x1 + x2
            return out / math.sqrt(2)
        else:
            return x2


class UnetDown(nn.Module):
    '''
    UNet down block (encoding)
    '''
    def __init__(self, in_channels, out_channels, pooling = True, res = True):
        super(UnetDown, self).__init__()
        layers = [
            ResConvBlock(in_channels, out_channels, res = res), 
            ResConvBlock(out_channels, out_channels, res = res)
        ]
        if pooling: layers.append(nn.MaxPool2d(2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    '''
    UNet up block (decoding)
    '''    
    def __init__(self, in_channels, out_channels, res = True):
        super(UnetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResConvBlock(out_channels, out_channels, res = res),
            ResConvBlock(out_channels, out_channels, res = res),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class ResYNet(nn.Module):
    def __init__(self, seg, cls, embed_dim = 64, in_channels = 1, num_classes = 1, res = True):
        super(ResYNet, self).__init__()

        self.in_channels = in_channels
        self.seg = seg
        self.cls = cls
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # down path for encoding
        self.downblock1 = UnetDown(in_channels, embed_dim, res = res)
        self.downblock2 = UnetDown(embed_dim, 2 * embed_dim, res = res)
        self.downblock3 = UnetDown(2 * embed_dim, 4 * embed_dim, res = res)
        self.downblock4 = UnetDown(4 * embed_dim, 8 * embed_dim, pooling = False, res = res)


        # up path for decoding
        self.upblock1 = UnetUp(12 * embed_dim, 4 * embed_dim, res = res)
        self.upblock2 = UnetUp(6 * embed_dim, 2 * embed_dim, res = res)
        self.upblock3 = UnetUp(3 * embed_dim, embed_dim, res = res)
        self.outblock = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, self.num_classes, 3, 1, 1),
            nn.Sigmoid()
        )

        if self.seg and self.cls:
            self.cls_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size = 1),
                nn.Flatten(),
                nn.Linear(embed_dim * 8, self.num_classes),
                nn.Sigmoid()
            )


    def forward(self, x):
        down1 = self.downblock1(x) # B, 64, 256, 256
        down2 = self.downblock2(down1) # B, 128, 128, 128
        down3 = self.downblock3(down2) # B, 256, 64, 64
        feature_map = self.downblock4(down3) # B, 512, 64, 64
        
        up1 = self.upblock1(feature_map, down3)
        up2 = self.upblock2(up1, down2) # 128
        up3 = self.upblock3(up2, down1) # 128

        x_seg = self.outblock(up3)

        if self.seg and not self.cls:
            return x_seg

        if self.seg and self.cls:
            x_cls = self.cls_head(feature_map)
            return x_seg, x_cls