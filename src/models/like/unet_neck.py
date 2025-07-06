import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from ..attention.CBAM import CBAM
from ..conv.DWConv import get_dwconv_layer2d

def auto_pad(kernel_size, stride=1, dilation=1):
    padding = ((kernel_size - 1) * dilation + 1 - stride) / 2
    return math.ceil(padding)


# use to neck, bottleneck
class InceptionAttnConv(nn.Module):
    def __init__(self, n_channels):
        super(InceptionAttnConv, self).__init__()

        self.branch1 = nn.Sequential(
            get_dwconv_layer2d(n_channels, n_channels, kernel_size=3, stride=1, bias=True),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=3, dilation=2, padding=2),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
            CBAM(n_channels, reduction=16),
        )
        self.bn1 = nn.BatchNorm2d(4 * n_channels)
        self.back = nn.Sequential(
            nn.Conv2d(4 * n_channels, n_channels, kernel_size=3, padding=1),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
        )
        self.bn2 = nn.BatchNorm2d(n_channels)

    def forward(self, x):
        y1 = self.branch1(x)
        y2 = self.branch2(y1)
        y3 = self.branch3(y2)
        y = torch.cat([x, y1, y2, y3], dim=1)
        x = self.bn1(y)
        x = self.back(x)
        x = self.bn2(x)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Bottleneck(nn.Module):
    def __init__(self, n_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            InceptionAttnConv(n_channels),
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # torch.div(dim_t, 2, rounding_mode='trunc')
        x1 = F.pad(x1, [torch.div(diffX, 2, rounding_mode='trunc'),
                        torch.div(diffX - diffX, 2, rounding_mode='trunc'),
                        torch.div(diffY, 2, rounding_mode='trunc'),
                        torch.div(diffY - diffY, 2, rounding_mode='trunc')])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Bottleneck(512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
