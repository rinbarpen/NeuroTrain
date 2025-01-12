# TODO: To test shared_attention
import torch
import math
from torch import nn
import torch.functional as F
from models.attention.fused_attention import CBAM
from models.conv.DWConv import get_dwconv_layer2d


def auto_pad(kernel_size, stride=1, dilation=1):
    padding = ((kernel_size - 1) * dilation + 1 - stride) / 2
    return math.ceil(padding)


# use to neck, bottleneck
class InceptionAttnConv(nn.Module):
    def __init__(self, n_channels):
        super(InceptionAttnConv, self).__init__()

        self.branch1 = nn.Sequential(
            get_dwconv_layer2d(n_channels, n_channels, 3, 1),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=3, dilation=2, padding=2),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            CBAM(n_channels, reduction=16),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
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

