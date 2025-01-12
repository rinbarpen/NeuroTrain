"""
    Rotate to Attend: Convolutional Triplet Attention Module
    https://arxiv.org/abs/2010.03045
"""
import torch
from torch import nn
from einops import Rearrange

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
        dilation=1, groups=1, relu=True, bn=True,bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias_attr=bias)
        self.bn = nn.BatchNorm2d(out_planes, epsilon=1e-5, momentum=0.01) \
            if bn else None

        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Z_Pool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.concat((torch.max(x, 1).unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), axis=1)

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = Z_Pool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = self.sigmoid(x_out)
        return x * scale

class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.ChannelGateH = nn.Sequential(Rearrange('b c h w -> b h c w'), SpatialGate(), Rearrange('b h c w -> b c h w'))
        self.ChannelGateW = nn.Sequential(Rearrange('b c h w -> b w h c'), SpatialGate(), Rearrange('b w h c -> b c h w'))
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out1 = self.ChannelGateH(x)
        x_out2 = self.ChannelGateW(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x)
            x_out = (1 / 3) * (x_out + x_out1 + x_out2)
        else:
            x_out = (1 / 2) * (x_out1 + x_out2)
        return x_out