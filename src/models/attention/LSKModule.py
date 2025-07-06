"""
    Selective Kernel Networks
    http://arxiv.org/abs/1903.06586
    
    Large Selective Kernel Network for Remote Sensing Object Detection
    http://arxiv.org/abs/2303.09030
"""

import torch
from torch import nn
import torch.nn.functional as F

from typing import *

class SKModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes: Tuple[int, int], reduction: int, **kwargs):
        super(SKModule, self).__init__()
        self.out_channels = out_channels

        self.conv_list = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes[0], padding=kernel_sizes[0] // 2),
                          nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes[1], padding=kernel_sizes[1] // 2)]
        self.conv_list = nn.ModuleList(self.conv_list)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        concat_channels = out_channels
        reduced_channels = max(concat_channels // reduction, 32)
        self.fc = nn.Sequential(
            nn.Linear(concat_channels, reduced_channels, bias=False),
            nn.BatchNorm1d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, concat_channels, bias=False),
            nn.BatchNorm1d(concat_channels),
            nn.ReLU(inplace=True)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        xs = [conv(x) for conv in self.conv_list]
        y = sum(xs)

        y = self.avg_pool(y) # (B, C, 1, 1)
        y = y.squeeze(-1).squeeze(-1)

        z = self.fc(y) # (B, d, 1, 1)
        attn = self.softmax(z) # (B, d, 1, 1)
        attn = attn.unsqueeze(-1).unsqueeze(-1)

        # TODO: this implementation is good for performance?
        o = xs[0] * attn.expand_as(xs[0]) + xs[1] * (1-attn).expand_as(xs[1])
        return o


class LSKModule(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(LSKModule, self).__init__()
        self.out_channels = out_channels

        N = len(kwargs['conv'])
        self.conv_list = [nn.Conv2d(in_channels, out_channels,
                               kernel_size=kwargs['conv'][i]['kernel_size'],
                               stride=kwargs['conv'][i]['stride'],
                               padding=kwargs['conv'][i]['padding'],
                               dilation=kwargs['conv'][i]['dilation'])
                      for i in range(N)]
        self.conv_fix_list = [nn.Conv2d(out_channels, out_channels, kernel_size=1)
                              for i in range(N)]

        self.conv_list = nn.ModuleList(self.conv_list)
        self.sa = nn.Sequential(
            nn.Conv2d(2, N, kernel_size=7, padding=3),
            nn.Sigmoid(N),
        )

    def forward(self, x):
        xs = [conv(x) for conv in self.conv_list]
        xs = [conv(xx) for xx, conv in zip(xs, self.conv_fix_list)]
        y = torch.concat(xs, dim=1)
        z = self.spatial_attention(y)

        gap = self.out_channels
        o = torch.zeros_like(xs[0])
        for i, xx in enumerate(xs):
            o += z[:][i*gap:(i+1)*gap][:][:] * xx
        o = o * x
        return o

    def spatial_attention(self, x):
        avg = F.adaptive_avg_pool2d(x, 1) # type: ignore
        max = F.adaptive_max_pool2d(x, 1) # type: ignore

        x = torch.concat([avg, max], dim=1)
        x = self.sa(x)
        return x
