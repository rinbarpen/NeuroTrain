"""
    Coordinate Attention for Efficient Mobile Network Design
    https://ieeexplore.ieee.org/document/9577301/
"""

import torch
from torch import nn
import torch.nn.functional as F

class CAModule(nn.Module):
    def __init__(self):
        super(CAModule, self).__init__()

        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        out_avg = torch.mean(x, dim=1, keepdim=True)
        out_max, _ = torch.max(x, dim=1, keepdim=True)
        a = torch.cat([out_avg, out_max], dim=1)
        a = F.sigmoid(self.conv(a))
        return x * a.reshape_as(x)

