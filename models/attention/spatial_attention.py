import torch
from torch import nn

class SpatialAttention(nn.Module):
    """
        Coordinate Attention for Efficient Mobile Network Design
        https://ieeexplore.ieee.org/document/9577301/
    """
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out_avg = torch.mean(x, dim=1, keepdim=True)
        out_max, _ = torch.max(x, dim=1, keepdim=True)
        o = torch.cat([out_avg, out_max], dim=1)
        o = self.sigmoid(self.conv(o))
        return x * o.expand_as(x)
