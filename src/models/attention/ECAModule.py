"""
    ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
    http://arxiv.org/abs/1910.03151
"""

import torch
from torch import nn
import torch.nn.functional as F

class ECAModule(nn.Module):
    def __init__(self, channels: int, kernel_size: int=7, **kwargs):
        super(ECAModule, self).__init__()

        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=True)

    def forward(self, x):
        avg_x = torch.mean(x, dim=(-1, -2)).unsqueeze(-1).transpose(-1, -2) # (B, 1, C)
        avg_x = self.conv(avg_x) # (B, 1, C)
        avg_x = avg_x.transpose(-1, -2).unsqueeze(-1) # (B, C, 1)
        scale = F.sigmoid(avg_x)

        return x * scale.expand_as(x)
