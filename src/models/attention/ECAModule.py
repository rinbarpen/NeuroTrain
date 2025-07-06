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

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.activate = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, groups=channels, bias=True),
            nn.Sigmoid(inplace=True)
        )

    def forward(self, x):
        y = self.avg_pool(x).unsqueeze(-1)
        scale = self.activate(y).squeeze(-1)

        return x * scale.expand_as(x)
