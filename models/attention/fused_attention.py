import torch
from torch import nn
import torch.nn.functional as F

from .channel_attention import ChannelAttention, EnhancedChannelAttention
from .spatial_attention import SpatialAttention

class CBAM(nn.Module):
    """
        CBAM: Convolutional Block Attention Module
        http://arxiv.org/abs/1807.06521
    """
    def __init__(self, channels, reduction: int=1, use_optimization: bool=True):
        super(CBAM, self).__init__()

        self.ca = EnhancedChannelAttention() if use_optimization else ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x
