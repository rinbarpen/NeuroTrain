"""
    CBAM: Convolutional Block Attention Module
    http://arxiv.org/abs/1807.06521
"""
import torch
from torch import nn
import torch.nn.functional as F

from .ECAModule import ECAModule
from .CAModule import CAModule
from .SEModule import SEModule

class CBAM(nn.Module):
    def __init__(self, channels, reduction: int=1, use_optimization: bool=True):
        super(CBAM, self).__init__()

        self.ca = ECAModule() if use_optimization else SEModule(channels, reduction)
        self.sa = CAModule()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x
