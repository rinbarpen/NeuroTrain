from torch import nn
import torch

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
    
        self.conv = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x
