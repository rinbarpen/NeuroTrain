# TODO: Test
import torch
from torch import nn

class ScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers: int=1):
        super(ScaleConv, self).__init__()

        self.n_layers = n_layers

        self.conv1 = nn.Conv2d(in_channels[0], out_channels[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels[1], out_channels[1], kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels[2], out_channels[2], kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels[3], out_channels[3], kernel_size=3, padding=1)


    def forward(self, x):
        pass
