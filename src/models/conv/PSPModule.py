"""
    Pyramid Scene Parsing Network
    http://arxiv.org/abs/1612.01105
"""

import torch
from torch import nn


# Use avgpool to protect smaller features
# If using maxpool, the bigger features may cover the smaller ones
# Perfer when in_channels is tiny (less than 128)
class PSPModule(nn.Module):
    def __init__(
        self, in_channels, out_channels, image_size, pool_sizes: tuple = (1, 2, 3, 6)
    ):
        super(PSPModule, self).__init__()

        self.pools = nn.ModuleList()
        for pool_size in pool_sizes:
            self.pools.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_size),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    nn.UpsamplingBilinear2d(image_size / pool_size),
                )
            )

    def forward(self, x):
        ys = [pool(x) for pool in self.pools]
        x = torch.cat([x, *ys], dim=1)
        return x
