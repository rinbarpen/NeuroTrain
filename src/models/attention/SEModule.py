"""
    Squeeze-and-Excitation Networks
    http://arxiv.org/abs/1709.01507
"""
import torch
from torch import nn
import torch.nn.functional as F

class SEModule(nn.Module):
    def __init__(self, channels, reduction: int, **kwargs):
        super(SEModule, self).__init__()

        self.channels = channels

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )

    def forward(self, x):
        avg_x = torch.mean(x, dim=(-1, -2))
        scale = self.fc(avg_x)
        scale = F.sigmoid(scale)

        return x * scale