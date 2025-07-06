import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return self.gamma * x + self.beta
    