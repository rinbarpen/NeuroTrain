# TODO: Test
import math
import torch
from torch import nn

class WeightGate(nn.Module):
    def __init__(self):
        super(WeightGate, self).__init__()
        self.weight = nn.Parameter(torch.randn(1))

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        # a, b are (B, C, X..)
        weight = torch.sigmoid(self.weight)
        weight = weight.to(a.device)
        return a * weight, b * (1.0 - weight)

class DirectMultiWeightGate(nn.Module):
    def __init__(self, num_tensor: int):
        super(DirectMultiWeightGate, self).__init__()
        self.weights = nn.Parameter(torch.randn(num_tensor))

    def forward(self, *x: torch.Tensor):
        # x are (B, C, X..)
        weights = torch.sigmoid(self.weights)
        weights = weights.to(x[0].device)
        weighted_outputs = []
        for i, tensor in enumerate(x):
            weighted_outputs.append(tensor * weights[i])
        return weighted_outputs
