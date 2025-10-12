from torch import nn
import torch
import torch.nn.functional as F

from typing import Type, Sequence

class TopKSelector(nn.Module):
    def __init__(self, k: int, dim: int, act: Type=nn.ReLU):
        super().__init__()
        self.k = k
        
        self.scorer = nn.Sequential(
            nn.Linear(dim, dim // 4),
            act(),
            nn.Linear(dim // 4, 1),
        )
    
    def forward(self, x: torch.Tensor):
        # (b, n, d)
        scores = self.scorer(x)

        scores = F.softmax(scores, dim=1)
        scores, indices = scores.topk(self.k)

        x = x[indices]
        return x, {
            'scores': scores,
            'indices': indices,
        }
