import torch
from torch import nn
import torch.nn.functional as F

class BBoxHead(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, hidden_dim: int = 1024):
        super(BBoxHead, self).__init__()
        self.is_multiclass = n_classes > 1
        if self.is_multiclass:
            self.act = nn.Softmax(dim=1)
        else:
            self.act = nn.Sigmoid()
        self.bbox_linear = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.bbox_head = nn.Linear(hidden_dim, 4 * n_classes)
        self._init_weights()
    
    def forward(self, x):
        x = self.bbox_linear(x)
        x = self.bbox_head(x)
        x = self.act(x)
        return x

    def _init_weights(self):
        nn.init.normal_(self.bbox_head.weight, std=0.01)
        nn.init.constant_(self.bbox_head.bias, 0)
