import torch
from torch import nn
import torch.nn.functional as F

class ClsHead(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, hidden_dim: int = 1024):
        super(ClsHead, self).__init__()
        self.is_multiclass = n_classes > 1
        if self.is_multiclass:
            self.act = nn.Softmax(dim=1)
        else:
            self.act = nn.Sigmoid()

        self.cls_linear = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.cls_head = nn.Linear(hidden_dim, n_classes)
        self._init_weights()

    def forward(self, x):
        x = self.cls_linear(x)
        x = self.cls_head(x)
        x = self.act(x)
        return x

    def _init_weights(self):
        """初始化网络权重"""
        # 分类层使用正态分布初始化
        nn.init.normal_(self.cls_head.weight, std=0.01)
        nn.init.constant_(self.cls_head.bias, 0)
