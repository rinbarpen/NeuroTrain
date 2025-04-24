import torch
from torch import nn
import torch.functional as F

from .position_encoding import PositionalEncoding

class PatchEmbedding(nn.Module):
    def __init__(self, n_channels: int, embed_dim: int, patch_size: int|tuple[int, int]):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.splitter = nn.Conv2d(n_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False)

    def forward(self, x):
        x = self.splitter(x)  # (B, D, H//P, W//P)
        x = x.flatten(2)  # (B, D, N) while N = (H*W)//(P*P)
        x = x.transpose(1, 2)  # (B, N, D)
        return x

class PatchEmbeddingWithPE(nn.Module):
    def __init__(self, n_channels: int, embed_dim: int, patch_size: int|tuple[int, int]):
        super(PatchEmbeddingWithPE, self).__init__()
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.splitter = nn.Conv2d(n_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False)
        self.pe = PositionalEncoding(embed_dim)

    def forward(self, x):
        x = self.splitter(x)  # (B, D, H//P, W//P)
        x = x.flatten(2)  # (B, D, N) while N = (H*W)//(P*P)
        x = x.transpose(1, 2)  # (B, N, D)
        x = self.pe(x)
        return x
