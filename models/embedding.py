import torch
from torch import nn
import torch.nn.functional as F

from .position_encoding import PositionalEncoding, LearnablePositionalEncoding

# class Embedding(nn.Module):
#     def __init__(self, vocab_size: int, embed_dim: int):
#         super(Embedding).__init__(self)
#         self.embedding = nn.Embedding(vocab_size, embed_dim)

#     def forward(self, x):
#         x = self.embedding(x)
#         return x

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
    def __init__(self, n_channels: int, embed_dim: int, patch_size: int|tuple[int, int], max_num_patch: int, *, use_learnable=False):
        super(PatchEmbeddingWithPE, self).__init__()
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.splitter = nn.Conv2d(n_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False)
        self.pe = PositionalEncoding(embed_dim, max_num_patch) if not use_learnable else LearnablePositionalEncoding(embed_dim, max_num_patch)

    def forward(self, x):
        x = self.splitter(x)  # (B, D, H//P, W//P)
        x = x.flatten(2)  # (B, D, N) while N = (H*W)//(P*P)
        x = x.transpose(1, 2)  # (B, N, D)
        x = self.pe(x)
        return x

class EmbeddingWithPE(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, max_len: int, *, use_learnable=False):
        super(EmbeddingWithPE, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pe = PositionalEncoding(embed_dim, max_len=max_len) if not use_learnable else LearnablePositionalEncoding(embed_dim, max_len)

    def forward(self, x):
        x = self.embedding(x)  
        x = self.pe(x)
        return x
