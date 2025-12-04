import torch
from torch import nn
import torch.nn.functional as F

from .position_encoding import PositionalEncoding, LearnablePositionalEncoding, SpatialPositionEncoding

class EmbeddingWithPE(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, max_len: int, *, use_learnable=False):
        super(EmbeddingWithPE, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pe = PositionalEncoding(embed_dim, max_len) if not use_learnable else LearnablePositionalEncoding(embed_dim, max_len)

    def forward(self, x):
        x = self.embedding(x)  
        x = self.pe(x)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, n_channels: int, embed_dim: int, patch_size: int|tuple[int, int], *, use_cls_token=False):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.splitter = nn.Conv2d(n_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False)
        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x):
        x = self.splitter(x)  # (B, D, H//P, W//P)
        x = x.flatten(2)  # (B, D, N) while N = (H*W)//(P*P)
        x = x.transpose(1, 2)  # (B, N, D)
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, D)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)
        return x

class PatchEmbeddingWithPE(nn.Module):
    def __init__(self, n_channels: int, embed_dim: int, patch_size: int|tuple[int, int], max_num_patch: int, *, use_learnable=False, use_cls_token=False):
        super(PatchEmbeddingWithPE, self).__init__()
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.splitter = nn.Conv2d(n_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False)
        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            max_num_patch += 1  # 为 CLS token 预留位置
        self.pe = PositionalEncoding(embed_dim, max_num_patch) if not use_learnable else LearnablePositionalEncoding(embed_dim, max_num_patch)

    def forward(self, x):
        x = self.splitter(x)  # (B, D, H//P, W//P)
        x = x.flatten(2)  # (B, D, N) while N = (H*W)//(P*P)
        x = x.transpose(1, 2)  # (B, N, D)
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, D)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)
        x = self.pe(x)
        return x

class SpatialEmbedding(nn.Module):
    def __init__(self, n_channels: int, embed_dim: int):
        super(SpatialEmbedding, self).__init__()
    
        self.conv = nn.Conv2d(n_channels, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # (B, C, H, W) -> (B, H, W, D)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        return x
class SpatialEmbeddingWithPE(nn.Module):
    def __init__(self, n_channels: int, embed_dim: int, scale: float|None=None):
        super(SpatialEmbeddingWithPE, self).__init__()
    
        self.embed = SpatialEmbedding(n_channels, embed_dim)
        self.pe = SpatialPositionEncoding(embed_dim, scale)

    def forward(self, x: torch.Tensor, img_size: tuple[int, int]):
        # (B, H, W, D)
        x = self.embed(x)
        x += self.pe(img_size)
        return x

class PatchEmbedding3D(nn.Module):
    def __init__(self, n_channels: int, embed_dim: int, patch_size: int|tuple[int, int, int], *, use_cls_token=False):
        super(PatchEmbedding3D, self).__init__()
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size, patch_size)
        else:
            self.patch_size = patch_size
        self.splitter = nn.Conv3d(n_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False)
        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x):
        x = self.splitter(x)  # (B, D, D'//P_d, H//P_h, W//P_w)
        x = x.flatten(2)  # (B, D, N) while N = (D'*H*W)//(P_d*P_h*P_w)
        x = x.transpose(1, 2)  # (B, N, D)
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, D)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)
        return x

class PatchEmbedding3DWithPE(nn.Module):
    def __init__(self, n_channels: int, embed_dim: int, patch_size: int|tuple[int, int, int], max_num_patch: int, *, use_learnable=False, use_cls_token=False):
        super(PatchEmbedding3DWithPE, self).__init__()
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size, patch_size)
        else:
            self.patch_size = patch_size
        self.splitter = nn.Conv3d(n_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False)
        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            max_num_patch += 1  # 为 CLS token 预留位置
        self.pe = PositionalEncoding(embed_dim, max_num_patch) if not use_learnable else LearnablePositionalEncoding(embed_dim, max_num_patch)

    def forward(self, x):
        x = self.splitter(x)  # (B, D, D'//P_d, H//P_h, W//P_w)
        x = x.flatten(2)  # (B, D, N) while N = (D'*H*W)//(P_d*P_h*P_w)
        x = x.transpose(1, 2)  # (B, N, D)
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, D)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)
        x = self.pe(x)
        return x
