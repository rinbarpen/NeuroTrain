import torch
from torch import nn
import torch.nn.functional as F
from typing import Literal
from einops import rearrange, einsum

AttentionType = Literal['mha', 'gqa', 'mqa']

class _MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_groups: int|None=None, attn_type: AttentionType='mha', qkv_bias=True, attn_dropout=0.2, *, share_kv=False):
        super(_MultiHeadAttention, self).__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        if attn_type == 'mha':
            self.num_groups = num_heads
        elif attn_type == 'mqa':
            self.num_groups = 1
        elif attn_type == 'gqa':
            self.num_groups = num_groups
            assert self.num_groups is not None, "num_groups must be set and greater than 0 with QGA"

        self.proj_q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.proj_v = nn.Linear(embed_dim, self.num_groups * self.head_dim, bias=qkv_bias)
        if share_kv:
            self.proj_k = nn.Linear(embed_dim, self.num_groups * self.head_dim, bias=qkv_bias)
        else:
            self.proj_k = self.proj_v

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor|None=None):
        q = self.proj_q(q)
        k = self.proj_k(k)
        v = self.proj_v(v)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (g d) -> b g n d', g=self.num_groups)
        v = rearrange(v, 'b n (g d) -> b g n d', g=self.num_groups)
        
        q = rearrange(q, 'b (g h_per_g) n d -> b g h_per_g n d', g=self.num_groups)

        attn_scores = einsum(q, k, 'b g h q d, b g n d -> b g h q n') * self.scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        o = einsum(attn_weights, v, 'b g h q n, b g n d -> b g h q d')
        o = rearrange(o, 'b g h n d -> b n (g h d)')

        o = self.out_proj(o)
        return o, attn_weights.sum(dim=(1, 2))

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_groups: int|None=None, attn_type: AttentionType='mha', qkv_bias=True, attn_dropout=0.2, *, share_kv=False):
        super(MultiHeadAttention, self).__init__()

        self.attn = _MultiHeadAttention(embed_dim, num_heads, num_groups, attn_type=attn_type, qkv_bias=qkv_bias, attn_dropout=attn_dropout, share_kv=share_kv)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor|None=None):
        return self.attn(x, x, x, mask=mask)

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_groups: int|None=None, attn_type: AttentionType='mha',
     qkv_bias=True, attn_dropout=0.2):
        super(MultiHeadCrossAttention, self).__init__()

        self.attn = _MultiHeadAttention(embed_dim, num_heads, num_groups, attn_type=attn_type, qkv_bias=qkv_bias, attn_dropout=attn_dropout, share_kv=False)
    
    def forward(self, qk: torch.Tensor, v: torch.Tensor, mask: torch.Tensor|None=None):
        return self.attn(qk, qk, v, mask=mask)

