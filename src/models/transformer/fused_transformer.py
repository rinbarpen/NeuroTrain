import math

import torch
from torch import nn
import torch.nn.functional as F
from ..embedding import PatchEmbedding
from ..position_encoding import PositionalEncoding

# embed_dim >= 30

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

# low memory while N < d
def linear_product(q, k, v, mask=None):
    # qkv: (B, H, N, d)
    d_k = q.size()[-1]
    attn_logits = torch.matmul(k, v.transpose(-2, -1)) # (N, N)
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, k) # (N, d)
    return values, attention


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o

class MultiheadCrossAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


# ViT
# H C Attention
# H W Attention
# W C Attention
class VisionTransformerEncoder(nn.Module):
    # max_num_patch = image_size // patch_size
    def __init__(self, n_layers, n_channels, embed_dim, num_heads, patch_size, max_num_patch, r, attn_dropout, mlp_dropout):
        self.n_layers = n_layers
        self.embedding = PatchEmbedding(n_channels, embed_dim, patch_size)
        self.pe = PositionalEncoding(embed_dim, max_num_patch)
        self.attn = MultiheadAttention(embed_dim, embed_dim, num_heads)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * r),
            nn.Dropout(mlp_dropout),
            nn.LeakyReLU(),
            nn.Linear(embed_dim * r, embed_dim),
        )
        self.mlp_dropout = nn.Dropout(mlp_dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)


    def forward(self, x):
        x = self.embedding(x)
        x = self.pe(x)
        for _ in range(self.n_layers):
            x = self.norm1(x)
            x = self.attn_dropout(self.attn(x)) + x
            x = self.norm2(x)
            x = self.mlp_dropout(self.mlp(x)) + x
        return x

class CrossVisionTransformerEncoder(nn.Module):
    # max_num_patch = image_size // patch_size
    def __init__(self, n_layers, n_channels, embed_dim, num_heads, patch_size, max_num_patch, r, attn_dropout, mlp_dropout):
        self.n_layers = n_layers
        self.embedding = PatchEmbedding(n_channels, embed_dim, patch_size)
        self.pe = PositionalEncoding(embed_dim, max_num_patch)
        self.attn = MultiheadAttention(embed_dim, embed_dim, num_heads)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * r),
            nn.Dropout(mlp_dropout),
            nn.LeakyReLU(),
            nn.Linear(embed_dim * r, embed_dim),
        )
        self.mlp_dropout = nn.Dropout(mlp_dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)


    def forward(self, x, y, z):
        x, y, z = self.embedding(x), self.embedding(x), self.embedding(z)
        x, y, z = self.pe(x), self.pe(y), self.pe(z)
        for _ in range(self.n_layers):
            x = self.norm1(x)
            x = self.attn_dropout(self.attn(x, y, z)) + x
            x = self.norm2(x)
            x = self.mlp_dropout(self.mlp(x)) + x
        return x
