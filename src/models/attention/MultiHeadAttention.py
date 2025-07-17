import torch
from torch import nn
import torch.nn.functional as F
from typing import Literal
from einops import rearrange, einsum

from ..position_encoding import Rotator

AttentionType = Literal['mha', 'gqa', 'mqa']

class _MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_groups: int|None=None, attn_type: AttentionType='mha', qkv_bias=True, attn_dropout=0.2, *, share_kv=False, rope=False):
        super(_MultiHeadAttention, self).__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.rope = rope

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
        if not share_kv:
            self.proj_k = nn.Linear(embed_dim, self.num_groups * self.head_dim, bias=qkv_bias)
        else:
            self.proj_k = self.proj_v

        self.attn_dropout = nn.Dropout(attn_dropout, inplace=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim)


    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor|None=None):
        q = self.proj_q(q)
        k = self.proj_k(k)
        v = self.proj_v(v)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (g d) -> b g n d', g=self.num_groups)
        v = rearrange(v, 'b n (g d) -> b g n d', g=self.num_groups)

        if self.rope:
            B, N, D = q.size()
            rotator = Rotator(D, N)
            q, k = rotator.rotate(q), rotator.rotate(k)

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
    def __init__(self, embed_dim: int, num_heads: int, num_groups: int|None=None, attn_type: AttentionType='mha', qkv_bias=True, attn_dropout=0.2, *, share_kv=False, rope=False):
        super(MultiHeadAttention, self).__init__()

        self.attn = _MultiHeadAttention(embed_dim, num_heads, num_groups, attn_type=attn_type, qkv_bias=qkv_bias, attn_dropout=attn_dropout, share_kv=share_kv, rope=rope)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor|None=None):
        return self.attn(x, x, x, mask=mask)

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_groups: int|None=None, attn_type: AttentionType='mha',
     qkv_bias=True, attn_dropout=0.2, *, rope=False):
        super(MultiHeadCrossAttention, self).__init__()

        self.attn = _MultiHeadAttention(embed_dim, num_heads, num_groups, attn_type=attn_type, qkv_bias=qkv_bias, attn_dropout=attn_dropout, share_kv=False, rope=rope)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor|None=None):
        return self.attn(q, k, v, mask=mask)

def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: tuple[int, int],
    k_size: tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class _SpatialMultiHeadAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: tuple[int, int]|None = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x

class SpatialMultiHeadAttention(nn.Module):        
    def __init__(
        self,
        in_channels: int, out_channels: int,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: tuple[int, int]|None = None,
        use_ffn = True,
        mlp_ratio = 4.0,
    ) -> None:
        super().__init__()

        self.use_ffn = use_ffn

        self.in_conv = nn.Conv2d(in_channels, dim, kernel_size=1)
        self.smha = _SpatialMultiHeadAttention(dim, num_heads, qkv_bias, use_rel_pos, rel_pos_zero_init, input_size)
        self.out_conv = nn.Conv2d(dim, out_channels, kernel_size=1)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.ReLU(inplace=True),
            nn.Linear(int(dim * mlp_ratio), dim),
        ) 

    def forward(self, x):
        x = self.in_conv(x)
        x += self.smha(x)
        if self.use_ffn:
            x += self.ffn(x)
        x = self.out_conv(x)
        return x
