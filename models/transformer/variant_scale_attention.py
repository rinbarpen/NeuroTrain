# TODO: Test
import torch
import torch.nn.functional as F
from torch import nn

from ..position_encoding import PositionalEncoding, LearnablePositionalEncoding


class PatchEmbedding(nn.Module):
    def __init__(
        self, in_channels: int, embed_dim: int, patch_size: int | tuple[int, int] = 16
    ):
        super(PatchEmbedding, self).__init__()
        self.patch_size = (
            patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        )
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.splitter = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.pe = PositionalEncoding(embed_dim, max_len=500)
        # self.pe = LearnablePositionalEncoding(embed_dim, max_len=500)

    def forward(self, x):
        assert (
            x.shape[-2] % self.patch_size[0] == 0
            and x.shape[-1] % self.patch_size[1] == 0
        ), "Cannot split entirely"
        x = self.splitter(x)  # (B, C, H, W) -> (N, D, H*W//P[0]//P[1])
        x = x.flatten(2)  # (B, D, N)
        x = x.transpose(1, 2)  # (B, D, N) -> (B, N, D)

        x = self.pe(x)

        return x


class VariantScaleImageAttention(nn.Module):
    """
    1. Flatten the 2D image features(B, C, H, W) cutted into groups of P*P blocks to (B, N, D)
    """

    def __init__(
        self,
        n_channels,
        hidden_dim,
        qkv_channels: tuple[int, int, int],
        patch_size: int | tuple[int, int] = 16,
        num_head: int = 8,
        dropout: float = 0.3,
        *,
        groups=1,
    ):
        super(VariantScaleImageAttention, self).__init__()

        self.adjust_conv_qkv = (
            nn.Conv2d(c, n_channels, kernel_size=1, groups=groups, bias=True)
            for c in qkv_channels
        )

        self.pe = PatchEmbedding(n_channels, hidden_dim, patch_size)
        self.attn = VariantScaleAttention(
            hidden_dim, num_head=num_head, dropout=dropout
        )

    def forward(self, q, k, v):
        q, k, v = self._adjust_channels(q, k, v)  # adjust qkv channels to n_channels

        q = self.pe(q)
        k = self.pe(k)
        v = self.pe(v)

        x = self.attn(q, k, v)
        return x

    def _adjust_channels(self, q, k, v):
        q = self.self.adjust_conv_qkv[0](q)
        k = self.self.adjust_conv_qkv[1](k)
        v = self.self.adjust_conv_qkv[2](v)
        return q, k, v


# 对 V 加权
class VariantScaleAttention(nn.Module):
    def __init__(self, hidden_dim, num_head: int = 8, dropout: float = 0.3):
        super(VariantScaleAttention, self).__init__()

        self.num_head = num_head
        self.head_dim = hidden_dim // num_head
        self.attn_dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, q, k, v):
        assert (
            q.shape[1] % self.num_head == 0
        ), "n_channels must be divisible by num_head"
        # qkv: (B, N, D) -> (B, N, H, D//H(d)) -> (B, H, N, d)
        q = q.view(q.shape[0], q.shape[1], self.num_head, self.head_dim).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.num_head, self.head_dim).transpose(1, 2)
        v = k.view(v.shape[0], v.shape[1], self.num_head, self.head_dim).transpose(1, 2)

        scale = self.head_dim**-0.5
        a = torch.matmul(q, k.transpose(-2, -1) * scale)
        a = F.softmax(a, dim=-1)
        a = self.attn_dropout(a)
        a = torch.matmul(a, v)
        return a


class FeedForward(nn.Module):
    def __init__(self, in_channels, reduction: int, act_layer=nn.ReLU):
        super(FeedForward, self).__init__()

        out_channels = in_channels // reduction
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, in_channels)

        self.act = act_layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dim,
        qkv_channels: tuple[int, int, int],
        reduction,
        patch_size: int | tuple[int, int] = 16,
        num_head=8,
        dropout: float = 0.3,
        act_layer=nn.ReLU,
        *,
        groups=1,
    ):
        super(TransformerEncoder, self).__init__()

        self.vision_attn = VariantScaleImageAttention(
            in_channels,
            hidden_dim,
            qkv_channels,
            patch_size,
            num_head,
            dropout,
            groups=groups,
        )

        self.feed_forward = FeedForward(in_channels, reduction, act_layer=act_layer)

    def forward(
        self, x: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor
    ):
        if isinstance(x, torch.Tensor):
            x = (x, x, x)

        a = self.vision_attn(x[0], x[1], x[2])
        o = self.feed_forward(a)
        return o
