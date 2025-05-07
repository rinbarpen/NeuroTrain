import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from ..conv.PSPModule import PSPModule
from ..conv.ACConv import ACConv
from ..conv.DWConv import get_dwconv, get_dwconv_layer2d
from ..embedding import PatchEmbeddingWithPE

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class LightDoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, use_dwconv=True):
        super(LightDoubleConv, self).__init__()

        self.conv1 = nn.Sequential(
            get_dwconv_layer2d(in_channels, out_channels, kernel_size=3),
            nn.ReLU(inplace=True),
        ) if use_dwconv else nn.Sequential(
            ACConv(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class InceptionDoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, use_dwconv=True):
        super(InceptionDoubleConv, self).__init__()

        self.branch1 = DoubleConv(in_channels, out_channels)
        self.branch2 = LightDoubleConv(in_channels, out_channels, use_dwconv=use_dwconv)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        return x1 + x2 + x

class ChannelAttention(nn.Module):
    def __init__(self, n_channels, kernel_size=7):
        super(ChannelAttention, self).__init__()

        self.conv = nn.Conv1d(n_channels, n_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=n_channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        avg_x = F.adaptive_avg_pool2d(x, 1)
        max_x = F.adaptive_max_pool2d(x, 1)
        o = torch.cat([avg_x, max_x], dim=1).unsqueeze(-1)
        o = self.conv(o).squeeze(-1)
        o = self.sigmoid(o)
        return x * o.expand_as(x)

class CrossAttention(nn.Module):
    def __init__(self, qkv_bias=False, embed_dim: int=512, n_heads: int=8, attn_dropout: float = 0.3, proj_dropout: float = 0.3):
        super(CrossAttention, self).__init__()

        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // n_heads

        self.proj_qk = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        self.proj_v = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(attn_dropout, inplace=True)
        self.proj_dropout = nn.Dropout(proj_dropout, inplace=True)

    def forward(self, qk, v):
        B, N, D = qk.shape
        # qk: (B, N, D), v: (B, N, D)
        qk, v = self.proj_qk(qk), self.proj_v(v)
        q, k = qk.split(self.head_dim)

        q = rearrange(q, 'b n (d h) -> b h n d', h=self.n_heads)
        k = rearrange(k, 'b n (d h) -> b h n d', h=self.n_heads)
        v = rearrange(v, 'b n (d h) -> b h n d', h=self.n_heads)

        scale = (self.head_dim) ** -0.5

        a = q.matmul(k.transpose(-1, -2) * scale)
        a = F.softmax(a, dim=-1)
        a = self.attn_dropout(a)
        y = a.matmul(v) # (B, H, N, d)
        y = rearrange(y, 'b h n d -> b n (h d)') # (B, N, D)
        y = self.proj(y)
        y = self.proj_dropout(y)
        return y

class EncoderDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderDoubleConv, self).__init__()

        self.double_conv = InceptionDoubleConv(in_channels, out_channels, use_dwconv=False)
        self.attn = ChannelAttention(out_channels)

    def forward(self, x):
        x = self.double_conv(x)
        x = self.attn(x)
        return x

class DecoderDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim: int=512, patch_size: int=32):
        super(DecoderDoubleConv, self).__init__()

        self.double_conv = LightDoubleConv(in_channels, out_channels, use_dwconv=True)
        self.pe_qk = PatchEmbeddingWithPE(out_channels, embed_dim=embed_dim, patch_size=patch_size)
        self.pe_v = PatchEmbeddingWithPE(out_channels, embed_dim=embed_dim, patch_size=patch_size)
        self.cross_attn = CrossAttention(qkv_bias=True, embed_dim=embed_dim)
        self.double_conv2 = DoubleConv(out_channels, out_channels)

    def forward(self, x, x_prev):
        x = self.double_conv(x)
        x, x_prev = self.pe_qk(x), self.pe_v(x_prev)
        x = self.cross_attn(x, x_prev)
        x = self.double_conv2(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor):
        return self.conv(x)

class SegmentNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int):
        super(SegmentNet, self).__init__()
    
        self.stage1 = nn.Sequential(
            EncoderDoubleConv(n_channels, 64),
        )
        self.stage2 = nn.Sequential(
            nn.MaxPool2d(2),
            EncoderDoubleConv(64, 128),
        )
        self.stage3 = nn.Sequential(
            nn.MaxPool2d(2),
            EncoderDoubleConv(128, 256),
        )
        self.stage4 = nn.Sequential(
            nn.MaxPool2d(2),
            EncoderDoubleConv(256, 512),
        )

        self.decoder1 = DecoderDoubleConv(512, 256)
        self.decoder2 = DecoderDoubleConv(256, 128)
        self.decoder3 = DecoderDoubleConv(128, 64)

        self.out_conv1 = OutConv(64, n_classes)
        self.out_conv2 = OutConv(128, n_classes)
        self.out_conv3 = OutConv(256, n_classes)

    def forward(self, x: torch.Tensor):
        x1 = self.stage1(x)  # (B, 64, 512, 512)
        x2 = self.stage2(x1) # (B, 128, 256, 256)
        x3 = self.stage3(x2) # (B, 256, 128, 128)
        x4 = self.stage4(x3) # (B, 512, 64, 64)

        y4 = x4
        y3 = self.decoder1(y4, x3)
        y2 = self.decoder2(y3, x2)
        y1 = self.decoder3(y2, x1)

        y1 = self.out_conv1(y1)
        y2 = self.out_conv2(y2)
        y3 = self.out_conv3(y3)

        return y1, y2, y3
