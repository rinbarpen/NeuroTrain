import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet34
import timm
from einops import einsum, rearrange, repeat, reduce

from typing import List, Tuple, Type, Sequence

from ..conv.DWConv import get_dwconv_layer3d, get_dwconv_layer2d
from ..conv.ACConv import ACConv
from ..attention.CAModule import CAModule
from ..attention.SEModule import SEModule
from ..attention.CBAM import CBAM

class DepthAttention3D(nn.Module):
    """
    Attention mechanism focusing on the depth dimension.
    Pools over spatial dimensions (H, W), then applies a 1D convolution along depth.
    """
    def __init__(self, in_channels):
        super().__init__()
        # Apply a 1D conv (1x1xk) along the depth dimension.
        # Input: B, C, D, 1, 1 -> Output: B, 1, D, 1, 1
        self.conv_depth = nn.Conv3d(in_channels, 1, kernel_size=1, bias=False)

    def forward(self, x):
        # x: B, C, D, H, W
        avg_out = torch.mean(x, dim=(-1, -2), keepdim=True) # B, C, D, 1, 1
        
        # Apply convolution along the depth dimension
        # The convolution combines channel information for each depth slice
        # and outputs a single value per depth slice.
        depth_attention_map = F.sigmoid(self.conv_depth(avg_out)) # B, 1, D, 1, 1
        return depth_attention_map * x # Broadcast attention map across H, W and channels

class ChannelAttention3D(nn.Module):
    """
    3D通道注意力模块
    输入: (B, C, D, H, W)
    输出: (B, C, D, H, W)
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.mlp = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attn = self.sigmoid(avg_out + max_out)
        return x * attn

class SpatialAttention3D(nn.Module):
    """
    3D空间注意力模块
    输入: (B, C, D, H, W)
    输出: (B, C, D, H, W)
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv(attn)
        attn = self.sigmoid(attn)
        return x * attn

class CBAM3D(nn.Module):
    """
    3D版CBAM注意力模块，包含通道注意力和空间注意力。
    输入: (B, C, D, H, W)
    输出: (B, C, D, H, W)
    """
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attn = ChannelAttention3D(in_channels, reduction)
        self.spatial_attn = SpatialAttention3D(kernel_size)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x

class TripleAttention3D(nn.Module):
    """
    Triple attention mechanism combining:
    1. Channel Attention (SEModule)
    2. Spatial Attention (CAModule)
    3. Depth Attention (DepthAttention3D)
    
    Each attention module is applied sequentially to the input tensor.
    """
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(TripleAttention3D, self).__init__()
        self.channel_attn = ChannelAttention3D(in_channels, reduction)
        self.spatial_attn = SpatialAttention3D(kernel_size)
        self.depth_attention = DepthAttention3D(in_channels)

    def forward(self, x):
        x = self.channel_attn(x)
        x1 = self.spatial_attn(x)
        x2 = self.depth_attention(x)
        return x1, x2

class LiteAttention(nn.Module):
    def __init__(self, in_channels: int):
        super(LiteAttention, self).__init__()

        self.dw_conv = get_dwconv_layer2d(in_channels, in_channels, kernel_size=7)
        self.ac_conv = ACConv(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        s = torch.cat([torch.mean(x, dim=1, keepdim=True), 
                        torch.max(x, dim=1, keepdim=True)[0]], dim=1)
        sa = self.ac_conv(F.sigmoid(s))

        c = torch.mean(x, dim=(-1, -2))
        ca = F.sigmoid(self.dw_conv(c))

        return x * ca.expand_as(x) * sa.expand_as(x)

class FlatConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3) -> None:
        super().__init__()
    
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size//2)

        self.out_conv = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1)
    
    def forward(self, x):
        x = torch.cat([x, self.conv1(x), self.conv2(x), self.conv3(x)], dim=1)
        x = self.out_conv(x)
        return x

class DecoderConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.attn = LiteAttention(in_channels)
        self.conv = FlatConv(in_channels, out_channels)
    
    def forward(self, x):
        x = self.attn(x)
        x = self.conv(x)

        return x

class MaskEncoder(nn.Module):
    def __init__(self, in_channels: list[int], out_channels: list[int]):
        super(MaskEncoder, self).__init__()

        self.down1 = nn.Sequential(
            nn.Conv3d(in_channels[0], out_channels[0], kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(in_channels[1], out_channels[1], kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            TripleAttention3D(out_channels[1]),
        )
        self.down2 = nn.Sequential(
            nn.Conv3d(in_channels[2], out_channels[2], kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(in_channels[3], out_channels[3], kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            TripleAttention3D(out_channels[3]),
        )

    def forward(self, x):
        sx, dx = self.down1(x)
        x = sx + dx
        sx, dx = self.down2(x)

        sx, dx = torch.max(sx, dim=2)[0], torch.max(dx, dim=2)[0] # (B, C, H, W)
        return torch.cat([sx, dx], dim=1)

class CrossAttention(nn.Module):
    """
    Multi-head Cross-Attention mechanism for 3D feature maps.
    Flattens D*H*W, applies attention, then reshapes back.
    Assumes query and key_value features have the same number of channels (dim).
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias) # For Key and Value
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query_feature, key_value_feature, mask=None):
        # 范式：query和key/value空间shape可不同，输出shape与kv一致
        B, C, Hq, Wq = query_feature.shape
        _, _, Hkv, Wkv = key_value_feature.shape

        # 先将query特征插值到key_value特征的空间shape
        if (Hq != Hkv) or (Wq != Wkv):
            query_feature = F.interpolate(query_feature, size=(Hkv, Wkv), mode="bilinear", align_corners=False)
        q = rearrange(query_feature, 'b c h w -> b (h w) c')
        kv = rearrange(key_value_feature, 'b c h w -> b (h w) c')

        # Q, K, V投影
        q = self.q_proj(q)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim)
        kv = self.kv_proj(kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads)
        kv = rearrange(kv, 'b n (h c d) -> b h c n d', h=self.num_heads, d=self.head_dim, c=2)
        k, v = torch.chunk(kv, 2, dim=2) # (b, h, Nkv, d)

        # 注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale # (b, h, Nq, Nkv)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 加权V
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C) # (B, Nq, D)
        x = self.proj(x)
        x = self.proj_drop(x)

        # 恢复kv空间shape
        x = x.transpose(1, 2).reshape(B, C, Hkv, Wkv)
        return x

class Encoder(nn.Module):
    def __init__(self, n_channels=1, backbone_fn=resnet34, pretrained=True):
        super().__init__()

        backbone = backbone_fn(pretrained=pretrained)
        self.prepare_layer = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.prepare_layer = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        xs = []
        x = self.prepare_layer(x) # (64, 224, 224)
        print(x.shape)
        x = self.layer1(x) # (64, 112, 112)
        xs.append(x)
        x = self.layer2(x) # (128, 56, 56)
        xs.append(x)
        x = self.layer3(x) # (256, 28, 28)
        xs.append(x)
        x = self.layer4(x) # (512, 14, 14)
        xs.append(x)
        return xs

class Decoder(nn.Module):
    def __init__(self, n_channels=[64, 128, 256, 512], dim=512, qkv_bias=True, attn_drop=0.0, proj_drop=0.0, ):
        super().__init__()
        self.ca1 = CrossAttention(dim, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.ca2 = CrossAttention(dim, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.ca3 = CrossAttention(dim, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.ca4 = CrossAttention(dim, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.block1 = DecoderConvBlock(n_channels[3], n_channels[2])
        self.block2 = DecoderConvBlock(n_channels[2], n_channels[1])
        self.block3 = DecoderConvBlock(n_channels[1], n_channels[0])
        self.block4 = DecoderConvBlock(n_channels[0], n_channels[0])

    def forward(self, xs, prompt):
        y1 = self.ca1(prompt, xs[3])
        y2 = self.ca2(prompt, xs[2])
        y3 = self.ca3(prompt, xs[1])
        y4 = self.ca4(prompt, xs[0])
        y2 += self.block1(y1)
        y3 += self.block2(y2)
        y4 += self.block3(y3)
        y = self.block4(y4)
        return y

class Model(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, dim=512, depth_size: int|None=None, qkv_bias=True, attn_drop=0.0, proj_drop=0.0, decoder_channels=[64, 128, 256, 512]):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.depth_size = depth_size if depth_size is not None else 1

        self.encoder = Encoder(n_channels)
        self.decoder = Decoder(decoder_channels, dim=dim, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.mask_encoder = MaskEncoder([n_classes, *decoder_channels[:3]], decoder_channels)
        self.head = nn.Conv2d(decoder_channels[0], n_classes, kernel_size=1)

    def forward(self, xs):
        # (B, C, H, W, D)
        o = []
        for x in xs.unbind(-1):
            try:
                masks = torch.stack(o[-self.depth_size:], dim=2)
                prompt = self.mask_encoder(masks)
            except RuntimeError:
                b, _, h, w = x.shape
                in_shape = (b, self.n_classes, 1, h, w)
                masks = torch.randn(in_shape, device=x.device)
                prompt = self.mask_encoder(masks)

            ys = self.encoder(x)
            y = self.decoder(ys, prompt)

            y = self.head(y)
            o.append(y)
        return torch.cat(o, dim=2)

