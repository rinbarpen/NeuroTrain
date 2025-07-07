import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, vgg19, vgg19_bn, resnet34

from typing import List, Tuple, Type, Sequence

from ...conv.DWConv import get_dwconv_layer3d, get_dwconv_layer2d
from ...conv.ACConv import ACConv
from ...attention.CAModule import CAModule
from ...attention.SEModule import SEModule
from ...attention.CBAM import CBAM

class DepthAttention3D(nn.Module):
    """
    Attention mechanism focusing on the depth dimension.
    Pools over spatial dimensions (H, W), then applies a 1D convolution along depth.
    """
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        # self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1)) # Pool to (D, 1, 1)
        # self.max_pool = nn.AdaptiveMaxPool3d((None, 1, 1)) # Pool to (D, 1, 1)
        # Apply a 1D conv (1x1xk) along the depth dimension.
        # Input: B, C, D, 1, 1 -> Output: B, 1, D, 1, 1
        self.conv_depth = nn.Conv3d(in_channels, 1, kernel_size=(kernel_size, 1, 1),
                                     padding=(kernel_size//2, 0, 0), bias=False)

    def forward(self, x):
        # x: B, C, D, H, W
        avg_out = torch.mean(x, dim=(-1, -2), keepdim=True) # B, C, D, 1, 1
        
        # Apply convolution along the depth dimension
        # The convolution combines channel information for each depth slice
        # and outputs a single value per depth slice.
        depth_attention_map = F.sigmoid(self.conv_depth(avg_out)) # B, 1, D, 1, 1
        return depth_attention_map * x # Broadcast attention map across H, W and channels

class CBAM3D(nn.Module):
    """
    3D版CBAM注意力模块，包含通道注意力和空间注意力。
    输入: (B, C, D, H, W)
    输出: (B, C, D, H, W)
    """
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.mlp = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        # 空间注意力
        self.conv_spatial = nn.Conv3d(2, 1, kernel_size=(kernel_size, kernel_size, kernel_size),
                                      padding=kernel_size//2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # 通道注意力
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_attn = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_attn

        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = torch.cat([avg_out, max_out], dim=1)
        spatial_attn = self.conv_spatial(spatial_attn)
        spatial_attn = self.sigmoid_spatial(spatial_attn)
        x = x * spatial_attn
        return x

class TripleAttention3D(nn.Module):
    """
    Triple attention mechanism combining:
    1. Channel Attention (SEModule)
    2. Spatial Attention (CAModule)
    3. Depth Attention (DepthAttention3D)
    
    Each attention module is applied sequentially to the input tensor.
    """
    def __init__(self, in_channels, kernel_size=3):
        super(TripleAttention3D, self).__init__()
        self.cbam = CBAM3D(in_channels)
        self.depth_attention = DepthAttention3D(in_channels, kernel_size)

    def forward(self, x):
        x = self.cbam(x)
        x = self.depth_attention(x)
        return x

class InceptionLightBlock(nn.Module):
    """
    Inception-like block with three branches:
    1. Standard Conv2d
    2. Depthwise Separable Conv2d
    3. ACConv (Attention Convolution)
    
    Each branch outputs a feature map, which are concatenated along the channel dimension.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True, activation: Type[nn.Module]=nn.ReLU):
        super(InceptionLightBlock, self).__init__()
        
        # Branch 1: Standard Conv2d
        self.branch_std = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=bias)
        
        # Branch 2: Depthwise Separable Conv2d
        self.branch_dw = get_dwconv_layer2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=bias)
        
        # Branch 3: ACConv (Attention Convolution)
        self.branch_ac = ACConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        
        # Final convolution to combine branches
        self.final_conv = nn.Conv2d(3 * out_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = activation(inplace=True)

    def forward(self, x):
        x_std = self.branch_std(x)
        x_dw = self.branch_dw(x)
        x_ac = self.branch_ac(x)

        # Concatenate along channel dimension
        x_combined = torch.cat([x_std, x_dw, x_ac], dim=1)

        x_out = self.final_conv(x_combined)
        x_out = self.bn(x_out)
        x_out = self.act(x_out)

        return x_out

class CrossAttention(nn.Module):
    """
    Multi-head Cross-Attention mechanism for 3D feature maps.
    Flattens D*H*W, applies attention, then reshapes back.
    Assumes query and key_value features have the same number of channels (dim).
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias) # For Key and Value
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query_feature, key_value_feature):
        # 范式：query和key/value空间shape可不同，输出shape与query一致
        B, C, Hq, Wq = query_feature.shape
        _, _, Hkv, Wkv = key_value_feature.shape

        # flatten空间维度
        query = query_feature.flatten(2).transpose(1, 2)  # (B, Nq, C)
        kv = key_value_feature.flatten(2).transpose(1, 2) # (B, Nkv, C)

        # Q, K, V投影
        q = self.q_proj(query).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # (B, num_heads, Nq, head_dim)
        kv_proj = self.kv_proj(kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads)
        k = kv_proj[:, :, 0].permute(0, 2, 1, 3) # (B, num_heads, Nkv, head_dim)
        v = kv_proj[:, :, 1].permute(0, 2, 1, 3) # (B, num_heads, Nkv, head_dim)

        # 注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale # (B, num_heads, Nq, Nkv)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 加权V
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C) # (B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # 恢复query空间shape
        x = x.transpose(1, 2).reshape(B, C, Hq, Wq)
        return x

class DownConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0, norm_layer=nn.BatchNorm3d, activation=nn.LeakyReLU):
        super(DownConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = norm_layer(out_channels) if norm_layer is not None else nn.Identity()
        self.act = activation(inplace=True) if activation is not None else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class UpBlockDecoder(nn.Module):
    """
    解码器上采样块：输入小特征(query)和大特征(kv)，先cross-attention融合，再经过两个InceptionLightBlock。
    """
    def __init__(self, in_channels, out_channels, num_heads=8, attn_drop=0., proj_drop=0., activation=nn.ReLU):
        super().__init__()
        self.upsample2x = nn.UpsamplingBilinear2d(scale_factor=2)
        self.cross_attn = CrossAttention(dim=in_channels, num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop)
        self.conv1 = InceptionLightBlock(in_channels, out_channels, activation=activation)
        self.conv2 = InceptionLightBlock(out_channels, out_channels, activation=activation)

    def forward(self, x, prompt):
        x = self.upsample2x(x)
        # cross-attention融合
        fused = self.cross_attn(prompt, x)
        # 两个InceptionLightBlock
        x = self.conv1(fused)
        x = self.conv2(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels=1, activation=nn.LeakyReLU):
        super(Encoder, self).__init__()
        backbone = resnet34(pretrained=True)
        self.prepare_layer = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            activation(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        xs = []
        x = self.prepare_layer(x) # (64, 224, 224)
        x = self.layer1(x) # (64, 112, 112)
        xs.append(x)
        x = self.layer2(x) # (128, 56, 56)
        xs.append(x)
        x = self.layer3(x) # (256, 28, 28)
        xs.append(x)
        x = self.layer4(x) # (512, 14, 14)
        xs.append(x)
        xs.reverse()
        return xs

class PreMaskEncoder(nn.Module):
    """
    PreMaskEncoder using InceptionLightBlocks.
    This encoder processes the input and generates a feature map.
    """
    def __init__(self, n_classes=1, activation=nn.LeakyReLU):
        super(PreMaskEncoder, self).__init__()
        self.conv1 = DownConv3d(n_classes, 64, activation=activation)
        self.conv2 = DownConv3d(64, 128, activation=activation)
        self.conv3 = DownConv3d(128, 256, activation=activation)
        self.conv4 = DownConv3d(256, 512, activation=activation)
        self.attn = TripleAttention3D(512)
        self.conv = nn.Conv3d(512, 512, kernel_size=(2, 1, 1), bias=False)

    def forward(self, x: torch.Tensor):
        # (B, C, D, H, W), D is 8x
        while x.size(-3) < 16:
            x = torch.cat([x, x], dim=-3)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.attn(x)

        avg_x = torch.mean(x, dim=-3, keepdim=True) 
        max_x = torch.max(x, dim=-3, keepdim=True)[0]
        x = self.conv(torch.cat([avg_x, max_x], dim=-3))
        x = x.squeeze(-3) # (B, C, H, W)
        return x

class Decoder(nn.Module):
    def __init__(self, n_classes, num_heads=8, attn_dropout=0.3, proj_drop=0.2, activation=nn.ReLU):
        super(Decoder, self).__init__()

        self.up_block1 = UpBlockDecoder(512, 256, num_heads=num_heads, attn_drop=attn_dropout, proj_drop=proj_drop, activation=activation)
        self.up_block2 = UpBlockDecoder(256, 128, num_heads=num_heads, attn_drop=attn_dropout, proj_drop=proj_drop, activation=activation)
        self.up_block3 = UpBlockDecoder(128, 64, num_heads=num_heads, attn_drop=attn_dropout, proj_drop=proj_drop, activation=activation)
        self.up_block4 = UpBlockDecoder(64, 64, num_heads=num_heads, attn_drop=attn_dropout, proj_drop=proj_drop, activation=activation)
        self.head = nn.Conv2d(64, n_classes, kernel_size=1)

        self.mask_upers = nn.ModuleList([
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, bias=False),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, bias=False),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=False),
        ])

    def forward(self, x: Sequence[torch.Tensor], mask_prompt: torch.Tensor):
        prompts = self.fuse(x, mask_prompt)
        y = x[0]
        y = self.up_block1(y, prompts[0])
        y = self.up_block2(y, prompts[1])
        y = self.up_block3(y, prompts[2])
        y = self.up_block4(y, prompts[3])
        x = y
        x = self.head(x)
        return x

    def fuse(self, xs: Sequence[torch.Tensor], mask: torch.Tensor):
        x = xs[0] + mask
        y = [x]
        for x, mask_uper in zip(xs[1:], self.mask_upers):
            mask = mask_uper(mask)
            x = x + mask
            y.append(x)
        return y

class Model(nn.Module):
    """
    主模型，包含 Encoder、PreMaskEncoder、Decoder。
    输入: 图像x, mask_prompt
    输出: mask 预测
    """
    def __init__(self, in_channels=1, n_classes=1, num_heads=8, attn_dropout=0.3, proj_drop=0.2, activation=(nn.ReLU, nn.LeakyReLU)):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, activation=activation[0])
        self.premask_encoder = PreMaskEncoder(n_classes=n_classes, activation=activation[1])
        self.decoder = Decoder(n_classes=n_classes, num_heads=num_heads, attn_dropout=attn_dropout, proj_drop=proj_drop, activation=activation[1])

    def forward(self, x, masks):
        # x: (B, in_channels, H, W)
        # masks: (B, n_classes, D, H, W)  (D为mask深度)
        enc_feats = self.encoder(x)  # x1, x2, x3, x4
        mask_feat = self.premask_encoder(masks)  # (B, C, H, W)
        out = self.decoder(enc_feats, mask_feat)
        return out
