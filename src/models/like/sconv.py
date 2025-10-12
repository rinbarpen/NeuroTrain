from re import S
from torch import nn
import torch
import torch.nn.functional as F
from ..conv.DWConv import get_dwconv, get_dwconv_layer2d, DepthWiseConv2d, SeparableConv2d
from ..attention.ECAModule import ECAModule
from ..attention.SEModule import SEModule
from ..attention.CAModule import CAModule
from ..attention.CBAM import CBAM
from ..attention.MultiHeadAttention import MultiHeadAttention, MultiHeadCrossAttention
from ..transformer.fused_transformer import VisionTransformerEncoder, CrossVisionTransformerEncoder
from ..transformer.base_transformers import TransformerEncoder, TransformerDecoder
from ..transformer.base_vit import BaseViT, CrossViT


# input tensor has been through batch norm, it is a tensor with stable distribution
# [64, 128, 256] is smooth version
# [64, 128, 512] is fast version
class SConv2d(nn.Module):
    def __init__(self, channels=[64, 128, 256], kernel_sizes=[13, 3, 3], r=4, act=nn.ReLU):
        super(SConv2d, self).__init__()
        self.bconv = nn.Conv2d(channels[0], channels[0], kernel_sizes[0], padding=kernel_sizes[0]//2)
        self.battn = ECAModule(channels[0])
        self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=1)
        self.bn1 = nn.BatchNorm2d(channels[1])
        self.act1 = act()
        
        self.mconv = nn.Sequential(
            nn.Conv2d(channels[1], channels[1], kernel_sizes[1], padding=kernel_sizes[1]//2, bias=False, groups=channels[1]//r),
            nn.GroupNorm(channels[1]//r, channels[1]),
            act(),
        )
        self.mattn = ECAModule(channels[1])
        self.conv2 = nn.Conv2d(channels[1], channels[2], kernel_size=1)
        self.bn2 = nn.BatchNorm2d(channels[2])
        self.act2 = act()

        self.tconv = nn.Conv2d(channels[2], channels[2], kernel_sizes[2], padding=kernel_sizes[2]//2)
        self.tattn = ECAModule(channels[2])
        self.o_conv = nn.Conv2d(sum(channels), channels[0], kernel_size=1)

    def forward(self, x):
        x = self.bconv(x)
        global_feats = x = self.battn(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.mconv(x)
        obj_feats = x = self.mattn(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.tconv(x)
        local_feats = x = self.tattn(x)
        x = self.o_conv(torch.cat([global_feats, obj_feats, local_feats], dim=1))
        return x, (global_feats, obj_feats, local_feats)

if __name__ == '__main__':
    config = {
        "channels": [64, 128, 256],
        "kernel_sizes": [13, 3, 3], # [7, 3, 3], [15, 5, 3]
        "r": 4,
        "act": nn.ReLU,
    }
    input = torch.randn(1, 64, 32, 32)
    model = SConv2d(**config)
    output = model(input)
    print(output.shape)
