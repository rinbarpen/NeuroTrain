"""
    FPN: Feature Pyramid Networks
    https://arxiv.org/abs/1612.03144
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class FPN(nn.Module):
    def __init__(self, backbone=resnet50(pretrained=True), out_channels=256):
        super(FPN, self).__init__()
        
        self.backbone = backbone
        self.backbone_layers = list(backbone.children())
        
        self.layer1 = nn.Sequential(*self.backbone_layers[:5])  # C2
        self.layer2 = nn.Sequential(*self.backbone_layers[5])   # C3
        self.layer3 = nn.Sequential(*self.backbone_layers[6])   # C4
        self.layer4 = nn.Sequential(*self.backbone_layers[7])   # C5

        # 1x1 卷积减少通道数
        self.lateral4 = nn.Conv2d(2048, out_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(1024, out_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(512, out_channels, kernel_size=1)

        # 3x3 卷积平滑输出特征图
        self.smooth4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # 自底向上
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # 自顶向下
        p5 = self.lateral4(c5)
        p4 = self.lateral3(c4) + F.interpolate(p5, scale_factor=2, mode='nearest')
        p3 = self.lateral2(c3) + F.interpolate(p4, scale_factor=2, mode='nearest')

        # 平滑操作
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p3)

        return p2, p3, p4, p5

if __name__ == '__main__':
    # 测试模型
    x = torch.randn(1, 3, 224, 224)
    model = FPN()
    outputs = model(x)
    for out in outputs:
        print(out.shape)  # 输出各个金字塔特征层的尺寸
