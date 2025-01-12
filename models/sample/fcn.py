"""
    Fully Convolutional Networks for Semantic Segmentation
    https://arxiv.org/abs/1411.4038
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FCN32s(nn.Module):
    def __init__(self, num_classes):
        super(FCN32s, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        features = list(vgg16.features.children())
        
        # 编码器部分
        self.encoder = nn.Sequential(*features[:30])  # conv5 之后的特征图
        
        # 1x1 卷积调整输出通道数
        self.conv_1x1 = nn.Conv2d(512, num_classes, kernel_size=1)
        
        # 上采样层，32倍上采样恢复原始分辨率
        self.upsample32x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, padding=16, bias=False)

    def forward(self, x):
        # 提取特征
        features = self.encoder(x)
        
        # 1x1 卷积调整通道数
        x = self.conv_1x1(features)
        
        # 32 倍上采样
        x = self.upsample32x(x)
        
        return x

class FCN16s(nn.Module):
    def __init__(self, num_classes):
        super(FCN16s, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        features = list(vgg16.features.children())
        
        # 编码器部分
        self.encoder = nn.Sequential(*features[:30])  # conv5 之后的特征图
        self.conv4 = nn.Sequential(*features[:24])    # conv4 之后的特征图
        
        # 1x1 卷积调整输出通道数
        self.conv_1x1 = nn.Conv2d(512, num_classes, kernel_size=1)
        
        # 上采样部分

    def forward(self, x):
        # 提取特征
        conv4 = self.conv4(x)
        conv5 = self.encoder(conv4)
        
        # 1x1 卷积调整通道数
        x = self.conv_1x1(conv5)
        
        # 2倍上采样 + 跳跃连接
        x = self.upsample2x(x)
        x = x + self.conv_1x1(conv4)  # 将 conv4 的特征与上采样结果相加
        
        # 16倍上采样恢复原始分辨率
        x = self.upsample16x(x)
        
        return x


class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super(FCN8s, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        features = list(vgg16.features.children())
        
        # 编码器部分
        self.encoder = nn.Sequential(*features[:30])  # conv5 之后的特征图
        self.conv4 = nn.Sequential(*features[:24])    # conv4 之后的特征图
        self.conv3 = nn.Sequential(*features[:17])    # conv3 之后的特征图
        
        # 1x1 卷积调整输出通道数
        self.conv_1x1_5 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.conv_1x1_4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.conv_1x1_3 = nn.Conv2d(256, num_classes, kernel_size=1)
        
        # 上采样部分
        self.upsample2x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsample4x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=8, stride=4, padding=2, bias=False)
        self.upsample8x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, padding=4, bias=False)

    def forward(self, x):
        # 提取特征
        conv3 = self.conv3(x)
        conv4 = self.conv4(conv3)
        conv5 = self.encoder(conv4)
        
        # 1x1 卷积调整通道数
        x = self.conv_1x1_5(conv5)
        
        # 2倍上采样 + 跳跃连接 (conv4)
        x = self.upsample2x(x)
        x = x + self.conv_1x1_4(conv4)
        
        # 2倍上采样 + 跳跃连接 (conv3)
        x = self.upsample2x(x)
        x = x + self.conv_1x1_3(conv3)
        
        # 最后上采样恢复原始分辨率
        x = self.upsample8x(x)
        
        return x

