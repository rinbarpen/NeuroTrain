import torch
from torch import nn

class ChannelAttention(nn.Module):
    """
        Squeeze-and-Excitation Networks
        http://arxiv.org/abs/1709.01507
    """
    def __init__(self, channels, reduction: int):
        super(ChannelAttention, self).__init__()

        self.channels = channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, _, _ = x.size()
        scale = self.avg_pool(x).view(B, C)
        scale = self.fc(scale).view(B, C, 1, 1)

        return x * scale.expand_as(x)

class EnhancedChannelAttention(nn.Module):
    """
        ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
        http://arxiv.org/abs/1910.03151
    """
    def __init__(self, kernel_size: int=7):
        super(EnhancedChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        scale = self.sigmoid(y)

        return x * scale.expand_as(x)


if __name__ == '__main__':
    eca = EnhancedChannelAttention()
    input_tensor = torch.randn(2, 16, 64, 64)
    output_tensor = eca(input_tensor)
    print('input:', input_tensor.size(), 'output:', output_tensor.size())
