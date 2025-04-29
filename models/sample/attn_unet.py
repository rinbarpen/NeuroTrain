import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    def __init__(self, in_channels_x, in_channels_g, inter_channels):  # inter_channels 是中间层通道数
        super(AttentionGate, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(in_channels_g, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels_x, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):  # x: encoder output, g: decoder output
        # Gating signal path
        g1 = self.W_g(g)
        # Encoder signal path
        x1 = self.W_x(x)

        # Combined signal
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        # Return attended feature map
        return x * psi
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2): # x1: decoder input  (来自上一层)  x2: encoder output (跳跃连接)
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class AttentionUnet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(AttentionUnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.attn_gate1 = AttentionGate(in_channels_x=512//factor, in_channels_g=512//factor, inter_channels=256//factor) # Attention Gate
        self.up2 = Up(512 // factor, 256 // factor, bilinear)
        self.attn_gate2 = AttentionGate(in_channels_x=256//factor, in_channels_g=256//factor, inter_channels=128//factor) # Attention Gate
        self.up3 = Up(256 // factor, 128 // factor, bilinear)
        self.attn_gate3 = AttentionGate(in_channels_x=128//factor, in_channels_g= 128//factor, inter_channels=64//factor) # Attention Gate
        self.up4 = Up(128 // factor, 64, bilinear)

        self.outc = OutConv(64, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x4_attn = self.attn_gate1(x4, x) # Apply attention gate
        x = self.up2(x, x3)
        x3_attn = self.attn_gate2(x3, x)  # Apply attention gate
        x = self.up3(x, x2)
        x2_attn = self.attn_gate3(x2, x)  # Apply attention gate
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
