import torch
from alembic.runtime.migration import HeadMaintainer
from torch import nn
import torch.functional as F

from models.attention.fused_attention import CBAM

# TODO: an experiment to identity whether attention or conv while raising to the number of channels
# TODO: only encoder; only decoder; encoder and decoder


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super(Conv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=kernel_size//2, dilation=dilation, groups=groups, bias=bias)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

# some AttnConv is Attn and then Conv | Conv and then Conv | DWConv and then Conv
class AttnConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttnConv, self).__init__()

        self.conv1 = Conv(in_channels, in_channels, 3)
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.conv2 = Conv(in_channels, out_channels, 3)
        self.norm2 = nn.BatchNorm2d(out_channels)

        self.attn = CBAM(out_channels, 2)

    def forward(self, x):
        x = self.conv1(x)
        # x = x + self.attn(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, n_iter, in_channels):
        super(Bottleneck, self).__init__()
        self.n_iter = n_iter

        self.attn_conv = AttnConv(in_channels, in_channels)

    def forward(self, x):
        for _ in range(self.n_iter):
            x = self.attn_conv(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, n_iter, in_channels, out_channels):
        super(EncoderLayer, self).__init__()

        self.n_iter = n_iter

        self.attn_conv = AttnConv(in_channels, out_channels)
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        for _ in range(self.n_iter):
            x = self.attn_conv(x)
        y = x
        x = self.max_pool(x)
        return x, y

class Encoder(nn.Module):
    def __init__(self, n_iter, in_channels=[64, 128, 256, 512]):
        super(Encoder, self).__init__()

        self.n_layer = len(in_channels)

        self.layers = [EncoderLayer(n_iter, c, c * 2) for c in in_channels]

    def forward(self, x):
        ys = []
        for layer in self.layers:
            x, y = layer(x)
            ys.append(y)
        ys.reverse()
        return ys, x

class DecoderLayer(nn.Module):
    def __init__(self, n_iter, in_channels, out_channels, bilinear=False):
        super(DecoderLayer, self).__init__()

        self.n_iter = n_iter

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if bilinear else nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.attn_conv = AttnConv(in_channels * 2, out_channels)

    def forward(self, x, y):
        x = self.up(x)
        x = torch.cat([y, x], dim=1)
        for _ in range(self.n_iter):
            x = self.attn_conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, n_iter, in_channels=[1024, 512, 256, 128]):
        super(Decoder, self).__init__()
        self.n_layer = len(in_channels)

        self.layers = [DecoderLayer(n_iter, c, c // 2) for c in in_channels]

    def forward(self, x, ys):
        for (layer, y) in zip(self.layers, ys):
            # 2nd noval point
            # y = skip_connection(y)
            x = layer(x, y)
        return x

class Head(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Head, self).__init__()

        self.conv = nn.Conv2d(n_channels, n_classes, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class MyUNet(nn.Module):
    def __init__(self, n_iter, n_channels, n_classes,
                 encoder_channels=[64, 128, 256, 512],
                 bottleneck_channels=1024,
                 decoder_channels=[1024, 512, 256, 128]):
        super(MyUNet, self).__init__()

        self.conv = nn.Conv2d(n_channels, encoder_channels[0], kernel_size=1, bias=False)
        self.encoder = Encoder(n_iter, in_channels=encoder_channels)
        self.neck = Bottleneck(n_iter, in_channels=bottleneck_channels)
        self.decoder = Decoder(n_iter, in_channels=decoder_channels)
        self.head = Head(decoder_channels[-1]//2, n_classes)

    def forward(self, x):
        x = self.conv(x)
        ys, x = self.encoder(x)
        x = self.neck(x)
        x = self.decoder(x, ys)
        x = self.head(x)
        return x


from torchsummary import summary
if __name__ == '__main__':
    x = torch.randn((1, 1, 512, 512)).cuda()
    net = MyUNet(1, 1, 10).cuda()
    y = net(x)
    print(y.shape)
    summary(net, input_size=(1, 512, 512))
