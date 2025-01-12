# TODO: Test
import torch
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
    def __init__(self, in_channel, out_channel, smooth_channel_variance=False):
        super(AttnConv, self).__init__()

        if smooth_channel_variance:
            mid_channel = (in_channel + out_channel) // 2
            self.conv1 = Conv(in_channel, mid_channel, 3)
            self.conv2 = Conv(mid_channel, out_channel, 3)
            self.attn = CBAM(mid_channel, 2)
        else:
            self.conv1 = Conv(in_channel, out_channel, 3)
            self.conv2 = Conv(out_channel, out_channel, 3)
            self.attn = CBAM(out_channel, 2)
        # self.norm1 = nn.BatchNorm2d(in_channels)

        # in encoder, use for extracting higher features
        # in decoder, use for combining and studying between features such more info than that being unused

    def forward(self, x):
        x = self.conv1(x)
        x = x + self.conv2(self.attn(x))
        # x = self.norm1(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, n_iter, in_channels):
        super(Bottleneck, self).__init__()
        self.n_iter = n_iter

        self.attn_conv = AttnConv(in_channels, in_channels) # or PSP

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
        return x, y # x for next layer and y for skip connection

class DecoderLayer(nn.Module):
    def __init__(self, n_iter, in_channels, out_channels, bilinear=False):
        super(DecoderLayer, self).__init__()

        self.n_iter = n_iter

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if bilinear else nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.attn_conv = AttnConv(in_channels, out_channels)

    def forward(self, x, y):
        x = self.up(x)

        diffX = torch.tensor([y.size()[-1] - y.size()[-1]])
        diffY = torch.tensor([y.size()[-2] - y.size()[-2]])

        x = F.pad(x, [torch.div(diffX, 2, rounding_mode='trunc'),
                      torch.div(diffX - diffX, 2, rounding_mode='trunc'),
                      torch.div(diffY, 2, rounding_mode='trunc'),
                      torch.div(diffY - diffY, 2, rounding_mode='trunc')])

        x = torch.cat([y, x], dim=1)
        for _ in range(self.n_iter):
            x = self.attn_conv(x)
        return x

class Encoder(nn.Module):
    def __init__(self, n_iter, in_channels, out_channels):
        super(Encoder, self).__init__()

        self.n_layer = len(in_channels)

        self.layers = [EncoderLayer(n_iter, ic, oc) for (ic, oc) in zip(in_channels, out_channels)]

    def forward(self, x):
        ys = []
        for layer in self.layers:
            x, y = layer(x)
            ys.append(y)
        ys.reverse()
        return x, ys

class Decoder(nn.Module):
    def __init__(self, n_iter, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.n_layer = len(in_channels)

        self.layers = [DecoderLayer(n_iter, ic, oc) for ic, oc in zip(in_channels, out_channels)]

    def forward(self, x, ys):
        for (layer, y) in zip(self.layers, ys):
            # 2nd noval point
            # y = skip_connection(y)
            x = layer(x, y)
        return x

class Head(nn.Module):
    def __init__(self, n_channels, n_classes, *, n_unet:int=1):
        super(Head, self).__init__()

        self.n_unet = n_unet
        self.conv = nn.Conv2d(n_channels, n_classes, 1)
        self.weights = nn.Parameter(torch.randn(n_unet))

    def forward(self, ys: list[torch.Tensor]):
        device = ys[0].device
        x = torch.zeros_like(ys[0]).to(device)
        self.weights = self.weights.to(device)
        for (y, weight) in zip(ys, self.weights):
            y += x * weight
        x = self.conv(x)
        return x

class MyUNet(nn.Module):
    def __init__(self, n_iters: tuple[int, int, int], n_channels: int, n_classes: int,
                 encoder_channels: tuple[list, list]=([32, 64, 128, 256], [64, 128, 256, 512]),
                 bottleneck_channels: int=512,
                 decoder_channels: tuple[list, list]=([1024, 512, 256, 128], [256, 128, 64, 32]),
                 *, n_unet: int=1):
        super(MyUNet, self).__init__()
        self.n_unet = n_unet

        self.first = AttnConv(n_channels, encoder_channels[0][0])
        self.recursive_first = AttnConv(encoder_channels[0][0], encoder_channels[0][0])
        self.encoder = Encoder(n_iters[0], in_channels=encoder_channels[0], out_channels=encoder_channels[1])
        self.neck = Bottleneck(n_iters[1], in_channels=bottleneck_channels)
        self.decoder = Decoder(n_iters[2], in_channels=decoder_channels[0], out_channels=decoder_channels[1])
        self.head = Head(decoder_channels[1][-1], n_classes)

    def forward(self, x):
        outs = []
        for i in range(self.n_unet):
            if i == 0:
                x = self.first(x)
            else:
                x = self.recursive_first(x)
            x, ys = self.encoder(x) # ys is features of each skip connection, x is to input to bottleneck
            x = self.neck(x)
            x = self.decoder(x, ys)
            outs.append(x)
        return self.head(outs)


from torchsummary import summary
if __name__ == '__main__':
    x = torch.randn((1, 1, 512, 512)).cuda()
    net = MyUNet((1, 3, 1), 1, 10, n_unet=3).cuda()
    y = net(x)
    print(y.shape)
    # summary(net, input_size=(1, 512, 512))
