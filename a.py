import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet34
from utils.util import model_info
from pathlib import Path

size = (224, 224)
batch = 2
n_channels = 64
n_classes = 1
in_shape = (batch, n_channels, *size)

x = torch.randn(in_shape)
from models.like.sconv import SConv2d
model = SConv2d(channels=[64, 128, 256], kernel_sizes=[13, 3, 3], r=4, act=nn.ReLU)

if torch.cuda.is_available():
    model = model.cuda()
    x = x.cuda()

y = model(x)
# Fix call signature: model_info(output_dir, model, input_sizes, ...)
model_info(Path('.'), model, in_shape)
# Handle tuple output from SConv2d
if isinstance(y, tuple):
    print(y[0].shape)
else:
    print(y.shape)
