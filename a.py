import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from src.models.like.test import Model
from torchvision.models import resnet50, resnet34

depth = 144
prompt_depth = 8
size = (224, 224)
batch = 2
n_channels = 3
n_classes = 1
in_shape = (batch, n_channels, depth, *size)

x = torch.randn(in_shape)
model = Model(n_channels=n_channels, n_classes=n_classes, depth_size=prompt_depth)

if torch.cuda.is_available():
    model = model.cuda()
    x = x.cuda()

y = model(x)
summary(model, input_size=in_shape)
print(y.shape)
