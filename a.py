import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from src.models.like.net3d.model import Model

depth = 8
size = (224, 224)
batch = 2
n_channels = 1
n_classes = 1
in_shape = (batch, n_channels, *size)
mask_shape = (batch, n_channels, depth, *size)

x = torch.randn(in_shape)
mask = torch.randn(mask_shape)
model = Model(in_channels=n_channels, n_classes=n_classes)

if torch.cuda.is_available():
    model = model.cuda()
    x = x.cuda()
    mask = mask.cuda()

y = model(x, mask)
# summary(model, input_size=(n_channels, *size))
print(y.shape)
