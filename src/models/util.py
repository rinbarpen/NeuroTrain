import torch
from torch import nn
from typing import Type

def stack_conv(n: int, conv: Type[nn.Module], bn: Type[nn.Module] = nn.BatchNorm2d, act: Type[nn.Module] = nn.ReLU, with_bn: bool = True, with_act: bool = True, **kwargs):
    assert n >= 1, "n must be greater than or equal to 1"
    modules = [conv(**kwargs)] * n
    if with_bn:
        modules.append(bn(modules[-1].out_channels))
    if with_act:
        modules.append(act())
    return nn.Sequential(*modules)

def chunk(m: Type[nn.Module], n: int=1, **kwargs) -> nn.Sequential:
    return nn.Sequential(*[m(**kwargs) for _ in range(n)])
