import torch
import torch.nn as nn
from typing import List, Dict, Union, Any, cast


class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(arch: str, cfg: str, batch_norm: bool, num_classes: int, **kwargs: Any) -> VGG:
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), num_classes=num_classes, **kwargs)
    return model


def vgg11(num_classes: int = 1000, **kwargs: Any) -> VGG:
    return _vgg("vgg11", "A", False, num_classes, **kwargs)


def vgg11_bn(num_classes: int = 1000, **kwargs: Any) -> VGG:
    return _vgg("vgg11_bn", "A", True, num_classes, **kwargs)


def vgg13(num_classes: int = 1000, **kwargs: Any) -> VGG:
    return _vgg("vgg13", "B", False, num_classes, **kwargs)


def vgg13_bn(num_classes: int = 1000, **kwargs: Any) -> VGG:
    return _vgg("vgg13_bn", "B", True, num_classes, **kwargs)


def vgg16(num_classes: int = 1000, **kwargs: Any) -> VGG:
    return _vgg("vgg16", "D", False, num_classes, **kwargs)


def vgg16_bn(num_classes: int = 1000, **kwargs: Any) -> VGG:
    return _vgg("vgg16_bn", "D", True, num_classes, **kwargs)


def vgg19(num_classes: int = 1000, **kwargs: Any) -> VGG:
    return _vgg("vgg19", "E", False, num_classes, **kwargs)


def vgg19_bn(num_classes: int = 1000, **kwargs: Any) -> VGG:
    return _vgg("vgg19_bn", "E", True, num_classes, **kwargs)
