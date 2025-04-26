import numpy as np
import torch
from PIL import Image
import cv2
from pathlib import Path
from torchvision import transforms
from config import get_config

from utils.typed import FilePath

def to_rgb(filename: FilePath, use_opencv=False):
    if use_opencv:
        x = cv2.imread(filename)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    else:
        x = Image.open(filename).convert('RGB')
    return x

def to_gray(filename: FilePath, use_opencv=False):
    if use_opencv:
        x = cv2.imread(filename)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    else:
        x = Image.open(filename).convert('L')
    return x

# VisionTransformersBuilder may be flawed
class VisionTransformersBuilder:
    def __init__(self):
        self._transforms = []

    def resize(self, size: tuple[int, int]):
        self._transforms.append(transforms.Resize(size))
        return self
    def crop(self, size: tuple[int, int]):
        self._transforms.append(transforms.CenterCrop(size))
        return self

    def random_rotation(self, degrees: float):
        self._transforms.append(transforms.RandomRotation(degrees=degrees))
        return self
    def random_horizontal_flip(self, p: float):
        self._transforms.append(transforms.RandomHorizontalFlip(p=p))
        return self
    def random_vertical_flip(self, p: float):
        self._transforms.append(transforms.RandomVerticalFlip(p=p))
        return self
    def random_invert(self, p: float):
        self._transforms.append(transforms.RandomInvert(p=p))
        return self

    def PIL_to_tensor(self):
        self._transforms.append(transforms.PILToTensor())
        return self
    def to_tensor(self):
        self._transforms.append(transforms.ToTensor())
        return self
    def to_pil_image(self):
        self._transforms.append(transforms.ToPILImage())
        return self
    def convert_image_dtype(self, ttype: torch.dtype=torch.float32):
        self._transforms.append(transforms.ConvertImageDtype(ttype))
        return self

    def normalize(self, is_rgb: bool):
        if is_rgb:
            self._transforms.append(
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        else:
            self._transforms.append(
                transforms.Normalize(mean=[0.5], std=[0.5]))
        return self

    def build(self) -> transforms.Compose:
        return transforms.Compose(self._transforms)

def get_transforms() -> transforms.Compose:
    c = get_config()
    builder = VisionTransformersBuilder()
    for k, v in c['transform'].items():
        match k.upper():
            case 'RESIZE':
                builder = builder.resize(tuple(v))
            case 'HFLIP':
                builder = builder.random_horizontal_flip(*v)
            case 'VFLIP':
                builder = builder.random_vertical_flip(*v)
            case 'ROTATION':
                builder = builder.random_rotation(*v)
            case 'INVERT':
                builder = builder.random_invert(*v)
            case 'CROP':
                builder = builder.crop(tuple(v))
            case 'NORMALIZE':
                builder = builder.normalize(*v)
            case 'TO_TENSOR':
                builder = builder.to_tensor()
            case 'PIL_TO_TENSOR':
                builder = builder.PIL_to_tensor()
            case 'CONVERT_IMAGE_DTYPE':
                match v[0]:
                    case 'float16':
                        ttype = torch.float16
                    case 'float32':
                        ttype = torch.float32
                    case 'float64':
                        ttype = torch.float64
                    case 'bfloat16':
                        ttype = torch.bfloat16
                    case 'uint8':
                        ttype = torch.uint8
                    case 'uint16':
                        ttype = torch.uint16
                    case 'uint32':
                        ttype = torch.uint32
                    case 'uint64':
                        ttype = torch.uint64
                    case 'int8':
                        ttype = torch.int8
                    case 'int16':
                        ttype = torch.int16
                    case 'int32':
                        ttype = torch.int32
                    case 'int64':
                        ttype = torch.int64
                builder = builder.convert_image_dtype(ttype)
    return builder.build()

def build_image_transforms(resize: tuple[int, int]|None=None, 
                           crop: tuple[int, int]|None=None, 
                           rotation: float|None=None, 
                           vflip: float|None=None,
                           hflip: float|None=None,
                           invert: float|None=None,
                           is_pil_image: bool=False,
                           ttype: torch.dtype=torch.float32,
                           norm: bool=False,
                           is_rgb: bool=False, **kwargs) -> transforms.Compose:
    builder = VisionTransformersBuilder()
    if resize: 
        builder = builder.resize(resize)
    if crop: 
        builder = builder.crop(crop)
    if rotation: 
        builder = builder.random_rotation(rotation)
    if vflip: 
        builder = builder.random_vertical_flip(vflip)
    if hflip: 
        builder = builder.random_horizontal_flip(vflip)
    if invert: 
        builder = builder.random_invert(invert)
    if is_pil_image: 
        builder = builder.PIL_to_tensor()
        builder = builder.convert_image_dtype(ttype)
    else:
        builder = builder.to_tensor()
    if norm:
        builder = builder.normalize(is_rgb)

    return builder.build()
