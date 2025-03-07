import numpy as np
import torch
from PIL import Image
import cv2
from pathlib import Path

def to_rgb(filename: Path|str, use_opencv=False):
    if use_opencv:
        x = cv2.imread(filename)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    else:
        x = Image.open(filename).convert('RGB')
    return x

def to_gray(filename: Path|str, use_opencv=False):
    if use_opencv:
        x = cv2.imread(filename)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    else:
        x = Image.open(filename).convert('L')
    return x


def image_transform(x: Image.Image|cv2.Mat, size: tuple[int, int], is_rgb=False, *, dtype=np.float32):    
    x = x.resize(size)
    x = np.array(x, dtype=dtype)

    tensor_x = torch.tensor(x)
    if is_rgb:
        x = tensor_x.permute(2, 1, 0)
    else:
        x = tensor_x.permute(1, 0).unsqueeze(0)

    if x.max() > 1:
        x /= 255
    return x


from torchvision import transforms
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
    def random_invert(self, p):
        self._transforms.append(transforms.RandomInvert(p))
        return self

    def PIL_to_tensor(self, dtype=np.float32):
        self._transforms.append(transforms.PILToTensor())
        self._transforms.append(transforms.ConvertImageDtype(dtype))
        return self
    def to_tensor(self):
        self._transforms.append(transforms.ToTensor())
        return self
    def to_pil_image(self):
        self._transforms.append(transforms.ToPILImage())
        return self

    def normalize(self, is_rgb: bool):
        if is_rgb:
            self._transforms.append(
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        else:
            self._transforms.append(
                transforms.Normalize(mean=[0.5], std=[0.5]))
        return self

    def build(self):
        return transforms.Compose(self._transforms)

def image_transforms(resize: tuple[int, int]|None=None, 
                     hflip_p: float|None=None, vflip_p: float|None=None, 
                     rotation: float|None=None, 
                     is_rgb: bool=False, is_PIL_image: bool=False) -> transforms.Compose:
    transforms = VisionTransformersBuilder()
    if resize:
        transforms = transforms.resize(resize)
    if hflip_p:
        transforms = transforms.random_horizontal_flip(hflip_p)
    if vflip_p:
        transforms = transforms.random_vertical_flip(vflip_p)
    if rotation:
        transforms = transforms.random_rotation(rotation)

    if is_PIL_image:
        transforms = transforms.PIL_to_tensor()
    else:
        transforms = transforms.to_tensor()
    transforms = transforms.normalize(is_rgb)
    return transforms.build()


if __name__ == '__main__':
    filename = r'E:\Program\Projects\py\lab\NeuroTrain\data\DRIVE\training\images\21.png'
    image = cv2.imread(filename)
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print(gray_image.shape)
    # x = np.array(gray_image, dtype=np.float32)
    # tensor_x = torch.tensor(x).permute(1, 0).unsqueeze(0)
    # print(tensor_x.shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image.shape)
    x = np.array(image, dtype=np.float32)
    tensor_x = torch.tensor(x).permute(2, 1, 0)
    print(tensor_x.shape)
