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
