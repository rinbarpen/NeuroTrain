import logging
import colorlog
import os.path
import os
import time
from time import strftime

import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch import nn
from torchsummary import summary
from pprint import pprint
from typing import TextIO

from utils.dataset import drive_dataset


def prepare_logger():
    log_colors = {
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'FATAL': 'bold_red',
    }
    formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s %(levelname)s %(name)s | %(message)s',
        log_colors=log_colors
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File handler
    os.makedirs('logs', exist_ok=True)
    filename = os.path.join('logs', strftime('%Y%m%d_%H%M%S.log', time.localtime()))
    file_handler = logging.FileHandler(filename, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s %(name)s | %(message)s'
    ))

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def save_model(path: Path, model: nn.Module,
               optimizer=None, lr_scheduler=None, scaler=None, **kwargs):
    checkpoint = dict()
    checkpoint['model'] = model.state_dict()
    if optimizer:
        checkpoint["optimizer"] = optimizer.state_dict()
    if lr_scheduler:
        checkpoint["lr_scheduler"] = lr_scheduler.state_dict()
    if scaler:
        checkpoint["scaler"] = scaler.state_dict()
    for k, v in kwargs.items():
        checkpoint[k] = v

    try:
        torch.save(checkpoint, path)
    except FileExistsError as e:
        path = path.parent / (path.stem +
                              strftime("%Y%m%d_%H%M%S", time.localtime()))
        torch.save(checkpoint, path)


def load_model(path: Path, map_location: str = 'cuda'):
    return torch.load(path, map_location=torch.device(map_location))

def save_model_to_onnx(path: Path, model: nn.Module, input_size: tuple):
    dummy_input = torch.randn(input_size)
    torch.onnx.export(model, dummy_input, path)

def save_model_safe(path: Path, model: nn.Module, optimizer=None, lr_scheduler=None, scaler=None, **kwargs):
    try:
        save_model(path, model, optimizer, lr_scheduler, scaler, **kwargs)
    except FileNotFoundError as e:
        path.parent.mkdir()
        save_model(path, model, optimizer, lr_scheduler, scaler, **kwargs)

def print_model_info(model_src: Path, output_stream: TextIO):
    checkpoint = load_model(model_src, "cpu")
    pprint(checkpoint, stream=output_stream)


def summary_model_info(model_src: Path | torch.nn.Module, input_size: torch.Tensor):
    if isinstance(model_src, Path):
        checkpoint = load_model(model_src, "cpu")
        summary(checkpoint['model'], input_size=input_size)
    elif isinstance(model_src, torch.nn.Module):
        summary(model_src, input_size=input_size)


def save_numpy_data(path: Path, data: np.ndarray | torch.Tensor):
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()

    try:
        np.save(path, data)
    except FileNotFoundError as e:
        path.parent.mkdir()
        np.save(path, data)


def load_numpy_data(path: Path):
    try:
        data = np.load(path)
        return data
    except FileNotFoundError as e:
        colorlog.error(f'File is not found: {e}')
        raise e


def tuple2list(t: tuple):
    return list(t)


def list2tuple(l: list):
    return tuple(l)


def get_train_valid_dataset(dataset_name: str, base_dir: Path):
    match dataset_name.lower():
        case 'drive':
            return drive_dataset.get_drive_train_valid_dataset(base_dir, 1.0)
    
    return None, None

def get_test_dataset(dataset_name: str, base_dir: Path):
    match dataset_name.lower():
        case 'drive':
            return drive_dataset.get_drive_test_dataset(base_dir)

    return None, None

def get_model(model_name: str, config: dict):
    match model_name:
        case 'unet_neck':
            from models.like.unet_neck import UNet
            model = UNet(config['n_channels'], config['n_classes'], bilinear=False)
            return model
        case 'unet':
            from models.sample.unet import UNet
            model = UNet(config['n_channels'], config['n_classes'], bilinear=False)
            return model

    return None


def time_cost(f):
    def wrapper(*args, **kwargs):
        begin = time.time_ns()
        result = f(*args, **kwargs)
        end = time.time_ns()
        print(f"Function {f.__name__} took {end - begin} ns")
        return result
    return wrapper

class timer:
    def __init__(self, name="block"):
        self.name = name

    def __enter__(self):
        self.begin = time.time_ns()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time_ns()
        print(f"{self.name} took {end - self.begin} ns")
