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
from PIL.Image import Image 
from torch import nn
from torchsummary import summary
from config import get_config
from typing import Literal
from fvcore.nn import FlopCountAnalysis

def prepare_logger(name: str|None = None):
    log_colors = {
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'FATAL': 'bold_red',
    }
    formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s %(levelname)s | %(name)s | %(message)s',
        log_colors=log_colors
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    os.makedirs('logs', exist_ok=True)
    filename = os.path.join('logs', strftime('%Y_%m_%d_%H_%M_%S.log', time.localtime()))
    file_handler = logging.FileHandler(filename, encoding='utf-8', delay=True)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s | %(name)s | %(message)s'
    ))
    def set_logger(name: str, level: int):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    c = get_config()
    log_level = logging.DEBUG if c['private']['verbose'] else logging.INFO
    if not name:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG if c['private']['verbose'] else logging.INFO)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

        set_logger('train', log_level)
        set_logger('test', log_level)
        set_logger('predict', log_level)
        set_logger('painter', log_level)
        set_logger('recorder', log_level)
    else:
        set_logger(name, log_level)

def get_logger(name: str):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        prepare_logger(name)
    return logger

def save_model(path: Path, model: nn.Module, *, 
               ext_path: Path|None=None,
               optimizer=None, lr_scheduler=None, scaler=None, **kwargs):
    model_cp = model.state_dict()

    try:
        torch.save(model_cp, path)
    except FileExistsError as e:
        path = path.parent / (path.stem +
                              strftime("%Y%m%d_%H%M%S", time.localtime()))
        torch.save(model_cp, path)

    if ext_path:
        ext_cp = dict()
        if optimizer:
            ext_cp["optimizer"] = optimizer.state_dict()
        if lr_scheduler:
            ext_cp["lr_scheduler"] = lr_scheduler.state_dict()
        if scaler:
            ext_cp["scaler"] = scaler.state_dict()
        for k, v in kwargs.items():
            ext_cp[k] = v
        try:
            torch.save(ext_cp, ext_path)
        except FileExistsError as e:
            ext_path = ext_path.parent / (ext_path.stem +
                                strftime("%Y%m%d_%H%M%S", time.localtime()))
            torch.save(ext_cp, ext_path)


def load_model(path: Path, map_location: str = 'cuda'):
    return torch.load(path, 
                      map_location=torch.device(map_location))
def load_model_ext(ext_path: Path, map_location: str = 'cuda'):
    return torch.load(ext_path, 
                      map_location=torch.device(map_location))

def save_model_to_onnx(path: Path, model: nn.Module, input_size: tuple):
    dummy_input = torch.randn(input_size)
    torch.onnx.export(model, dummy_input, path)

def summary_model_info(model_src: Path | torch.nn.Module, input_size: torch.Tensor, device: str="cpu"):
    if isinstance(model_src, Path):
        checkpoint = load_model(model_src, device)
        summary(checkpoint, input_size=input_size, device=device)
    elif isinstance(model_src, torch.nn.Module):
        summary(model_src, input_size=input_size, device=device)


def save_numpy_data(path: Path, data: np.ndarray | torch.Tensor):
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()

    try:
        np.save(path, data)
    except FileNotFoundError as e:
        path.parent.mkdir(parents=True)
        np.save(path, data)

def load_numpy_data(path: Path):
    try:
        data = np.load(path)
        return data
    except FileNotFoundError as e:
        logging.error(f'File is not found: {e}')
        raise e


def tuple2list(t: tuple):
    return list(t)

def list2tuple(l: list):
    return tuple(l)

def image_to_numpy(img: Image|cv2.Mat) -> np.ndarray:
    if isinstance(img, Image):
        img_np = np.array(img) # (H, W, C)
        if img_np.ndim == 3:
            img_np = img_np.transpose(2, 0, 1) # (C, H, W)
    elif isinstance(img, cv2.Mat):
        img_np = np.array(img)
    
    # output shape: (C, H, W) for RGB or (H, W) for gray
    return img_np

def model_gflops(model: nn.Module, input_size: tuple, device: str = 'cuda') -> float:
    dummy_input = torch.randn(input_size).to(device)
    flops = FlopCountAnalysis(model, dummy_input)
    total_flops = flops.total()
    return total_flops / 1e9  # Convert to GFLOPs


class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def elapsed_time(self):
        if self.end_time is None:
            return time.time() - self.start_time
        else:
            return self.end_time - self.start_time