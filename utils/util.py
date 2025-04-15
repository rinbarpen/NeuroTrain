import logging
import colorlog
import os.path
import os
from pydantic import Field, BaseModel
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
from fvcore.nn import FlopCountAnalysis

def prepare_logger(name: str|None = None):
    log_colors = {
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'FATAL': 'bold_red',
    }
    c = get_config()
    formatter = colorlog.ColoredFormatter(
        '%(log_color)s' + c['private']['log_format'],
        log_colors=log_colors
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    os.makedirs('logs', exist_ok=True)
    filename = os.path.join('logs', strftime(c['private']['log_file_format']+'.log', time.localtime()))
    file_handler = logging.FileHandler(filename, encoding='utf-8', delay=True)
    file_handler.setFormatter(logging.Formatter(
        c['private']['log_format']
    ))
    def set_logger(name: str, level: int):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    def get_log_level():
        if c['private']['debug']:
            return logging.DEBUG
        elif c['private']['verbose']:
            return logging.INFO
        else:
            return logging.WARNING

    log_level = get_log_level()
    if not name:
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

        set_logger('train', log_level)
        set_logger('test', log_level)
        set_logger('predict', log_level)
    else:
        set_logger(name, log_level)

def get_logger(name: str|None = None):
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


def load_model(path: str|Path, map_location: str = 'cuda'):
    return torch.load(path, 
                      map_location=torch.device(map_location))
def load_model_ext(ext_path: str|Path, map_location: str = 'cuda'):
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


class TimerUnit(BaseModel):
    start_time: float = Field(float('NAN'))
    end_time: float = Field(float('NAN'))

class Timer:

    def __init__(self):
        self.time_map: dict[str, TimerUnit] = {}

    def start(self, task: str=""):
        unit = TimerUnit()
        unit.start_time = time.time()
        self.time_map[task] = unit

    def stop(self, task: str=""):
        try:
            self.time_map[task].end_time = time.time()
        except Exception:
            pass

    def elapsed_time(self, task: str=""):
        try:
            return self.time_map[task].end_time - self.time_map[task].start_time
        except Exception:
            return float('NAN')
    def all_elapsed_time(self):
        costs = {}
        for task in self.time_map.keys():
            costs[task] = self.elapsed_time(task)
        return costs
    
    def total_elapsed_time(self) -> float:
        total_cost = np.array(self.all_elapsed_time().values()).sum()
        return total_cost
