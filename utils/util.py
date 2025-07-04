import logging
import os.path
import os
import time
import math
import random
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler, StepLR, MultiStepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.amp.grad_scaler import GradScaler

from pathlib import Path
from PIL import Image
from torchsummary import summary
from fvcore.nn import FlopCountAnalysis

from config import get_config, get_config_value
from utils.typed import FilePath, ImageInstance
from utils.dataset import get_dataset

def prepare_logger():
    c = get_config_value('private.log', default={
        'debug': False,
        'verbose': False,
        'log_file_format': '%Y-%m-%d %H_%M_%S',
        'log_format': '%(asctime)s %(levelname)s | %(name)s | %(message)s',
    })

    os.makedirs('logs', exist_ok=True)
    filename = os.path.join('logs', time.strftime(c['log_file_format'], time.localtime()) + '.log')
    file_handler = logging.FileHandler(filename, encoding='utf-8', delay=True)
    file_handler.setFormatter(logging.Formatter(c['log_format']))

    def get_log_level():
        if c['debug']:
            return logging.DEBUG
        elif c['verbose']:
            return logging.INFO
        else:
            return logging.WARNING

    logging.basicConfig(filename=filename, 
                        level=get_log_level(), 
                        format=c['log_format'], 
                        handlers=[file_handler])

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_train_tools(model: nn.Module):
    c = get_config()
    optimizer_c = c['train']['optimizer']
    match optimizer_c['type'].lower():
        case 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(), lr=optimizer_c['learning_rate'], weight_decay=optimizer_c['weight_decay'])
        case 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=optimizer_c['learning_rate'], weight_decay=optimizer_c['weight_decay'])
        case 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(), lr=optimizer_c['learning_rate'], weight_decay=optimizer_c['weight_decay'])
        case _:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=optimizer_c['learning_rate'], weight_decay=optimizer_c['weight_decay'])

    if not 'lr_scheduler' in c['train']:
        scheduler = None
    else:
        scheduler_c = c['train']['lr_scheduler']
        match scheduler_c['type'].lower():
            case 'step':
                step_size = scheduler_c['step_size']
                gamma = scheduler_c.get('gamma', 0.1)
                scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
            case 'mstep':
                milestones = scheduler_c['step_size']
                gamma = scheduler_c.get('gamma', 0.1)
                scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
            case _:
                scheduler = LRScheduler(optimizer)

    return {
        'optimizer': optimizer,
        'lr_scheduler': scheduler,
        'scaler': GradScaler() if 'scaler' in c['train'] else None,
    }

def get_train_valid_test_dataloader(use_valid=False):
    c = get_config()

    train_dataset = get_dataset('train')
    test_dataset = get_dataset('test')
    num_workers = c["dataloader"]["num_workers"]
    shuffle = c["dataloader"]["shuffle"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=c["train"]["batch_size"],
        pin_memory=True,
        num_workers=num_workers,
        shuffle=shuffle,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=c["test"]["batch_size"],
        pin_memory=True,
        num_workers=num_workers,
        shuffle=shuffle,
    )
    if use_valid:
        valid_dataset = get_dataset('valid')
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=c["valid"]["batch_size"],
            pin_memory=True,
            num_workers=num_workers,
            shuffle=shuffle,
        )

        return train_loader, valid_loader, test_loader

    return train_loader, None, test_loader

def save_model(path: FilePath, model: nn.Module, *, 
               ext_path: FilePath|None=None,
               optimizer=None, lr_scheduler=None, scaler=None, **kwargs):
    model_cp = model.state_dict()
    path = Path(path)

    try:
        torch.save(model_cp, path)
    except FileExistsError as e:
        path = path.parent / (path.stem +
                              time.strftime("%Y%m%d_%H%M%S", time.localtime()))
        torch.save(model_cp, path)

    if ext_path:
        ext_path = Path(ext_path)
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
                                time.strftime("%Y%m%d_%H%M%S", time.localtime()))
            torch.save(ext_cp, ext_path)
def load_model(path: FilePath, map_location: str = 'cuda'):
    return torch.load(path, map_location)
def load_model_ext(ext_path: FilePath, map_location: str = 'cuda'):
    return torch.load(ext_path, map_location)

def summary_model_info(model: torch.nn.Module, input_size: tuple[int, ...], device: str="cuda"):
    summary(model, input_size=input_size, device=device)

# def disable_torch_init():
#     """
#     Disable the redundant torch default initialization to accelerate model creation.
#     """
#     setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
#     setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def save_numpy_data(path: FilePath, data: np.ndarray | torch.Tensor):
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    if isinstance(path, str):
        path = Path(path)

    try:
        np.save(path, data)
    except FileNotFoundError as e:
        path.parent.mkdir(parents=True)
        np.save(path, data)
def load_numpy_data(path: FilePath):
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

def image_to_numpy(img: ImageInstance) -> np.ndarray:
    if isinstance(img, Image.Image):
        img_np = np.array(img) # (H, W, C)
        if img_np.ndim == 3:
            img_np = img_np.transpose(2, 0, 1)  # (C, H, W)
    elif isinstance(img, cv2.Mat):
        img_np = np.array(img)
    
    # output shape: (C, H, W) for RGB or (H, W) for gray
    return img_np

def model_gflops(model: nn.Module, input_size: tuple, device: str = 'cuda') -> float:
    dummy_input = torch.randn(input_size).to(device)
    flops = FlopCountAnalysis(model, dummy_input)
    total_flops = flops.total()
    return total_flops / 1e9  # Convert to GFLOPs


# freeze_filter = lambda n: ("clip" in n) or ("bert" in n)
# optimizer_filters = [lambda n: "encoder" not in n, lambda n: "encoder" in n and "clip" not in n]
# c = {"lr": [None, 0.01]}

# freeze_layers(model, freeze_filter, optimizer_filters, **c)
# layer_filter: 
#  layer_name: str [input]
#  result: bool [output]
def freeze_layers(model: nn.Module, freeze_filter, optimizer_filters, **kwargs):
    named_params = model.named_parameters()
    for n, p in named_params:
        if freeze_filter(n) and p.requires_grad:
            p.requires_grad = False

#     param_dicts = [{'params': [p for n, p in named_params if optimizer_filter(n) and p.requires_grad], 'lr': kwargs['lr'][i]} for i, optimizer_filter in enumerate(optimizer_filters)]
#     return param_dicts

