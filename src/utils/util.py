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
from torchinfo import summary
from typing import Sequence, Type
from pathlib import Path
from PIL import Image
from fvcore.nn import flop_count, flop_count_table, FlopCountAnalysis, parameter_count, parameter_count_table


from src.config import get_config, get_config_value
from src.utils.typed import FilePath, ImageInstance
from src.utils.criterion import CombineCriterion, get_criterion

def prepare_logger():
    c = get_config_value('private.log', default={
        'debug': False,
        'verbose': False,
        'log_file_format': '%Y-%m-%d %H_%M_%S',
        'log_format': '%(asctime)s %(levelname)s | %(name)s | %(message)s',
    })
    assert c is not None

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
    train_c = c['train']
    optimizer_c = train_c['optimizer']
    match optimizer_c['type'].lower():
        case 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(), lr=optimizer_c['learning_rate'], weight_decay=optimizer_c['weight_decay'])
        case 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=optimizer_c['learning_rate'], weight_decay=optimizer_c['weight_decay'])
        case 'adam' | _:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=optimizer_c['learning_rate'], weight_decay=optimizer_c['weight_decay'])

    if 'lr_scheduler' not in train_c:
        scheduler = None
    else:
        scheduler_c = c['lr_scheduler']
        match scheduler_c['type'].lower():
            case 'step':
                step_size = scheduler_c['step_size']
                gamma = scheduler_c.get('gamma', 0.1)
                scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
            case 'mstep':
                milestones = scheduler_c['step_size'] # list
                if isinstance(milestones, int):
                    milestones = [milestones]
                gamma = scheduler_c.get('gamma', 0.1)
                scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
            case 'cos':
                T_max = scheduler_c.get('T_max', train_c['epoch'])  # 周期
                eta_min = scheduler_c.get('eta_min', 0)  # 最小学习率
                scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
            case 'cos_warm':
                T_0 = scheduler_c.get('T_0', 10)  # 首次周期
                T_mult = scheduler_c.get('T_mult', 2)  # 周期倍数
                eta_min = scheduler_c.get('eta_min', 0.001)  # 最小学习率
                scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
            case _:
                scheduler = LRScheduler(optimizer)

    return {
        'optimizer': optimizer,
        'lr_scheduler': scheduler,
        'scaler': GradScaler() if 'scaler' in train_c else None,
    }

def get_train_criterion():
    c = get_config()
    return CombineCriterion(*[get_criterion(cc) for cc in c['criterion']])

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

def image_to_numpy(img: ImageInstance) -> np.ndarray:
    if isinstance(img, Image.Image):
        img_np = np.array(img) # (H, W, C)
        if img_np.ndim == 3:
            img_np = img_np.transpose(2, 0, 1)  # (C, H, W)
    elif isinstance(img, cv2.Mat):
        img_np = np.array(img)
    
    # output shape: (C, H, W) for RGB or (H, W) for gray
    return img_np

def model_info(output_dir: Path, model: nn.Module, input_sizes: Sequence[int]|Sequence[Sequence[int]], dtypes: Type|Sequence[Type]|None=None, device: str = 'cuda', *, rich_print=True):
    if isinstance(input_sizes, Sequence[int]):
        input_sizes = [input_sizes]
    if dtypes is None:
        pass
    elif isinstance(dtypes, Type):
        dtypes = [dtypes] * len(input_sizes)
    else:
        assert len(dtypes) == len(input_sizes), 'dtypes and input_sizes must have the same length'

    model_stats = summary(model, input_size=input_sizes, dtypes=dtypes, device=device, verbose=0)

    summary_file = output_dir / 'model_summary.txt'
    with summary_file.open('w', encoding='utf-8') as f:
        f.write(str(model_stats))
    
    if rich_print:
        from rich import print
        print("Model Summary:")
        print(f"Total params: {model_stats.total_params}")
        print(f"Trainable params: {model_stats.trainable_params}")
        print(f"Model size: {model_stats.total_mult_adds}")


def model_flops(output_dir: Path, model: nn.Module, input_sizes: Sequence[int]|Sequence[Sequence[int], ...], device: str = 'cuda', *, rich_print=True) -> float:
    if isinstance(input_sizes, Sequence[int]):
        input_sizes = [input_sizes]
    input_tensors = [torch.randn(input_size) for input_size in input_sizes]
    input_tensors = tuple(input_tensors)
    
    analysis = FlopCountAnalysis(model, input_tensors)
    table = flop_count_table(analysis)

    flop_count_file = output_dir / 'model_flop_count.txt'
    with flop_count_file.open('w', encoding='utf-8') as f:
        f.write(table)

    if rich_print:
        from rich import print
        print(table)

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

def str2dtype(dtype: str) -> torch.dtype:
    match dtype:
        case 'float16' | 'f16':
            ttype = torch.float16
        case 'float32' | 'f32':
            ttype = torch.float32
        case 'float64' | 'f64':
            ttype = torch.float64
        case 'bfloat16' | 'bf16':
            ttype = torch.bfloat16
        case 'uint8' | 'u8':
            ttype = torch.uint8
        case 'uint16' | 'u16':
            ttype = torch.uint16
        case 'uint32' | 'u32':
            ttype = torch.uint32
        case 'uint64' | 'u64':
            ttype = torch.uint64
        case 'int8' | 'i8':
            ttype = torch.int8
        case 'int16' | 'i16':
            ttype = torch.int16
        case 'int32' | 'i32':
            ttype = torch.int32
        case 'int64' | 'i64':
            ttype = torch.int64
        case _:
            ttype = torch.get_default_dtype()
    return ttype
