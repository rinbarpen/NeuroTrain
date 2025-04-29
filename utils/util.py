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
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.amp.grad_scaler import GradScaler
from pathlib import Path
from PIL import Image 
from torchsummary import summary
from fvcore.nn import FlopCountAnalysis

from config import get_config, get_config_value
from utils.typed import FilePath, ImageInstance
from utils.dataset.dataset import get_chained_datasets

def prepare_logger():
    c = get_config()['private']['log']

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
    match c['train']['optimizer_type'].lower():
        case 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(), lr=c['train']['optimizer']['learning_rate'], weight_decay=c['train']['optimizer']['weight_decay'])
        case 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(), lr=c['train']['optimizer']['learning_rate'], weight_decay=c['train']['optimizer']['weight_decay'], eps=c['train']['optimizer']['eps'])
        case 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=c['train']['optimizer']['learning_rate'], weight_decay=c['train']['optimizer']['weight_decay'], eps=c['train']['optimizer']['eps'])

    return {
        'optimizer': optimizer,
        'lr_scheduler': LRScheduler(optimizer) if 'lr_scheduler' in c['train'].keys() else None,
        'scaler': GradScaler() if 'scaler' in c['train'].keys() else None,
    }

def get_train_valid_test_dataloader(use_valid=False):
    c = get_config()

    train_dataset = get_chained_datasets('train')
    test_dataset = get_chained_datasets('test')
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
        valid_dataset = get_chained_datasets('valid')
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=c["valid"]["batch_size"],
            pin_memory=True,
            num_workers=num_workers,
            shuffle=shuffle,
        )

        return train_loader, valid_loader, test_loader
    
    return train_loader, None, test_loader
    
def save_model(path: Path, model: nn.Module, *, 
               ext_path: Path|None=None,
               optimizer=None, lr_scheduler=None, scaler=None, **kwargs):
    model_cp = model.state_dict()

    try:
        torch.save(model_cp, path)
    except FileExistsError as e:
        path = path.parent / (path.stem +
                              time.strftime("%Y%m%d_%H%M%S", time.localtime()))
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
                                time.strftime("%Y%m%d_%H%M%S", time.localtime()))
            torch.save(ext_cp, ext_path)
def load_model(path: FilePath, map_location: str = 'cuda'):
    return torch.load(path, 
                      map_location=torch.device(map_location))
def load_model_ext(ext_path: FilePath, map_location: str = 'cuda'):
    return torch.load(ext_path, 
                      map_location=torch.device(map_location))

def summary_model_info(model_src: FilePath | torch.nn.Module, input_size: torch.Tensor, device: str="cpu"):
    if isinstance(model_src, FilePath):
        checkpoint = load_model(model_src, device)
        summary(checkpoint, input_size=input_size, device=device)
    elif isinstance(model_src, torch.nn.Module):
        summary(model_src, input_size=input_size, device=device)

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


def split_image(
    image: Image.Image, num_patches: int, *, output_dir: Path | None = None
) -> list[Image.Image]:
    """
    将图像分割成小块并保存到指定目录。

    Args:
        image (Image.Image): 图像文件实例。
        num_patches (int): 小块的数量。
        output_dir (Path): 保存小块图像的目录。
    """

    width, height = image.size
    n_rows, n_cols = (int(math.sqrt(num_patches)), int(math.sqrt(num_patches)))
    tile_width, tile_height = width // n_cols, height // n_rows

    tile_images = []
    for i in range(0, height, tile_height):
        for j in range(0, width, tile_width):
            # 定义当前小块的边界
            box = (j, i, j + tile_height, i + tile_width)

            # 避免超出图像边界
            if box[2] > width:
                box = (box[0], box[1], width, box[3])  # Adjust right boundary
            if box[3] > height:
                box = (box[0], box[1], box[2], height)  # Adjust bottom boundary

            # 提取小块
            try:
                tile_image = image.crop(box)
            except Exception as e:
                logging.error(f"切除图像失败。 边界：{box},错误：{e}")
                raise e

            tile_images.append(tile_image)

    if output_dir:
        for i, tile_image in enumerate(tile_images):
            output_filename = output_dir / f"tile_{i:04d}.png"
            tile_image.save(output_filename, "PNG")

    return tile_images
