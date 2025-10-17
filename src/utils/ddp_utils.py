"""
DDP (Distributed Data Parallel) 工具函数
用于支持PyTorch的分布式训练
"""

import os
import logging
import torch
import torch.distributed as dist
from typing import Dict, Any, Optional


def init_ddp_distributed() -> Dict[str, int]:
    """
    初始化DDP分布式环境
    
    Returns:
        Dict[str, int]: 包含rank信息的字典
    """
    # 从环境变量获取分布式训练参数
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # 设置CUDA设备
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f'cuda:{local_rank}'
    else:
        device = 'cpu'
    
    # 初始化分布式进程组
    if world_size > 1:
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
    
    return {
        'rank': rank,
        'local_rank': local_rank,
        'world_size': world_size,
        'device': device
    }


def is_main_process() -> bool:
    """
    检查当前进程是否为主进程
    
    Returns:
        bool: 是否为主进程
    """
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def setup_ddp_logging(log_level: str = "INFO") -> None:
    """
    设置DDP日志
    
    Args:
        log_level: 日志级别
    """
    if is_main_process():
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        # 非主进程只记录错误
        logging.basicConfig(
            level=logging.ERROR,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )


def cleanup_ddp() -> None:
    """
    清理DDP环境
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def get_ddp_rank_info() -> Dict[str, int]:
    """
    获取DDP rank信息
    
    Returns:
        Dict[str, int]: rank信息
    """
    if not dist.is_initialized():
        return {
            'rank': 0,
            'local_rank': 0,
            'world_size': 1
        }
    
    return {
        'rank': dist.get_rank(),
        'local_rank': int(os.environ.get('LOCAL_RANK', 0)),
        'world_size': dist.get_world_size()
    }


def create_ddp_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True
):
    """
    创建DDP数据加载器
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 工作进程数
        pin_memory: 是否固定内存
        drop_last: 是否丢弃最后一批不完整的数据
        
    Returns:
        DataLoader: 数据加载器
    """
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    
    # 创建分布式采样器
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size() if dist.is_initialized() else 1,
        rank=dist.get_rank() if dist.is_initialized() else 0,
        shuffle=shuffle
    )
    
    # 创建数据加载器
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )


def sync_model_parameters(model: torch.nn.Module) -> None:
    """
    同步模型参数（用于确保所有进程的模型参数一致）
    
    Args:
        model: 模型
    """
    if not dist.is_initialized():
        return
    
    for param in model.parameters():
        if param.requires_grad:
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= dist.get_world_size()


def sync_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    同步张量到所有进程
    
    Args:
        tensor: 要同步的张量
        
    Returns:
        torch.Tensor: 同步后的张量
    """
    if not dist.is_initialized():
        return tensor
    
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor


def barrier() -> None:
    """
    同步所有进程
    """
    if dist.is_initialized():
        dist.barrier()


def is_ddp_available() -> bool:
    """
    检查DDP是否可用
    
    Returns:
        bool: DDP是否可用
    """
    return torch.cuda.is_available() and torch.cuda.device_count() > 1
