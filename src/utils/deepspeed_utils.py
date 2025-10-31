"""
DeepSpeed 工具函数

提供 DeepSpeed 相关的工具函数，包括：
- DeepSpeed 环境检测和初始化
- 配置文件管理
- 分布式训练设置
- 内存和性能监控
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import torch
import torch.distributed as dist

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    deepspeed = None

logger = logging.getLogger(__name__)


def is_deepspeed_available() -> bool:
    """检查 DeepSpeed 是否可用"""
    return DEEPSPEED_AVAILABLE


def init_deepspeed_distributed():
    """初始化 DeepSpeed 分布式环境"""
    if not DEEPSPEED_AVAILABLE:
        raise ImportError("DeepSpeed is not available. Please install deepspeed: pip install deepspeed")

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size_env = os.environ.get('WORLD_SIZE')

    # 当未通过 deepspeed/torchrun 启动时，允许退化为单进程模式
    if world_size_env is None or int(world_size_env) <= 1:
        logger.info("DeepSpeed running in single-process mode (WORLD_SIZE not set).")
        return {
            'local_rank': local_rank,
            'world_size': 1,
            'rank': 0,
            'distributed': False
        }

    # 初始化分布式环境
    deepspeed.init_distributed()

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))

    logger.info(f"DeepSpeed distributed initialized - Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}")

    return {
        'local_rank': local_rank,
        'world_size': world_size,
        'rank': rank,
        'distributed': True
    }


def load_deepspeed_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    加载 DeepSpeed 配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        DeepSpeed 配置字典
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"DeepSpeed config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Loaded DeepSpeed config from {config_path}")
    return config


def create_deepspeed_config(
    zero_stage: int = 2,
    train_batch_size: int = 32,
    micro_batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    optimizer_type: str = "AdamW",
    scheduler_type: str = "WarmupLR",
    warmup_steps: int = 1000,
    gradient_clipping: float = 1.0,
    cpu_offload: bool = False,
    fp16: bool = False,
    bf16: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    创建 DeepSpeed 配置
    
    Args:
        zero_stage: ZeRO 优化阶段 (1, 2, 3)
        train_batch_size: 训练批次大小
        micro_batch_size: 每个GPU的微批次大小
        gradient_accumulation_steps: 梯度累积步数
        learning_rate: 学习率
        weight_decay: 权重衰减
        optimizer_type: 优化器类型
        scheduler_type: 学习率调度器类型
        warmup_steps: 预热步数
        gradient_clipping: 梯度裁剪阈值
        cpu_offload: 是否启用CPU卸载
        fp16: 是否启用FP16
        bf16: 是否启用BF16
        **kwargs: 其他配置参数
        
    Returns:
        DeepSpeed 配置字典
    """
    config = {
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "optimizer": {
            "type": optimizer_type,
            "params": {
                "lr": learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": weight_decay
            }
        },
        "scheduler": {
            "type": scheduler_type,
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": learning_rate,
                "warmup_num_steps": warmup_steps
            }
        },
        "zero_optimization": {
            "stage": zero_stage,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },
        "gradient_clipping": gradient_clipping,
        "steps_per_print": 2000,
        "wall_clock_breakdown": False,
        "memory_breakdown": False
    }
    
    # 根据 ZeRO 阶段添加特定配置
    if zero_stage >= 2:
        config["zero_optimization"]["cpu_offload"] = cpu_offload
    
    if zero_stage >= 3:
        config["zero_optimization"].update({
            "offload_optimizer": {
                "device": "cpu" if cpu_offload else "none",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu" if cpu_offload else "none",
                "pin_memory": True
            },
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        })
    
    # 添加混合精度配置
    if fp16:
        config["fp16"] = {
            "enabled": True,
            "auto_cast": False,
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        }
    elif bf16:
        config["bf16"] = {
            "enabled": True
        }
    
    # 添加其他配置
    config.update(kwargs)
    
    logger.info(f"Created DeepSpeed config with ZeRO stage {zero_stage}")
    return config


def save_deepspeed_config(config: Dict[str, Any], output_path: Union[str, Path]):
    """
    保存 DeepSpeed 配置到文件
    
    Args:
        config: DeepSpeed 配置字典
        output_path: 输出文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Saved DeepSpeed config to {output_path}")


def get_deepspeed_memory_usage() -> Dict[str, float]:
    """
    获取 DeepSpeed 内存使用情况
    
    Returns:
        内存使用情况字典 (MB)
    """
    if not DEEPSPEED_AVAILABLE:
        return {}
    
    memory_stats = {}
    
    if torch.cuda.is_available():
        # GPU 内存使用
        memory_stats['gpu_allocated'] = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        memory_stats['gpu_reserved'] = torch.cuda.memory_reserved() / 1024 / 1024   # MB
        memory_stats['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    
    # CPU 内存使用
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_stats['cpu_rss'] = memory_info.rss / 1024 / 1024  # MB
    memory_stats['cpu_vms'] = memory_info.vms / 1024 / 1024  # MB
    
    return memory_stats


def log_deepspeed_memory_usage(logger: logging.Logger, prefix: str = ""):
    """记录 DeepSpeed 内存使用情况"""
    memory_stats = get_deepspeed_memory_usage()
    
    if memory_stats:
        logger.info(f"{prefix}Memory Usage:")
        for key, value in memory_stats.items():
            logger.info(f"  {key}: {value:.2f} MB")


def get_deepspeed_rank_info() -> Dict[str, int]:
    """
    获取 DeepSpeed 进程排名信息
    
    Returns:
        排名信息字典
    """
    if not dist.is_initialized():
        return {'rank': 0, 'local_rank': 0, 'world_size': 1}
    
    return {
        'rank': dist.get_rank(),
        'local_rank': int(os.environ.get('LOCAL_RANK', 0)),
        'world_size': dist.get_world_size()
    }


def is_main_process() -> bool:
    """检查是否为主进程"""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def barrier():
    """同步所有进程"""
    if dist.is_initialized():
        dist.barrier()


def setup_deepspeed_logging(log_level: str = "INFO"):
    """
    设置 DeepSpeed 日志记录
    
    Args:
        log_level: 日志级别
    """
    # 设置 DeepSpeed 日志级别
    os.environ['DEEPSPEED_LOG_LEVEL'] = log_level.upper()
    
    # 设置 PyTorch 分布式日志级别
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL' if log_level.upper() == 'DEBUG' else 'INFO'


def create_deepspeed_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    **kwargs
):
    """
    创建适用于 DeepSpeed 的数据加载器
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 工作进程数
        pin_memory: 是否固定内存
        drop_last: 是否丢弃最后不完整的批次
        **kwargs: 其他 DataLoader 参数
        
    Returns:
        DataLoader 实例
    """
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    
    # 如果是分布式训练，使用 DistributedSampler
    if dist.is_initialized() and dist.get_world_size() > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle
        )
        shuffle = False  # DistributedSampler 已经处理了 shuffle
    else:
        sampler = None
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        **kwargs
    )


def cleanup_deepspeed():
    """清理 DeepSpeed 环境"""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("DeepSpeed process group destroyed")


def get_deepspeed_config_template(zero_stage: int = 2) -> str:
    """
    获取 DeepSpeed 配置模板
    
    Args:
        zero_stage: ZeRO 阶段
        
    Returns:
        配置模板字符串
    """
    templates = {
        1: "configs/deepspeed/deepspeed_zero1.json",
        2: "configs/deepspeed/deepspeed_zero2.json",
        3: "configs/deepspeed/deepspeed_zero3.json"
    }
    
    template_path = templates.get(zero_stage, templates[2])
    return template_path


def validate_deepspeed_config(config: Dict[str, Any]) -> List[str]:
    """
    验证 DeepSpeed 配置
    
    Args:
        config: DeepSpeed 配置字典
        
    Returns:
        验证错误列表
    """
    errors = []
    
    # 检查必需字段
    required_fields = ['train_batch_size', 'train_micro_batch_size_per_gpu', 'gradient_accumulation_steps']
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # 检查 ZeRO 配置
    if 'zero_optimization' in config:
        zero_config = config['zero_optimization']
        if 'stage' not in zero_config:
            errors.append("Missing 'stage' in zero_optimization")
        elif zero_config['stage'] not in [1, 2, 3]:
            errors.append("Invalid ZeRO stage. Must be 1, 2, or 3")
    
    # 检查优化器配置
    if 'optimizer' in config:
        opt_config = config['optimizer']
        if 'type' not in opt_config:
            errors.append("Missing 'type' in optimizer")
        if 'params' not in opt_config:
            errors.append("Missing 'params' in optimizer")
    
    return errors


def print_deepspeed_config(config: Dict[str, Any], logger: logging.Logger = None):
    """
    打印 DeepSpeed 配置
    
    Args:
        config: DeepSpeed 配置字典
        logger: 日志记录器
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("DeepSpeed Configuration:")
    logger.info(f"  Train Batch Size: {config.get('train_batch_size', 'N/A')}")
    logger.info(f"  Micro Batch Size: {config.get('train_micro_batch_size_per_gpu', 'N/A')}")
    logger.info(f"  Gradient Accumulation: {config.get('gradient_accumulation_steps', 'N/A')}")
    
    if 'zero_optimization' in config:
        zero_config = config['zero_optimization']
        logger.info(f"  ZeRO Stage: {zero_config.get('stage', 'N/A')}")
        logger.info(f"  CPU Offload: {zero_config.get('cpu_offload', False)}")
    
    if 'optimizer' in config:
        opt_config = config['optimizer']
        logger.info(f"  Optimizer: {opt_config.get('type', 'N/A')}")
        if 'params' in opt_config:
            params = opt_config['params']
            logger.info(f"  Learning Rate: {params.get('lr', 'N/A')}")
            logger.info(f"  Weight Decay: {params.get('weight_decay', 'N/A')}")
    
    if 'fp16' in config and config['fp16'].get('enabled', False):
        logger.info("  Mixed Precision: FP16")
    elif 'bf16' in config and config['bf16'].get('enabled', False):
        logger.info("  Mixed Precision: BF16")
    else:
        logger.info("  Mixed Precision: None")
