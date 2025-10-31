import subprocess
import json
from argparse import ArgumentParser
import logging
from pathlib import Path
import toml
import yaml
import os
from typing import List, Dict, Any, Optional
from itertools import product
import threading
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GPUAllocator:
    """GPU分配器，确保不同进程的GPU不重复"""
    
    def __init__(self, total_gpus: Optional[List[int]] = None):
        """
        初始化GPU分配器
        
        Args:
            total_gpus: 可用的GPU列表，如果为None则自动检测所有GPU
        """
        if total_gpus is None:
            import torch
            if torch.cuda.is_available():
                self.total_gpus = list(range(torch.cuda.device_count()))
            else:
                self.total_gpus = []
        else:
            self.total_gpus = total_gpus
        
        self.used_gpus = set()
        self.lock = threading.Lock()
        logger.info(f"初始化GPU分配器，可用GPU: {self.total_gpus}")
    
    def allocate(self, requested_devices: List[int]) -> Optional[List[int]]:
        """
        分配GPU设备
        
        Args:
            requested_devices: 请求的设备列表
            
        Returns:
            如果分配成功返回设备列表，否则返回None
        """
        with self.lock:
            # 检查请求的设备是否都被占用
            if any(gpu in self.used_gpus for gpu in requested_devices):
                return None
            
            # 检查请求的设备是否在可用设备列表中
            if any(gpu not in self.total_gpus for gpu in requested_devices):
                logger.warning(f"请求的设备 {requested_devices} 不在可用设备列表 {self.total_gpus} 中")
                return None
            
            # 分配设备
            for gpu in requested_devices:
                self.used_gpus.add(gpu)
            
            logger.info(f"分配GPU设备: {requested_devices}, 已使用设备: {sorted(self.used_gpus)}")
            return requested_devices
    
    def release(self, devices: List[int]):
        """
        释放GPU设备
        
        Args:
            devices: 要释放的设备列表
        """
        with self.lock:
            for gpu in devices:
                if gpu in self.used_gpus:
                    self.used_gpus.remove(gpu)
            logger.info(f"释放GPU设备: {devices}, 已使用设备: {sorted(self.used_gpus)}")
    
    def format_devices(self, devices: List[int]) -> str:
        """
        格式化设备列表为CUDA_VISIBLE_DEVICES格式
        
        Args:
            devices: 设备列表
            
        Returns:
            格式化的设备字符串
        """
        return ','.join(map(str, devices))


def parse_devices(devices: Any) -> List[int]:
    """
    解析设备配置
    
    Args:
        devices: 设备配置，可以是整数、字符串、列表等
        
    Returns:
        设备ID列表
    """
    if devices is None:
        return []
    
    if isinstance(devices, int):
        return [devices]
    elif isinstance(devices, str):
        # 支持 "0", "0,1", "0:1" 等格式
        if ',' in devices:
            return [int(d.strip()) for d in devices.split(',')]
        elif ':' in devices:
            # 支持范围格式 "0:2" 表示 [0, 1]
            start, end = map(int, devices.split(':'))
            return list(range(start, end))
        else:
            return [int(devices)]
    elif isinstance(devices, list):
        return [int(d) for d in devices]
    else:
        raise ValueError(f"不支持设备格式: {devices}")


def expand_variants(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    展开参数变体
    
    Args:
        config: 任务配置，可能包含variants字段
        
    Returns:
        展开后的配置列表
    """
    if 'variants' not in config:
        # 如果没有variants，返回原始配置的单个元素列表
        return [config]
    
    variants = config['variants']
    base_config = {k: v for k, v in config.items() if k != 'variants'}
    
    # 生成所有变体的组合
    expanded_configs = []
    variant_keys = list(variants.keys())
    variant_values = list(variants.values())
    
    # 使用product生成所有组合
    for combination in product(*variant_values):
        variant_config = base_config.copy()
        variant_ext_args = list(variant_config.get('ext_args', []))
        
        # 为每个变体参数添加命令行参数
        for key, value in zip(variant_keys, combination):
            if key == 'seed':
                variant_ext_args.extend(['--seed', str(value)])
            elif key == 'device':
                # device在variants中会被devices字段覆盖
                pass
            else:
                # 其他自定义参数
                variant_ext_args.extend(['--' + key.replace('_', '-'), str(value)])
        
        variant_config['ext_args'] = variant_ext_args
        variant_config['_variant'] = dict(zip(variant_keys, combination))
        expanded_configs.append(variant_config)
    
    return expanded_configs


def run_task(
    name: str,
    config: Dict[str, Any],
    gpu_allocator: GPUAllocator,
    parallel: bool = False
):
    """
    运行单个任务
    
    Args:
        name: 任务名称
        config: 任务配置
        gpu_allocator: GPU分配器
        parallel: 是否并行执行
        
    Returns:
        如果parallel=True，返回(process, allocated_devices)，否则返回bool表示是否成功
    """
    config_file = config['config']
    ext_args = list(config.get('ext_args', []))
    
    # 解析并分配GPU设备
    devices = None
    devices_str = None
    if 'devices' in config:
        requested_devices = parse_devices(config['devices'])
        if requested_devices:
            devices = gpu_allocator.allocate(requested_devices)
            if devices is None:
                logger.error(f"任务 {name}: 无法分配GPU设备 {requested_devices}")
                return None if parallel else False
            devices_str = gpu_allocator.format_devices(devices)
            # 添加设备参数到ext_args
            if len(devices) == 1:
                ext_args.extend(['--device', f"cuda:0"])  # 单卡时，由于设置了CUDA_VISIBLE_DEVICES，应该用cuda:0
            else:
                # 多卡训练通过CUDA_VISIBLE_DEVICES环境变量控制
                # DDP配置应该在配置文件中启用
                ext_args.extend(['--device', 'cuda'])
    
    # 获取变体信息用于日志
    variant_info = config.get('_variant', {})
    variant_str = f", 变体: {variant_info}" if variant_info else ""
    device_str = f", GPU: {devices_str}" if devices_str else ""
    
    logger.info(f"开始任务: {name}{variant_str}{device_str}")
    logger.info(f"  配置文件: {config_file}")
    logger.info(f"  额外参数: {ext_args}")
    
    try:
        # 准备环境变量
        env = os.environ.copy()
        if devices_str:
            env['CUDA_VISIBLE_DEVICES'] = devices_str
        
        # 执行任务
        cmd = ["python", "main.py", "-c", config_file] + ext_args
        logger.debug(f"执行命令: {' '.join(cmd)}")
        
        if parallel:
            # 并行模式：启动进程但不等待，返回进程和分配的设备
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return (process, devices)
        else:
            # 串行模式：等待完成
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate()
            success = process.returncode == 0
            
            if not success:
                logger.error(f"任务 {name} 执行失败:")
                if stdout:
                    logger.error(f"STDOUT:\n{stdout}")
                if stderr:
                    logger.error(f"STDERR:\n{stderr}")
            else:
                logger.info(f"任务 {name} 执行成功")
            
            # 释放GPU设备
            if devices:
                gpu_allocator.release(devices)
            
            return success
    except Exception as e:
        logger.error(f"任务 {name} 执行异常: {e}")
        # 释放GPU设备
        if devices:
            gpu_allocator.release(devices)
        return None if parallel else False


def main():
    parser = ArgumentParser("Train Pipeline")
    parser.add_argument('-c', '--config', required=True, help="pipeline config file")
    parser.add_argument('--parallel', action='store_true', help="并行执行不冲突的任务")
    parser.add_argument('--gpus', type=str, help="可用GPU列表，例如: 0,1,2,3 或 0:4")

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        from src.constants import ProjectFilenameEnv
        config_path = ProjectFilenameEnv().pipeline_config_dir / args.config

    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        return

    # 加载配置文件
    match config_path.suffix:
        case '.toml':
            with config_path.open('r', encoding='utf-8') as f:
                pipeline_config = toml.load(f)
        case '.yaml'|'.yml':
            with config_path.open('r', encoding='utf-8') as f:
                pipeline_config = yaml.safe_load(f)
        case '.json':
            with config_path.open('r', encoding='utf-8') as f:
                pipeline_config = json.load(f)
        case _:
            raise ValueError(f"不支持的文件格式: {config_path.suffix}")

    # 初始化GPU分配器
    if args.gpus:
        total_gpus = parse_devices(args.gpus)
    else:
        total_gpus = None
    gpu_allocator = GPUAllocator(total_gpus)

    # 展开所有任务的变体
    all_tasks = []
    for name, config in pipeline_config.items():
        expanded_configs = expand_variants(config)
        for idx, exp_config in enumerate(expanded_configs):
            task_name = f"{name}"
            if len(expanded_configs) > 1:
                task_name += f"_variant{idx}"
            all_tasks.append((task_name, exp_config, config.get('parallel', args.parallel)))

    logger.info(f"共 {len(all_tasks)} 个任务需要执行")

    # 执行任务
    if args.parallel:
        # 并行执行模式：尝试并行启动所有不冲突的任务
        processes = []
        for name, config, task_parallel in all_tasks:
            result = run_task(name, config, gpu_allocator, parallel=True)
            if result is not None:
                process, allocated_devices = result
                processes.append((name, process, allocated_devices))
            else:
                logger.warning(f"任务 {name} 启动失败或GPU分配失败，将在串行模式重试")
        
        # 等待所有进程完成
        for name, process, allocated_devices in processes:
            logger.info(f"等待任务 {name} 完成...")
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                logger.error(f"任务 {name} 执行失败:")
                if stdout:
                    logger.error(f"STDOUT:\n{stdout}")
                if stderr:
                    logger.error(f"STDERR:\n{stderr}")
            else:
                logger.info(f"任务 {name} 执行成功")
            
            # 释放GPU
            if allocated_devices:
                gpu_allocator.release(allocated_devices)
    else:
        # 串行执行模式
        for name, config, _ in all_tasks:
            success = run_task(name, config, gpu_allocator, parallel=False)
            if not success:
                logger.error(f"任务 {name} 失败，停止执行后续任务")
                break
    
    logger.info("所有任务执行完成")


if __name__ == '__main__':
    main()
