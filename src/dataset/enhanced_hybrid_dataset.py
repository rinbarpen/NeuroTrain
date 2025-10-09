from .custom_dataset import CustomDataset
import numpy as np
import torch
import logging
from typing import Sequence, Dict, List, Optional, Union, Tuple
from torch.utils.data import Dataset, Subset
import random

class EnhancedHybridDataset(Dataset):
    """
    增强版混合数据集，支持以下功能：
    1. 数据集占比控制 - 可以设置每个数据集的采样比例
    2. 加载顺序控制 - 可以指定数据集的优先级和加载顺序
    3. 动态重采样 - 支持在训练过程中动态调整数据集比例
    4. 数据集权重 - 支持为不同数据集设置不同的权重
    """
    
    def __init__(
        self, 
        datasets: Sequence[CustomDataset],
        ratios: Optional[Sequence[float]] = None,
        priorities: Optional[Sequence[int]] = None,
        weights: Optional[Sequence[float]] = None,
        shuffle_order: bool = False,
        random_seed: Optional[int] = None,
        enable_dynamic_resampling: bool = False
    ):
        """
        初始化增强版混合数据集
        
        Args:
            datasets: 数据集列表
            ratios: 每个数据集的采样比例，如果为None则使用原始长度比例
            priorities: 数据集优先级，数字越小优先级越高，如果为None则按输入顺序
            weights: 数据集权重，用于加权采样，如果为None则均等权重
            shuffle_order: 是否在每个epoch开始时随机打乱数据集顺序
            random_seed: 随机种子
            enable_dynamic_resampling: 是否启用动态重采样
        """
        if not datasets:
            raise ValueError("数据集列表不能为空")
        
        self.original_datasets = list(datasets)
        self.num_datasets = len(datasets)
        self.shuffle_order = shuffle_order
        self.random_seed = random_seed
        self.enable_dynamic_resampling = enable_dynamic_resampling
        
        # 设置随机种子
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # 处理优先级排序
        if priorities is not None:
            if len(priorities) != self.num_datasets:
                raise ValueError("优先级列表长度必须与数据集数量相同")
            # 按优先级排序数据集
            sorted_indices = np.argsort(priorities)
            self.datasets = [datasets[i] for i in sorted_indices]
            self.original_priorities = priorities
            self.sorted_priorities = [priorities[i] for i in sorted_indices]
            if ratios is not None:
                ratios = [ratios[i] for i in sorted_indices]
            if weights is not None:
                weights = [weights[i] for i in sorted_indices]
        else:
            self.datasets = list(datasets)
            self.original_priorities = None
            self.sorted_priorities = None
        
        # 处理数据集比例
        self.original_lengths = [len(dataset) for dataset in self.datasets]
        self.total_original_length = sum(self.original_lengths)
        
        if ratios is not None:
            if len(ratios) != self.num_datasets:
                raise ValueError("比例列表长度必须与数据集数量相同")
            if not np.isclose(sum(ratios), 1.0, rtol=1e-5):
                logging.warning(f"数据集比例之和不等于1.0: {sum(ratios)}，将自动归一化")
                ratios = np.array(ratios) / sum(ratios)
            self.ratios = list(ratios)
        else:
            # 使用原始长度比例
            self.ratios = [length / self.total_original_length for length in self.original_lengths]
        
        # 处理权重
        if weights is not None:
            if len(weights) != self.num_datasets:
                raise ValueError("权重列表长度必须与数据集数量相同")
            self.weights = list(weights)
        else:
            self.weights = [1.0] * self.num_datasets
        
        # 计算实际采样长度
        self._calculate_sampling_lengths()
        
        # 构建索引映射
        self._build_index_mapping()
        
        logging.info(f"EnhancedHybridDataset初始化完成:")
        for i, dataset in enumerate(self.datasets):
            dataset_name = getattr(dataset, 'name', f'Dataset_{i}')
            logging.info(f"  数据集{i}: {dataset_name} - 原始长度: {self.original_lengths[i]}, "
                        f"采样长度: {self.sampling_lengths[i]}, 比例: {self.ratios[i]:.3f}, "
                        f"权重: {self.weights[i]:.3f}")
    
    def _calculate_sampling_lengths(self):
        """计算每个数据集的采样长度"""
        # 基于比例计算目标长度
        target_total_length = max(self.original_lengths)  # 使用最大数据集长度作为基准
        self.sampling_lengths = []
        
        for i, ratio in enumerate(self.ratios):
            target_length = int(target_total_length * ratio)
            # 确保不超过原始数据集长度
            actual_length = min(target_length, self.original_lengths[i])
            self.sampling_lengths.append(actual_length)
        
        self.total_length = sum(self.sampling_lengths)
    
    def _build_index_mapping(self):
        """构建索引映射"""
        self.cumulative_lengths = np.cumsum(self.sampling_lengths)
        
        # 为每个数据集创建采样索引
        self.dataset_indices = []
        for i, (dataset, sampling_length, original_length) in enumerate(
            zip(self.datasets, self.sampling_lengths, self.original_lengths)
        ):
            if sampling_length <= original_length:
                # 随机采样
                indices = np.random.choice(original_length, sampling_length, replace=False)
            else:
                # 需要重复采样
                indices = np.random.choice(original_length, sampling_length, replace=True)
            
            self.dataset_indices.append(sorted(indices))
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, index: int):
        if index >= self.total_length:
            raise IndexError(f"索引 {index} 超出范围 [0, {self.total_length})")
        
        # 找到对应的数据集
        dataset_index = np.searchsorted(self.cumulative_lengths, index, side='right')
        
        # 计算在该数据集中的本地索引
        if dataset_index == 0:
            local_index = index
        else:
            local_index = index - self.cumulative_lengths[dataset_index - 1]
        
        # 获取实际的数据索引
        actual_index = self.dataset_indices[dataset_index][local_index]
        
        # 获取数据并添加数据集信息
        data = self.datasets[dataset_index][actual_index]
        
        # 如果数据是字典，添加数据集元信息
        if isinstance(data, dict):
            data = data.copy()
            # 获取数据集名称，确保是字符串而不是函数
            dataset_name = getattr(self.datasets[dataset_index], 'name', f'Dataset_{dataset_index}')
            if callable(dataset_name):
                dataset_name = dataset_name()
            
            data['_dataset_info'] = {
                'dataset_index': dataset_index,
                'dataset_name': dataset_name,
                'original_index': actual_index,
                'weight': self.weights[dataset_index]
            }
        
        return data
    
    def name(self):
        dataset_names = []
        for i, dataset in enumerate(self.datasets):
            name = getattr(dataset, 'name', f'Dataset_{i}')
            if callable(name):
                name = name()
            dataset_names.append(name)
        return f"EnhancedHybrid({', '.join(dataset_names)})"
    
    def get_dataset_info(self) -> Dict:
        """获取数据集详细信息"""
        info = {
            'total_length': self.total_length,
            'num_datasets': self.num_datasets,
            'dataset_names': [],
            'ratios': self.ratios,
            'weights': self.weights,
            'datasets': []
        }
        
        for i, dataset in enumerate(self.datasets):
            dataset_name = getattr(dataset, 'name', f'Dataset_{i}')
            if callable(dataset_name):
                dataset_name = dataset_name()
            info['dataset_names'].append(dataset_name)
            
            dataset_info = {
                'index': i,
                'name': dataset_name,
                'original_length': self.original_lengths[i],
                'sampling_length': self.sampling_lengths[i],
                'ratio': self.ratios[i],
                'weight': self.weights[i],
                'cumulative_start': 0 if i == 0 else self.cumulative_lengths[i-1],
                'cumulative_end': self.cumulative_lengths[i]
            }
            info['datasets'].append(dataset_info)
        
        return info
    
    def update_ratios(self, new_ratios: Sequence[float]):
        """动态更新数据集比例"""
        if not self.enable_dynamic_resampling:
            raise RuntimeError("动态重采样未启用，请在初始化时设置enable_dynamic_resampling=True")
        
        if len(new_ratios) != self.num_datasets:
            raise ValueError("新比例列表长度必须与数据集数量相同")
        
        if not np.isclose(sum(new_ratios), 1.0, rtol=1e-5):
            logging.warning(f"数据集比例之和不等于1.0: {sum(new_ratios)}，将自动归一化")
            new_ratios = np.array(new_ratios) / sum(new_ratios)
        
        self.ratios = list(new_ratios)
        self._calculate_sampling_lengths()
        self._build_index_mapping()
        
        logging.info("数据集比例已更新")
    
    def update_weights(self, new_weights: Sequence[float]):
        """动态更新数据集权重"""
        if len(new_weights) != self.num_datasets:
            raise ValueError("新权重列表长度必须与数据集数量相同")
        
        self.weights = list(new_weights)
        logging.info("数据集权重已更新")
    
    def shuffle_datasets(self):
        """随机打乱数据集顺序"""
        if not self.shuffle_order:
            logging.warning("数据集顺序打乱未启用")
            return
        
        # 创建随机排列
        indices = list(range(self.num_datasets))
        random.shuffle(indices)
        
        # 重新排列所有相关列表
        self.datasets = [self.datasets[i] for i in indices]
        self.original_lengths = [self.original_lengths[i] for i in indices]
        self.ratios = [self.ratios[i] for i in indices]
        self.weights = [self.weights[i] for i in indices]
        self.sampling_lengths = [self.sampling_lengths[i] for i in indices]
        
        # 重新构建索引映射
        self._build_index_mapping()
        
        logging.info("数据集顺序已随机打乱")
    
    def get_dataset_statistics(self) -> Dict:
        """获取数据集统计信息"""
        stats = {
            'total_samples': self.total_length,
            'datasets': []
        }
    
        for i, dataset in enumerate(self.datasets):
            dataset_name = getattr(dataset, 'name', f'Dataset_{i}')
            dataset_stats = {
                'name': dataset_name,
                'original_samples': self.original_lengths[i],
                'sampled_samples': self.sampling_lengths[i],
                'sampling_ratio': self.ratios[i],
                'weight': self.weights[i],
                'effective_contribution': self.sampling_lengths[i] / self.total_length,
                'oversampling_factor': self.sampling_lengths[i] / self.original_lengths[i]
            }
            stats['datasets'].append(dataset_stats)
        
        return stats


def create_enhanced_hybrid_dataset_from_config(dataset_config: Dict, mode: str, global_config: Dict = None) -> EnhancedHybridDataset:
    """
    从配置创建增强混合数据集
    
    Args:
        dataset_config: 数据集配置字典，包含 enhanced_hybrid 配置
        mode: 数据集模式 ('train', 'test', 'valid')
        global_config: 全局配置字典，用于设置变换等
    
    Returns:
        EnhancedHybridDataset: 创建的增强混合数据集
    """
    from .dataset import get_train_dataset, get_test_dataset, get_valid_dataset
    from pathlib import Path
    from src.config import set_config
    
    # 如果提供了全局配置，设置到全局变量中
    if global_config is not None:
        set_config(global_config)
    
    # 获取 enhanced_hybrid 配置
    enhanced_config = dataset_config.get('enhanced_hybrid', {})
    datasets_list = enhanced_config.get('datasets', [])
    
    if not datasets_list:
        raise ValueError("enhanced_hybrid 配置中没有找到数据集列表")
    
    # 创建数据集列表
    datasets = []
    ratios = []
    priorities = []
    weights = []
    
    has_ratios = False
    has_priorities = False
    has_weights = False
    
    # 处理每个数据集配置
    for config in datasets_list:
        # 根据mode获取对应的数据集
        dataset_name = config.get('class_name', config.get('name', ''))
        root_dir = Path(config['root_dir'])
        
        # 提取数据集特定的参数
        dataset_kwargs = {}
        for key, value in config.items():
            if key not in ['name', 'class_name', 'root_dir', 'train_split', 'test_split', 'valid_split', 'val_split', 'ratio', 'priority', 'weight']:
                dataset_kwargs[key] = value
        
        # 根据模式选择对应的split
        if mode == 'train':
            split = config.get('train_split', 'train')
            dataset = get_train_dataset(dataset_name, root_dir, **dataset_kwargs)
        elif mode == 'test':
            split = config.get('test_split', 'test')
            dataset = get_test_dataset(dataset_name, root_dir, **dataset_kwargs)
        elif mode in ['valid', 'val']:
            split = config.get('valid_split', config.get('val_split', 'valid'))
            dataset = get_valid_dataset(dataset_name, root_dir, **dataset_kwargs)
        else:
            raise ValueError(f"不支持的模式: {mode}")
        
        datasets.append(dataset)
        
        # 收集比例、优先级和权重信息
        if 'ratio' in config:
            ratios.append(config['ratio'])
            has_ratios = True
        else:
            ratios.append(None)
        
        if 'priority' in config:
            priorities.append(config['priority'])
            has_priorities = True
        else:
            priorities.append(None)
        
        if 'weight' in config:
            weights.append(config['weight'])
            has_weights = True
        else:
            weights.append(None)
    
    # 如果部分配置缺失，则使用None
    final_ratios = ratios if has_ratios and all(r is not None for r in ratios) else None
    final_priorities = priorities if has_priorities and all(p is not None for p in priorities) else None
    final_weights = weights if has_weights and all(w is not None for w in weights) else None
    
    # 获取全局设置
    shuffle_order = enhanced_config.get('shuffle_order', False)
    random_seed = enhanced_config.get('random_seed', None)
    enable_dynamic_resampling = enhanced_config.get('enable_dynamic_resampling', False)
    
    return EnhancedHybridDataset(
        datasets=datasets,
        ratios=final_ratios,
        priorities=final_priorities,
        weights=final_weights,
        shuffle_order=shuffle_order,
        random_seed=random_seed,
        enable_dynamic_resampling=enable_dynamic_resampling
    )