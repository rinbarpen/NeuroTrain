import torch
from pathlib import Path
import numpy as np
import yaml
from typing import Literal
from torchvision import transforms, datasets
from torch.utils.data import Subset

from .custom_dataset import CustomDataset, Betweens

class MNISTDataset(CustomDataset):
    """MNIST数据集实现
    
    用于验证框架是否可以正确运行，提供基础的图像分割任务示例。
    """
    
    mapping = {
        "train": ("train", "train"), 
        "valid": ("valid", "valid"),
        "test": ("test", "test")
    }
    
    def __init__(self, root_dir: Path, split: Literal['train', 'test', 'valid'], 
                 download=True, **kwargs):
        """
        初始化MNIST数据集
        
        Args:
            root_dir: 数据集根目录  
            split: 数据集类型
            download: 是否下载数据集
            **kwargs: 其他配置参数
                - enable_cache: 是否启用缓存，默认True
        """
        super(MNISTDataset, self).__init__(root_dir, split, **kwargs)
        
        # 如果从缓存加载成功，直接返回
        if self._cache_loaded:
            return
        
        self.download = download
        
        # 加载MNIST数据集
        if split in ['train', 'valid']:
            # 训练集和验证集都从MNIST训练数据中分割
            mnist_dataset = datasets.MNIST(root=str(root_dir), train=True, download=download, transform=None)
            
            # 分割训练集和验证集 (50000 train, 10000 valid)
            if split == 'train':
                indices = list(range(0, 50000))
            else:  # valid
                indices = list(range(50000, 60000))
                
            self.dataset = Subset(mnist_dataset, indices)
        else:  # test
            self.dataset = datasets.MNIST(root=str(root_dir), train=False, download=download, transform=None)
            
        self.n = len(self.dataset)
        
        # 将dataset转换为samples列表以便缓存
        self.samples = [(img, label) for img, label in self.dataset]
        
        # 自动保存到缓存
        self._save_to_cache_if_needed()
        
    def __getitem__(self, index):
        """获取指定索引的数据样本"""
        # 从缓存加载的情况
        if self._cache_loaded and hasattr(self, 'samples'):
            image, label = self.samples[index]
        else:
            # 从dataset加载
            image, label = self.dataset[index]
        
        # 转换为numpy数组并归一化
        image = np.array(image, dtype=np.float32) / 255.0
        
        # 为了兼容分割任务，将标签转换为mask格式
        label_mask = np.zeros((28, 28), dtype=np.float32)
        label_mask[image > 0.5] = float(label) / 9.0  # 归一化到[0,1]
        
        # 转换为tensor并增加通道维度
        image_tensor = torch.from_numpy(image).unsqueeze(0)
        mask_tensor = torch.from_numpy(label_mask).unsqueeze(0)
        
        return {
            'image': image_tensor,  # 图像张量 (1, 28, 28)
            'mask': mask_tensor,    # 掩码张量 (1, 28, 28)
            'metadata': {
                'label': int(label),  # 原始数字标签
                'split': self.split,
                'download': getattr(self, 'download', True)
            }
        }
    
    @staticmethod
    def name():
        return "MNIST"
    
    @staticmethod
    def metadata(**kwargs):
        """获取MNIST数据集元数据"""
        return {
            'num_classes': 10,
            'class_names': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            'image_size': (28, 28, 1),
            'task_type': 'classification',  # 或 'segmentation' 取决于使用方式
            'metrics': ['accuracy', 'precision', 'recall', 'f1_score'],
            'num_train': 50000,
            'num_valid': 10000,
            'num_test': 10000,
            'dataset_name': 'MNIST'
        }
    
    @staticmethod
    def get_train_dataset(root_dir: Path, **kwargs):
        """获取训练数据集"""
        return MNISTDataset(root_dir, 'train', **kwargs)
    
    @staticmethod
    def get_valid_dataset(root_dir: Path, **kwargs):
        """获取验证数据集"""
        return MNISTDataset(root_dir, 'valid', **kwargs)
    
    @staticmethod
    def get_test_dataset(root_dir: Path, **kwargs):
        """获取测试数据集"""
        return MNISTDataset(root_dir, 'test', **kwargs)
