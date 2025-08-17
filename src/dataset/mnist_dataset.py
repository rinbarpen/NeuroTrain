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
    
    def __init__(self, base_dir: Path, dataset_type: Literal['train', 'test', 'valid'], 
                 download=True, **kwargs):
        """
        初始化MNIST数据集
        
        Args:
            base_dir: 数据集根目录  
            dataset_type: 数据集类型
            download: 是否下载数据集
            **kwargs: 其他配置参数
        """
        super(MNISTDataset, self).__init__(base_dir, dataset_type, **kwargs)
        
        self.download = download
        
        # 加载MNIST数据集
        if dataset_type in ['train', 'valid']:
            # 训练集和验证集都从MNIST训练数据中分割
            mnist_dataset = datasets.MNIST(root=str(base_dir), train=True, download=download, transform=None)
            
            # 分割训练集和验证集 (50000 train, 10000 valid)
            if dataset_type == 'train':
                indices = list(range(0, 50000))
            else:  # valid
                indices = list(range(50000, 60000))
                
            self.dataset = Subset(mnist_dataset, indices)
        else:  # test
            self.dataset = datasets.MNIST(root=str(base_dir), train=False, download=download, transform=None)
            
        self.n = len(self.dataset)
        
    def __getitem__(self, index):
        """获取指定索引的数据样本"""
        image, label = self.dataset[index]
        
        # 转换为numpy数组并归一化
        image = np.array(image, dtype=np.float32) / 255.0
        
        # 为了兼容分割任务，将标签转换为mask格式
        label_mask = np.zeros((28, 28), dtype=np.float32)
        label_mask[image > 0.5] = float(label) / 9.0  # 归一化到[0,1]
        
        # 转换为tensor并增加通道维度
        image_tensor = torch.from_numpy(image).unsqueeze(0)
        mask_tensor = torch.from_numpy(label_mask).unsqueeze(0)
        
        return image_tensor, mask_tensor
    
    @staticmethod
    def to_numpy(save_dir: Path, base_dir: Path, betweens: Betweens, **kwargs):
        """将数据集转换为numpy格式保存
        
        Args:
            save_dir: 保存目录
            base_dir: 数据集根目录
            betweens: 兼容性参数，实际不再使用
            **kwargs: 其他配置参数
        """
        save_dir = save_dir / MNISTDataset.name()
        save_dir.mkdir(parents=True, exist_ok=True)
        
        train_dataset = MNISTDataset.get_train_dataset(base_dir, **kwargs)
        valid_dataset = MNISTDataset.get_valid_dataset(base_dir, **kwargs)
        test_dataset = MNISTDataset.get_test_dataset(base_dir, **kwargs)
        
        from torch.utils.data import DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)
        valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
        
        for dataloader, dataset_type in zip(
            (train_dataloader, valid_dataloader, test_dataloader), 
            ('train', 'valid', 'test')
        ):
            type_dir = save_dir / dataset_type
            type_dir.mkdir(parents=True, exist_ok=True)
            
            for i, (image, mask) in enumerate(dataloader):
                image_path = type_dir / f'image_{i:05d}.npy'
                mask_path = type_dir / f'mask_{i:05d}.npy'
                
                np.save(image_path, image.numpy())
                np.save(mask_path, mask.numpy())
        
        # 保存配置文件
        config_file = save_dir / "config.yaml"
        with config_file.open('w', encoding='utf-8') as f:
            yaml.dump({**kwargs}, f)
    
    @staticmethod
    def name():
        return "MNIST"
    
    @staticmethod
    def get_train_dataset(base_dir: Path, **kwargs):
        """获取训练数据集"""
        return MNISTDataset(base_dir, 'train', **kwargs)
    
    @staticmethod
    def get_valid_dataset(base_dir: Path, **kwargs):
        """获取验证数据集"""
        return MNISTDataset(base_dir, 'valid', **kwargs)
    
    @staticmethod
    def get_test_dataset(base_dir: Path, **kwargs):
        """获取测试数据集"""
        return MNISTDataset(base_dir, 'test', **kwargs)