import torch
from pathlib import Path
from PIL import Image
import numpy as np
import yaml
from typing import Literal
from torchvision import transforms, datasets
from torch.utils.data import Subset

from .custom_dataset import CustomDataset, Betweens

class MNISTDataset(CustomDataset):
    """MNIST数据集实现，用于验证框架是否可以正确运行"""
    
    mapping = {
        "train": ("train", "train"), 
        "valid": ("valid", "valid"),
        "test": ("test", "test")
    }
    
    def __init__(self, base_dir: Path, dataset_type: Literal['train', 'test', 'valid'], 
                 between: tuple[float, float]=(0.0, 1.0), transforms: transforms.Compose|None=None, 
                 use_numpy=False, download=True, **kwargs):
        super(MNISTDataset, self).__init__(base_dir, dataset_type, between, use_numpy=use_numpy)
        
        self.transforms = transforms
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
        
        # 应用between参数进行进一步分割
        total_len = len(self.dataset)
        slice_obj = self.get_slice(total_len)
        
        if isinstance(self.dataset, Subset):
            # 如果是Subset，需要重新计算indices
            original_indices = self.dataset.indices[slice_obj]
            self.dataset = Subset(self.dataset.dataset, original_indices)
        else:
            # 如果是完整数据集，直接创建Subset
            indices = list(range(total_len))[slice_obj]
            self.dataset = Subset(self.dataset, indices)
            
        self.n = len(self.dataset)
        
    def __getitem__(self, index):
        if self.use_numpy:
            # 如果使用numpy格式
            image, label = self.dataset[index]
            image = np.array(image, dtype=np.float32) / 255.0
            # 为了兼容分割任务，将标签转换为mask格式
            label_mask = np.zeros((28, 28), dtype=np.float32)
            label_mask[image > 0.5] = float(label) / 9.0  # 归一化到[0,1]
            return torch.from_numpy(image).unsqueeze(0), torch.from_numpy(label_mask).unsqueeze(0)
        
        image, label = self.dataset[index]
        
        # 转换为PIL图像以便应用transforms
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))
            
        # 创建简单的分割mask（用于验证分割任务）
        label_mask = Image.fromarray(np.array(image))
        
        if self.transforms:
            image = self.transforms(image)
            label_mask = self.transforms(label_mask)
        else:
            # 默认转换
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)
            label_mask = to_tensor(label_mask)
            
        return image, label_mask
    
    @staticmethod
    def to_numpy(save_dir: Path, base_dir: Path, betweens: Betweens, **kwargs):
        """将数据集转换为numpy格式保存"""
        save_dir = save_dir / MNISTDataset.name()
        save_dir.mkdir(parents=True, exist_ok=True)
        
        train_dataset = MNISTDataset.get_train_dataset(base_dir, between=betweens['train'], **kwargs)
        valid_dataset = MNISTDataset.get_valid_dataset(base_dir, between=betweens['valid'], **kwargs)
        test_dataset = MNISTDataset.get_test_dataset(base_dir, between=betweens['test'], **kwargs)
        
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
            yaml.dump({"betweens": betweens, **kwargs}, f)
    
    @staticmethod
    def name():
        return "MNIST"
    
    @staticmethod
    def get_train_dataset(base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        return MNISTDataset(base_dir, 'train', between, use_numpy=use_numpy, **kwargs)
    
    @staticmethod
    def get_valid_dataset(base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        return MNISTDataset(base_dir, 'valid', between, use_numpy=use_numpy, **kwargs)
    
    @staticmethod
    def get_test_dataset(base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        return MNISTDataset(base_dir, 'test', between, use_numpy=use_numpy, **kwargs)