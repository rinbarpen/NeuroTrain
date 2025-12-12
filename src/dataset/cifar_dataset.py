import torch
from pathlib import Path
import numpy as np
import yaml
from typing import Literal, Dict, Union, Tuple, Any
from torchvision import datasets, transforms
from torch.utils.data import Subset
import logging

from .custom_dataset import CustomDataset


class CIFARDataset(CustomDataset):
    """CIFAR数据集实现
    
    支持CIFAR-10和CIFAR-100两个数据集，用于图像分类任务。
    
    CIFAR-10:
        - 10个类别
        - 训练集: 50,000张图像
        - 测试集: 10,000张图像
        - 图像尺寸: 32x32x3
        
    CIFAR-100:
        - 100个类别
        - 训练集: 50,000张图像
        - 测试集: 10,000张图像
        - 图像尺寸: 32x32x3
    
    Args:
        root_dir: 数据集根目录
        split: 数据集划分 ('train', 'valid', 'test')
        cifar_type: CIFAR类型 ('cifar10' 或 'cifar100')
        download: 是否自动下载数据集
        valid_ratio: 从训练集中划分验证集的比例
        transform: 数据增强变换
        **kwargs: 其他配置参数
    """
    
    # 数据集统计信息
    # CIFAR-10均值和标准差 (RGB通道)
    MEAN_CIFAR10 = [0.4914, 0.4822, 0.4465]
    STD_CIFAR10 = [0.2023, 0.1994, 0.2010]
    
    # CIFAR-100均值和标准差 (RGB通道)  
    MEAN_CIFAR100 = [0.5071, 0.4867, 0.4408]
    STD_CIFAR100 = [0.2675, 0.2565, 0.2761]
    
    # CIFAR-10类别名称
    CIFAR10_CLASSES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    # CIFAR-100超类
    CIFAR100_COARSE_LABELS = [
        'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
        'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
        'large_man-made_outdoor_things', 'large_natural_outdoor_scenes',
        'large_omnivores_and_herbivores', 'medium_mammals', 'non-insect_invertebrates',
        'people', 'reptiles', 'small_mammals', 'trees', 'vehicles_1', 'vehicles_2'
    ]
    
    # CIFAR-100细粒度类别（100个类别）
    CIFAR100_FINE_LABELS = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
        'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
        'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
        'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
        'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
        'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
        'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
        'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
        'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    ]
    
    mapping = {
        "train": ("train", "train"),
        "valid": ("valid", "valid"),
        "test": ("test", "test")
    }
    
    def __init__(
        self,
        root_dir: Path,
        split: Literal['train', 'valid', 'test'],
        cifar_type: Literal['cifar10', 'cifar100'] = 'cifar10',
        download: bool = True,
        valid_ratio: float = 0.1,
        transform=None,
        target_transform=None,
        return_image_only: bool = False,
        **kwargs
    ):
        """
        初始化CIFAR数据集
        
        Args:
            root_dir: 数据集根目录
            split: 数据集划分
            cifar_type: CIFAR类型
            download: 是否下载
            valid_ratio: 验证集比例
            transform: 数据变换
            target_transform: 标签变换
            return_image_only: 是否只返回图像和标签（简化格式）
            **kwargs: 其他参数
        """
        super(CIFARDataset, self).__init__(root_dir, split, **kwargs)
        
        self.cifar_type = cifar_type.lower()
        self.download = download
        self.valid_ratio = valid_ratio
        self.transform = transform
        self.target_transform = target_transform
        self.return_image_only = return_image_only
        
        # 选择数据集类型
        if self.cifar_type == 'cifar10':
            self.dataset_class = datasets.CIFAR10
            self.num_classes = 10
            self.class_names = self.CIFAR10_CLASSES
        elif self.cifar_type == 'cifar100':
            self.dataset_class = datasets.CIFAR100
            self.num_classes = 100
            self.class_names = None  # CIFAR-100有100个细粒度类别
        else:
            raise ValueError(f"Unsupported CIFAR type: {cifar_type}. Use 'cifar10' or 'cifar100'")
        
        # 加载数据集
        if split in ['train', 'valid']:
            # 训练集和验证集都从CIFAR训练数据中分割
            cifar_dataset = self.dataset_class(
                root=str(root_dir),
                train=True,
                download=download,
                transform=None  # 先不应用transform，后面再应用
            )
            
            # 分割训练集和验证集
            total_size = len(cifar_dataset)
            valid_size = int(total_size * valid_ratio)
            train_size = total_size - valid_size
            
            if split == 'train':
                indices = list(range(0, train_size))
            else:  # valid
                indices = list(range(train_size, total_size))
            
            self.dataset = Subset(cifar_dataset, indices)
            
        else:  # test
            self.dataset = self.dataset_class(
                root=str(root_dir),
                train=False,
                download=download,
                transform=None
            )
        
        self.n = len(self.dataset)
        
        logging.info(f"Loaded {self.cifar_type.upper()} {split} set: {self.n} images")
    
    def __getitem__(self, index: int) -> Union[Dict, Tuple[Any, Any]]:
        """获取指定索引的数据样本"""
        # 获取图像和标签
        if isinstance(self.dataset, Subset):
            # 获取原始数据集
            base_dataset = self.dataset.dataset
            idx = self.dataset.indices[index]
            # 访问CIFAR数据集的data和targets属性
            image_array = base_dataset.data[idx]  # type: ignore
            label = base_dataset.targets[idx]  # type: ignore
            # 转换为PIL图像
            from PIL import Image
            image = Image.fromarray(image_array)
        else:
            image, label = self.dataset[index]
        
        # 应用变换
        if self.transform is not None:
            image = self.transform(image)
        else:
            # 默认转换为tensor
            image = transforms.ToTensor()(image)
        
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        # 如果只返回图像和标签（简化格式）
        if self.return_image_only:
            return image, label
        
        # 返回标准化 Sample，便于训练流程自动识别 inputs/targets
        from src.utils.ndict import Sample
        return Sample(
            inputs=image,
            targets=torch.tensor(label, dtype=torch.long),
            metadata={
                'index': index,
                'class_name': self.get_class_name(label),
                'split': self.split,
                'cifar_type': self.cifar_type
            }
        )
    
    def get_class_name(self, label: int) -> str:
        """获取类别名称"""
        if self.class_names is not None and label < len(self.class_names):
            return self.class_names[label]
        return f"class_{label}"
    
    def get_num_classes(self) -> int:
        """获取类别数量"""
        return self.num_classes
    
    @staticmethod
    def name() -> str:
        """返回数据集名称"""
        return "CIFAR"
    
    @staticmethod
    def get_train_dataset(root_dir: Path, **kwargs) -> "CIFARDataset":
        """获取训练数据集"""
        return CIFARDataset(root_dir, 'train', **kwargs)
    
    @staticmethod
    def get_valid_dataset(root_dir: Path, **kwargs) -> "CIFARDataset":
        """获取验证数据集"""
        return CIFARDataset(root_dir, 'valid', **kwargs)
    
    @staticmethod
    def get_test_dataset(root_dir: Path, **kwargs) -> "CIFARDataset":
        """获取测试数据集"""
        return CIFARDataset(root_dir, 'test', **kwargs)
    
    @staticmethod
    def metadata(cifar_type: str = 'cifar10') -> Dict:
        """获取数据集元数据信息
        
        Args:
            cifar_type: CIFAR类型 ('cifar10' 或 'cifar100')
            
        Returns:
            包含数据集元数据的字典，包括:
                - num_classes: 类别数量
                - class_names: 类别名称列表
                - mean: RGB通道均值
                - std: RGB通道标准差
                - image_size: 图像尺寸
                - task_type: 任务类型
                - metrics: 推荐使用的评估指标
        """
        cifar_type = cifar_type.lower()
        
        if cifar_type == 'cifar10':
            return {
                'num_classes': 10,
                'class_names': CIFARDataset.CIFAR10_CLASSES,
                'mean': CIFARDataset.MEAN_CIFAR10,
                'std': CIFARDataset.STD_CIFAR10,
                'image_size': (32, 32, 3),
                'task_type': 'classification',
                'metrics': ['accuracy', 'top1_accuracy', 'precision', 'recall', 'f1_score', 'confusion_matrix'],
                'num_train': 50000,
                'num_test': 10000,
                'dataset_name': 'CIFAR-10'
            }
        elif cifar_type == 'cifar100':
            return {
                'num_classes': 100,
                'class_names': CIFARDataset.CIFAR100_FINE_LABELS,
                'coarse_labels': CIFARDataset.CIFAR100_COARSE_LABELS,
                'num_coarse_classes': 20,
                'mean': CIFARDataset.MEAN_CIFAR100,
                'std': CIFARDataset.STD_CIFAR100,
                'image_size': (32, 32, 3),
                'task_type': 'classification',
                'metrics': ['accuracy', 'top1_accuracy', 'top5_accuracy', 'precision', 'recall', 'f1_score'],
                'num_train': 50000,
                'num_test': 10000,
                'dataset_name': 'CIFAR-100'
            }
        else:
            raise ValueError(f"Unsupported CIFAR type: {cifar_type}. Use 'cifar10' or 'cifar100'")


class CIFAR10Dataset(CIFARDataset):
    """CIFAR-10数据集
    
    便捷类，固定使用CIFAR-10
    """
    
    # CIFAR-10专用统计信息
    MEAN = CIFARDataset.MEAN_CIFAR10
    STD = CIFARDataset.STD_CIFAR10
    NUM_CLASSES = 10
    CLASSES = CIFARDataset.CIFAR10_CLASSES
    
    def __init__(self, root_dir: Path, split: Literal['train', 'valid', 'test'], **kwargs):
        kwargs['cifar_type'] = 'cifar10'
        super().__init__(root_dir, split, **kwargs)
    
    @staticmethod
    def name() -> str:
        return "CIFAR10"
    
    @staticmethod
    def get_train_dataset(root_dir: Path, **kwargs):
        return CIFAR10Dataset(root_dir, 'train', **kwargs)
    
    @staticmethod
    def get_valid_dataset(root_dir: Path, **kwargs):
        return CIFAR10Dataset(root_dir, 'valid', **kwargs)
    
    @staticmethod
    def get_test_dataset(root_dir: Path, **kwargs):
        return CIFAR10Dataset(root_dir, 'test', **kwargs)
    
    @staticmethod
    def metadata(cifar_type: str = 'cifar10') -> Dict:
        """获取CIFAR-10数据集元数据"""
        return CIFARDataset.metadata('cifar10')


class CIFAR100Dataset(CIFARDataset):
    """CIFAR-100数据集
    
    便捷类，固定使用CIFAR-100
    """
    
    # CIFAR-100专用统计信息
    MEAN = CIFARDataset.MEAN_CIFAR100
    STD = CIFARDataset.STD_CIFAR100
    NUM_CLASSES = 100
    FINE_LABELS = CIFARDataset.CIFAR100_FINE_LABELS
    COARSE_LABELS = CIFARDataset.CIFAR100_COARSE_LABELS
    NUM_COARSE_CLASSES = 20
    
    def __init__(self, root_dir: Path, split: Literal['train', 'valid', 'test'], **kwargs):
        kwargs['cifar_type'] = 'cifar100'
        super().__init__(root_dir, split, **kwargs)
    
    @staticmethod
    def name() -> str:
        return "CIFAR100"
    
    @staticmethod
    def get_train_dataset(root_dir: Path, **kwargs):
        return CIFAR100Dataset(root_dir, 'train', **kwargs)
    
    @staticmethod
    def get_valid_dataset(root_dir: Path, **kwargs):
        return CIFAR100Dataset(root_dir, 'valid', **kwargs)
    
    @staticmethod
    def get_test_dataset(root_dir: Path, **kwargs):
        return CIFAR100Dataset(root_dir, 'test', **kwargs)
    
    @staticmethod
    def metadata(cifar_type: str = 'cifar100') -> Dict:
        """获取CIFAR-100数据集元数据"""
        return CIFARDataset.metadata('cifar100')


def get_cifar_transforms(
    split: str = 'train', 
    image_size: int = 32, 
    cifar_type: str = 'cifar10'
) -> transforms.Compose:
    """获取CIFAR数据增强变换
    
    Args:
        split: 数据集划分
        image_size: 目标图像大小
        cifar_type: CIFAR类型 ('cifar10' 或 'cifar100')
    
    Returns:
        transforms组合
    """
    # 根据数据集类型选择均值和标准差
    if cifar_type.lower() == 'cifar10':
        mean = CIFARDataset.MEAN_CIFAR10
        std = CIFARDataset.STD_CIFAR10
    else:
        mean = CIFARDataset.MEAN_CIFAR100
        std = CIFARDataset.STD_CIFAR100
    
    if split == 'train':
        # 训练集数据增强
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.Resize(image_size) if image_size != 32 else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        # 验证集/测试集
        return transforms.Compose([
            transforms.Resize(image_size) if image_size != 32 else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

