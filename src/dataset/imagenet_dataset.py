import torch
from pathlib import Path
import numpy as np
from typing import Literal, Optional, List, Dict, Tuple, Union
from PIL import Image
import logging
import json

from .custom_dataset import CustomDataset, Betweens


class ImageNet1KDataset(CustomDataset):
    """ImageNet-1K (ILSVRC2012) 数据集实现
    
    ImageNet-1K 是大规模图像分类数据集，包含1000个类别，约128万张训练图像，
    5万张验证图像，10万张测试图像。
    
    数据集结构:
        root_dir/
            train/
                n01440764/
                    n01440764_10026.JPEG
                    ...
                n01443537/
                    ...
                ...
            val/
                n01440764/
                    ILSVRC2012_val_00000293.JPEG
                    ...
                n01443537/
                    ...
                ...
            或者（验证集的另一种结构）:
            val/
                ILSVRC2012_val_00000001.JPEG
                ILSVRC2012_val_00000002.JPEG
                ...
            val_annotations.txt 或 val.txt  (如果验证集是扁平结构)
    
    Args:
        root_dir: ImageNet数据集根目录
        split: 数据集划分 ('train', 'val', 'test', 'valid')
        transform: 数据变换函数
        target_transform: 标签变换函数
        return_path: 是否返回图像路径
        class_to_idx_file: 类别到索引的映射文件路径（可选）
        **kwargs: 其他配置参数
    """
    
    mapping = {
        "train": ("train", "train"),
        "valid": ("val", "val"),
        "test": ("test", "test")
    }
    
    def __init__(
        self, 
        root_dir: Union[str, Path], 
        split: Literal['train', 'val', 'test', 'valid'],
        transform=None,
        target_transform=None,
        return_path: bool = False,
        class_to_idx_file: Optional[str] = None,
        **kwargs
    ):
        """
        初始化ImageNet数据集
        
        Args:
            root_dir: 数据集根目录
            split: 数据集划分
            transform: 图像变换
            target_transform: 标签变换
            return_path: 是否返回图像路径
            class_to_idx_file: 类别到索引映射文件
            **kwargs: 其他配置参数
        """
        super(ImageNet1KDataset, self).__init__(root_dir, split, **kwargs)
        
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.return_path = return_path
        
        # 根据split确定数据目录
        if split == 'valid':
            split_dir = 'val'
        else:
            split_dir = split
        
        self.data_dir = self.root_dir / split_dir
        
        # 检查数据目录是否存在
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}\n"
                f"Please download ImageNet dataset and place it in {root_dir}"
            )
        
        # 加载类别到索引的映射
        if class_to_idx_file and Path(class_to_idx_file).exists():
            with open(class_to_idx_file, 'r') as f:
                self.class_to_idx = json.load(f)
        else:
            self.class_to_idx = self._build_class_to_idx()
        
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # 加载数据集样本
        self.samples = self._load_samples()
        self.n = len(self.samples)
        
        logging.info(f"Loaded {self.n} images from ImageNet {split} set")
        logging.info(f"Number of classes: {len(self.class_to_idx)}")
    
    def _build_class_to_idx(self) -> Dict[str, int]:
        """构建类别到索引的映射"""
        # 获取所有类别目录
        class_dirs = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        
        if not class_dirs:
            # 如果没有类别目录，尝试从文件名中提取类别
            logging.warning(f"No class directories found in {self.data_dir}")
            return {}
        
        # 创建类别到索引的映射
        class_to_idx = {class_name: idx for idx, class_name in enumerate(class_dirs)}
        
        return class_to_idx
    
    def _load_samples(self) -> List[Tuple[Path, int]]:
        """加载数据集样本列表"""
        samples = []
        
        # 检查是否是分层结构（每个类别一个目录）
        class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        if class_dirs:
            # 分层结构
            for class_dir in class_dirs:
                class_name = class_dir.name
                if class_name not in self.class_to_idx:
                    logging.warning(f"Class {class_name} not in class_to_idx mapping, skipping")
                    continue
                
                class_idx = self.class_to_idx[class_name]
                
                # 获取该类别下的所有图像
                image_files = list(class_dir.glob('*.JPEG')) + \
                             list(class_dir.glob('*.jpg')) + \
                             list(class_dir.glob('*.png'))
                
                for img_path in image_files:
                    samples.append((img_path, class_idx))
        else:
            # 扁平结构，需要标注文件
            ann_file = self.root_dir / f'{self.split}_annotations.txt'
            if not ann_file.exists():
                ann_file = self.root_dir / f'{self.split}.txt'
            
            if not ann_file.exists():
                raise FileNotFoundError(
                    f"Annotation file not found: {ann_file}\n"
                    f"For flat directory structure, please provide annotation file"
                )
            
            # 读取标注文件
            with open(ann_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        img_name, class_name = parts[0], parts[1]
                        img_path = self.data_dir / img_name
                        
                        if class_name not in self.class_to_idx:
                            logging.warning(f"Class {class_name} not in class_to_idx mapping, skipping")
                            continue
                        
                        class_idx = self.class_to_idx[class_name]
                        
                        if img_path.exists():
                            samples.append((img_path, class_idx))
        
        return samples
    
    def __getitem__(self, index: int) -> Dict:
        """获取指定索引的数据样本"""
        img_path, class_idx = self.samples[index]
        
        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
            # 返回一个黑色图像作为fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # 应用变换
        if self.transform is not None:
            image = self.transform(image)
        else:
            # 默认转换为tensor
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # 标签
        label = class_idx
        if self.target_transform is not None:
            label = self.target_transform(label)
        else:
            label = torch.tensor(label, dtype=torch.long)
        
        result = {
            'image': image,
            'label': label,
            'target': label,  # 为了兼容性
            'metadata': {
                'index': index,
                'class_name': self.idx_to_class[class_idx],
                'class_idx': class_idx,
                'split': self.split,
            }
        }
        
        if self.return_path:
            result['metadata']['path'] = str(img_path)
        
        return result
    
    @staticmethod
    def name() -> str:
        """返回数据集名称"""
        return "ImageNet1K"
    
    @staticmethod
    def metadata(**kwargs) -> Dict:
        """获取ImageNet-1K数据集元数据"""
        return {
            'num_classes': 1000,
            'dataset_name': 'ImageNet1K',
            'task_type': 'classification',
            'num_train': 1281167,
            'num_val': 50000,
            'num_test': 100000,
            'image_size': 224,
            'metrics': ['top1_accuracy', 'top5_accuracy', 'loss'],
            'mean': [0.485, 0.456, 0.406],  # ImageNet normalization mean
            'std': [0.229, 0.224, 0.225],   # ImageNet normalization std
        }
    
    @staticmethod
    def get_train_dataset(root_dir: Union[str, Path], **kwargs):
        """获取训练数据集"""
        return ImageNet1KDataset(root_dir, 'train', **kwargs)
    
    @staticmethod
    def get_valid_dataset(root_dir: Union[str, Path], **kwargs):
        """获取验证数据集"""
        return ImageNet1KDataset(root_dir, 'val', **kwargs)
    
    @staticmethod
    def get_test_dataset(root_dir: Union[str, Path], **kwargs):
        """获取测试数据集"""
        return ImageNet1KDataset(root_dir, 'test', **kwargs)
    
    def get_class_name(self, class_idx: int) -> str:
        """根据类别索引获取类别名称"""
        return self.idx_to_class.get(class_idx, 'unknown')
    
    def get_class_idx(self, class_name: str) -> int:
        """根据类别名称获取类别索引"""
        return self.class_to_idx.get(class_name, -1)

