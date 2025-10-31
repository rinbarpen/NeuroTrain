import torch
from torchvision import transforms
from pathlib import Path
from PIL import Image
import yaml
import numpy as np
from typing import Literal

from ..custom_dataset import CustomDataset, Betweens

class StareDataset(CustomDataset):
    mapping = {
        "train": ("training/images/*.png", "training/1st_labels_ah/*.png"),
        "valid": ("test/images/*.png", "test/1st_labels_ah/*.png"),
        "test": ("test/images/*.png", "test/1st_labels_ah/*.png"),
    }

    def _get_transforms(self):
        """
        获取预处理/增强变换；默认仅 ToTensor。
        说明：
            - transforms 在具体数据集类内实现，避免在 CustomDataset 中耦合。
        """
        return transforms.ToTensor()

    def __init__(self, root_dir: Path, split: Literal['train', 'test'], is_rgb: bool = False, **kwargs):
        """
        STARE 数据集
        
        Args:
            root_dir: 数据集根目录
            split: 数据集类型 ('train' 或 'test')
            is_rgb: 是否以RGB方式读取图像（默认灰度）
            **kwargs: 预留扩展参数
        """
        super(StareDataset, self).__init__(root_dir, split, **kwargs)

        if 'source' in kwargs and 'target' in kwargs:
            image_glob, label_glob = kwargs['source'], kwargs['target']
        else:
            image_glob, label_glob = self.mapping[split]

        self.transforms = self._get_transforms()
        self.config = {"is_rgb": is_rgb, "source": image_glob, "target": label_glob}

        images = [p for p in self.root_dir.glob(image_glob)]
        masks = [p for p in self.root_dir.glob(label_glob)]

        self.images = sorted(images)
        self.masks = sorted(masks)
        self.n = len(self.images)

    def __getitem__(self, index: int):
        """返回图像和掩码张量"""
        image_path, mask_path = self.images[index], self.masks[index]
        if self.config['is_rgb']:
            image = Image.open(image_path).convert('RGB')
            mask = Image.open(mask_path).convert('RGB')
        else:
            image = Image.open(image_path).convert('L')
            mask = Image.open(mask_path).convert('L')

        image, mask = self.transforms(image), self.transforms(mask)
        return {
            'image': image,  # 图像张量
            'mask': mask,    # 掩码张量
            'metadata': {
                'image_path': str(image_path),
                'mask_path': str(mask_path),
                'is_rgb': self.config['is_rgb'],
                'split': self.dataset_type
            }
        }

    @staticmethod
    def name():
        return "STARE"
    
    @staticmethod
    def metadata(**kwargs):
        """获取STARE数据集元数据"""
        return {
            'num_classes': 2,
            'class_names': ['background', 'vessel'],
            'task_type': 'segmentation',
            'metrics': ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity'],
            'image_size': (700, 605, 3),
            'num_train': 10,
            'num_test': 10,
            'dataset_name': 'STARE',
            'description': 'Structured Analysis of the Retina'
        }

    @staticmethod
    def get_train_dataset(root_dir: Path, **kwargs):
        """获取训练集实例"""
        return StareDataset(root_dir, 'train', **kwargs)

    @staticmethod
    def get_valid_dataset(root_dir: Path, **kwargs):
        """STARE 无验证集，返回 None"""
        return None

    @staticmethod
    def get_test_dataset(root_dir: Path, **kwargs):
        """获取测试集实例"""
        return StareDataset(root_dir, 'test', **kwargs)
