import torch
import yaml
from PIL import Image
import numpy as np
from pathlib import Path
from torchvision import transforms
from typing import Literal

from ..custom_dataset import CustomDataset, Betweens

class ChaseDB1Dataset(CustomDataset):
    mapping = {
        "train": ("training/images/*.jpg", "training/1st_label/*.png"),
        "test": ("test/images/*.jpg", "test/1st_label/*.png"),
    }

    def _get_transforms(self):
        """
        获取预处理/增强变换；默认仅 ToTensor。
        说明：
            - transforms 逻辑在具体数据集类内维护，避免在基类中耦合。
        """
        return transforms.ToTensor()

    def __init__(self, root_dir: Path, split: Literal['train', 'test'], is_rgb: bool = False, **kwargs):
        """
        CHASE_DB1 数据集
        
        Args:
            root_dir: 数据集根目录
            split: 数据集类型 ('train' 或 'test')
            is_rgb: 是否以RGB方式读取图像（默认灰度）
            **kwargs: 预留扩展参数
        """
        super(ChaseDB1Dataset, self).__init__(root_dir, split, **kwargs)

        if 'source' in kwargs and 'target' in kwargs:
            image_glob, label_glob = kwargs['source'], kwargs['target']
        else:
            image_glob, label_glob = self.mapping[split]

        # 默认变换：仅张量化，通过 _get_transforms() 提供
        self.transforms = self._get_transforms()
        self.config = {"is_rgb": is_rgb, "source": image_glob, "target": label_glob}

        images = [p for p in self.root_dir.glob(image_glob)]
        masks = [p for p in self.root_dir.glob(label_glob)]

        # 不再切片，完整使用
        self.images = images
        self.masks = masks
        self.n = len(self.images)

    def __getitem__(self, index: int):
        """返回指定索引的图像和分割掩码张量"""
        image_path, mask_path = self.images[index], self.masks[index]

        # 读取图像和掩码
        if self.config['is_rgb']:
            image = Image.open(image_path).convert('RGB')
            mask = Image.open(mask_path).convert('RGB')
        else:
            image = Image.open(image_path).convert('L')
            mask = Image.open(mask_path).convert('L')

        # 应用变换
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
        return "CHASEDB1"
    
    @staticmethod
    def metadata(**kwargs):
        """获取ChaseDB1数据集元数据"""
        return {
            'num_classes': 2,
            'class_names': ['background', 'vessel'],
            'task_type': 'segmentation',
            'metrics': ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity'],
            'image_size': (999, 960, 3),
            'num_train': 20,
            'num_test': 8,
            'dataset_name': 'ChaseDB1',
            'description': 'CHASE_DB1 retinal vessel segmentation dataset'
        }

    @staticmethod
    def get_train_dataset(root_dir: Path, **kwargs):
        """获取训练集实例"""
        return ChaseDB1Dataset(root_dir, 'train', **kwargs)

    @staticmethod
    def get_valid_dataset(root_dir: Path, **kwargs):
        """CHASE_DB1 无验证集，返回 None"""
        return None

    @staticmethod
    def get_test_dataset(root_dir: Path, **kwargs): 
        """获取测试集实例"""
        return ChaseDB1Dataset(root_dir, 'test', **kwargs)
