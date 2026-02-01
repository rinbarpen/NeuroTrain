import torch
from pathlib import Path
from PIL import Image
import numpy as np
import yaml
from typing import Literal, Union
from torchvision import transforms

from ..custom_dataset import CustomDataset

class ISIC2017Dataset(CustomDataset):
    mapping = {
        "train": ("2017_Training_Data/*.jpg", "2017_Training_Part1_GroundTruth/*.jpg"),
        "valid": ("ISIC-2017_Validation_Data/*.jpg", "ISIC-2017_Validation_Part1_GroundTruth/*.jpg"),
        "test": ("ISIC-2017_Test_v2_Data/*.jpg", "ISIC-2017_Test_v2_Part1_GroundTruth/*.jpg"),
    }

    def _get_transforms(self):
        """
        获取数据增强/预处理变换管道
        返回:
            - 一个可调用的图像变换（默认使用统一的变换配置）
        说明:
            - transforms 的定义由具体数据集维护，避免在 CustomDataset 中耦合。
        """
        from src.utils.transform import get_transforms
        return get_transforms()

    def __init__(self, root_dir: Union[str, Path], split: Literal['train', 'test', 'valid'], is_rgb: bool = False, **kwargs):
        """
        ISIC 2017 皮肤病变分割数据集
        
        Args:
            root_dir: 数据集根目录
            split: 'train' | 'valid' | 'test'
            is_rgb: 是否以RGB方式读取图像（默认灰度）
            **kwargs: 预留扩展参数，支持自定义source/target
        """
        super(ISIC2017Dataset, self).__init__(root_dir, split, **kwargs)

        if 'source' in kwargs and 'target' in kwargs:
            image_glob, label_glob = kwargs['source'], kwargs['target']
        else:
            image_glob, label_glob = self.mapping[split]

        # 默认仅做张量化转换（通过 _get_transforms 提供，便于后续在本类内扩展）
        self.transforms = self._get_transforms()
        self.config = {"is_rgb": is_rgb, "source": image_glob, "target": label_glob}

        images = [p for p in self.root_dir.glob(image_glob)]
        masks = [p for p in self.root_dir.glob(label_glob)]

        # 不再进行between切片，直接使用完整数据集
        self.images = sorted(images)
        self.masks = sorted(masks)
        self.n = len(self.images)

    def __getitem__(self, index: int):
        """返回图像与掩码张量"""
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
                'task_type': 'segmentation',
                'split': self.split
            }
        }

    @staticmethod
    def name():
        return "ISIC2017"
    
    @staticmethod
    def metadata(**kwargs):
        """获取ISIC2017数据集元数据"""
        return {
            'num_classes': 2,
            'class_names': ['background', 'lesion'],
            'task_type': 'segmentation',
            'metrics': ['dice', 'iou', 'accuracy', 'jaccard'],
            'num_train': 2000,
            'num_val': 150,
            'num_test': 600,
            'dataset_name': 'ISIC2017',
            'description': 'ISIC 2017 Skin Lesion Analysis Challenge'
        }

    @staticmethod
    def get_train_dataset(root_dir: Union[str, Path], **kwargs):
        """获取训练集实例"""
        return ISIC2017Dataset(root_dir, 'train', **kwargs)

    @staticmethod
    def get_valid_dataset(root_dir: Union[str, Path], **kwargs):
        """获取验证集实例"""
        return ISIC2017Dataset(root_dir, 'valid', **kwargs)

    @staticmethod
    def get_test_dataset(root_dir: Union[str, Path], **kwargs):
        """获取测试集实例"""
        return ISIC2017Dataset(root_dir, 'test', **kwargs)
