from pathlib import Path
from PIL import Image
from typing import Literal

from .custom_dataset import CustomDataset, Betweens
from src.utils.ndict import Sample

class DriveDataset(CustomDataset):
    mapping = {
        "train": ("training/images/*.png", "training/1st_manual/*.png"),
        "test": ("test/images/*.png", "test/1st_manual/*.png"),
    }

    def __init__(self, root_dir: Path, split: Literal['train', 'valid', 'test'], is_rgb: bool = False, **kwargs):
        """
        DRIVE 数据集实现
        
        Args:
            root_dir: 数据集根目录
            split: 数据集类型 ('train', 'valid' 或 'test')
            is_rgb: 是否以RGB方式读取图像（默认灰度）
            **kwargs: 预留扩展参数
        """
        super(DriveDataset, self).__init__(root_dir, split, **kwargs)

        if 'source' in kwargs and 'target' in kwargs:
            image_glob, label_glob = kwargs['source'], kwargs['target']
        else:
            image_glob, label_glob = self.mapping[split]

        # 默认的图像变换，仅进行张量化
        self.transforms = self._get_transforms()
        self.config = {"is_rgb": is_rgb, "source": image_glob, "target": label_glob}

        images = [p for p in root_dir.glob(image_glob)]
        masks = [p for p in root_dir.glob(label_glob)]

        self.images = sorted(images)
        self.masks = sorted(masks)
        self.n = len(self.images)

    def __getitem__(self, index: int):
        """返回指定索引的图像和分割掩码张量"""
        image_path, mask_path = self.images[index], self.masks[index]

        # 根据配置选择RGB或灰度读取
        if self.config['is_rgb']:
            image = Image.open(image_path).convert('RGB')
            mask = Image.open(mask_path).convert('RGB')
        else:
            image = Image.open(image_path).convert('L')
            mask = Image.open(mask_path).convert('L')

        image, mask = self.transforms(image), self.transforms(mask)

        sample = Sample(inputs=image, targets=mask)
        return sample

    @staticmethod
    def name():
        return "DRIVE"
    
    @staticmethod
    def metadata(**kwargs):
        """获取DRIVE数据集元数据"""
        return {
            'num_classes': 2,
            'class_names': ['background', 'vessel'],
            'task_type': 'segmentation',
            'metrics': ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity'],
            'image_size': (584, 565, 3),  # 原始图像尺寸
            'num_train': 20,
            'num_test': 20,
            'dataset_name': 'DRIVE',
            'description': 'Digital Retinal Images for Vessel Extraction'
        }

    @staticmethod
    def get_train_dataset(root_dir: Path, **kwargs):
        return DriveDataset(root_dir, 'train', **kwargs)

    @staticmethod
    def get_valid_dataset(root_dir: Path, **kwargs):
        return DriveDataset(root_dir, 'test', **kwargs)

    @staticmethod
    def get_test_dataset(root_dir: Path, **kwargs):
        return DriveDataset(root_dir, 'test', **kwargs)

    def _get_transforms(self):
        """获取数据集的变换"""
        from src.utils.transform import get_transforms
        return get_transforms()