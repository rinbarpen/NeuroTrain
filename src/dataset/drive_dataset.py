import torch
from pathlib import Path
from torchvision import transforms
import yaml
import numpy as np
from PIL import Image
from typing import Literal

from .custom_dataset import CustomDataset, Betweens

class DriveDataset(CustomDataset):
    mapping = {
        "train": ("training/images/*.png", "training/1st_manual/*.png"),
        "test": ("test/images/*.png", "test/1st_manual/*.png"),
    }

    def __init__(self, base_dir: Path, dataset_type: Literal['train', 'test'], is_rgb: bool = False, **kwargs):
        """
        DRIVE 数据集实现
        
        Args:
            base_dir: 数据集根目录
            dataset_type: 数据集类型 ('train' 或 'test')
            is_rgb: 是否以RGB方式读取图像（默认灰度）
            **kwargs: 预留扩展参数
        """
        super(DriveDataset, self).__init__(base_dir, dataset_type, **kwargs)

        if 'source' in kwargs and 'target' in kwargs:
            image_glob, label_glob = kwargs['source'], kwargs['target']
        else:
            image_glob, label_glob = self.mapping[dataset_type]

        # 默认的图像变换，仅进行张量化
        self.transforms = self._get_transforms()
        self.config = {"is_rgb": is_rgb, "source": image_glob, "target": label_glob}

        images = [p for p in base_dir.glob(image_glob)]
        masks = [p for p in base_dir.glob(label_glob)]

        # 不再进行between切片，直接使用完整数据集
        self.images = images
        self.masks = masks
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
        return image, mask

    @staticmethod
    def to_numpy(save_dir: Path, base_dir: Path, betweens: Betweens, **kwargs):
        """将数据集转换为numpy格式保存（兼容保留betweens参数，但内部不再使用切片）"""
        save_dir = save_dir / DriveDataset.name()
        save_dir.mkdir(parents=True, exist_ok=True)

        train_dataset = DriveDataset.get_train_dataset(base_dir, **kwargs)
        test_dataset = DriveDataset.get_test_dataset(base_dir, **kwargs)

        from torch.utils.data import DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

        for dataloader, data_dir, dataset_type in zip((train_dataloader, test_dataloader), (save_dir, save_dir), ('train', 'test')):
            image_dir = (data_dir / DriveDataset.mapping[dataset_type][0]).parent
            mask_dir = (data_dir / DriveDataset.mapping[dataset_type][1]).parent
            image_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)

            for i, (image, mask) in enumerate(dataloader):
                image_path = data_dir / DriveDataset.mapping[dataset_type][0].replace('*.png', f'{i}.npy')
                mask_path = data_dir / DriveDataset.mapping[dataset_type][1].replace('*.png', f'{i}.npy')

                np.save(image_path, image.numpy())
                np.save(mask_path, mask.numpy())

        # 不再保存betweens配置
        config_file = save_dir / "config.yaml"
        with config_file.open('w', encoding='utf-8') as f:
            yaml.dump({**kwargs}, f, sort_keys=False)

    @staticmethod
    def name():
        return "DRIVE"

    @staticmethod
    def get_train_dataset(base_dir: Path, **kwargs):
        """获取训练集实例"""
        return DriveDataset(base_dir, 'train', **kwargs)

    @staticmethod
    def get_valid_dataset(base_dir: Path, **kwargs):
        """DRIVE无单独验证集，返回None"""
        return None

    @staticmethod
    def get_test_dataset(base_dir: Path, **kwargs):
        """获取测试集实例"""
        return DriveDataset(base_dir, 'test', **kwargs)

    def _get_transforms(self):
        """获取数据集的变换"""
        from utils.transform import get_transforms
        return get_transforms()