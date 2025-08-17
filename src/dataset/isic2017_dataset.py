import torch
from pathlib import Path
from PIL import Image
import numpy as np
import yaml
from typing import Literal
from torchvision import transforms

from .custom_dataset import CustomDataset, Betweens

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
        from utils.transform import get_transforms
        return get_transforms()

    def __init__(self, base_dir: Path, dataset_type: Literal['train', 'test', 'valid'], is_rgb: bool = False, **kwargs):
        """
        ISIC 2017 皮肤病变分割数据集
        
        Args:
            base_dir: 数据集根目录
            dataset_type: 'train' | 'valid' | 'test'
            is_rgb: 是否以RGB方式读取图像（默认灰度）
            **kwargs: 预留扩展参数，支持自定义source/target
        """
        super(ISIC2017Dataset, self).__init__(base_dir, dataset_type, **kwargs)

        if 'source' in kwargs and 'target' in kwargs:
            image_glob, label_glob = kwargs['source'], kwargs['target']
        else:
            image_glob, label_glob = self.mapping[dataset_type]

        # 默认仅做张量化转换（通过 _get_transforms 提供，便于后续在本类内扩展）
        self.transforms = self._get_transforms()
        self.config = {"is_rgb": is_rgb, "source": image_glob, "target": label_glob}

        images = [p for p in base_dir.glob(image_glob)]
        masks = [p for p in base_dir.glob(label_glob)]

        # 不再进行between切片，直接使用完整数据集
        self.images = images
        self.masks = masks
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
        return image, mask

    @staticmethod
    def to_numpy(save_dir: Path, base_dir: Path, betweens: Betweens, **kwargs):
        """
        导出数据为numpy格式
        - 兼容保留 betweens 形参，但内部不再使用切片
        - 目录结构基于mapping推导，统一保存到save_dir/ISIC2017/<split>/... 下
        """
        save_dir = save_dir / ISIC2017Dataset.name()
        save_dir.mkdir(parents=True, exist_ok=True)

        train_dataset = ISIC2017Dataset.get_train_dataset(base_dir, **kwargs)
        valid_dataset = ISIC2017Dataset.get_valid_dataset(base_dir, **kwargs)
        test_dataset = ISIC2017Dataset.get_test_dataset(base_dir, **kwargs)

        from torch.utils.data import DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)
        valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

        for dataloader, dataset_type in (
            (train_dataloader, 'train'),
            (valid_dataloader, 'valid'),
            (test_dataloader, 'test'),
        ):
            image_dir = (save_dir / ISIC2017Dataset.mapping[dataset_type][0]).parent
            mask_dir = (save_dir / ISIC2017Dataset.mapping[dataset_type][1]).parent
            image_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)

            for i, (image, mask) in enumerate(dataloader):
                image_path = image_dir / f'{i}.npy'
                mask_path = mask_dir / f'{i}.npy'
                np.save(image_path, image.numpy())
                np.save(mask_path, mask.numpy())

        config_file = save_dir / "config.yaml"
        with config_file.open('w', encoding='utf-8') as f:
            yaml.dump({**kwargs}, f, sort_keys=False)

    @staticmethod
    def name():
        return "ISIC2017"

    @staticmethod
    def get_train_dataset(base_dir: Path, **kwargs):
        """获取训练集实例"""
        return ISIC2017Dataset(base_dir, 'train', **kwargs)

    @staticmethod
    def get_valid_dataset(base_dir: Path, **kwargs):
        """获取验证集实例"""
        return ISIC2017Dataset(base_dir, 'valid', **kwargs)

    @staticmethod
    def get_test_dataset(base_dir: Path, **kwargs):
        """获取测试集实例"""
        return ISIC2017Dataset(base_dir, 'test', **kwargs)
