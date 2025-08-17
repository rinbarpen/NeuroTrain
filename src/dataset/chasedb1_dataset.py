import torch
import yaml
from PIL import Image
import numpy as np
from pathlib import Path
from torchvision import transforms
from typing import Literal

from .custom_dataset import CustomDataset, Betweens

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

    def __init__(self, base_dir: Path, dataset_type: Literal['train', 'test'], is_rgb: bool = False, **kwargs):
        """
        CHASE_DB1 数据集
        
        Args:
            base_dir: 数据集根目录
            dataset_type: 数据集类型 ('train' 或 'test')
            is_rgb: 是否以RGB方式读取图像（默认灰度）
            **kwargs: 预留扩展参数
        """
        super(ChaseDB1Dataset, self).__init__(base_dir, dataset_type, **kwargs)

        if 'source' in kwargs and 'target' in kwargs:
            image_glob, label_glob = kwargs['source'], kwargs['target']
        else:
            image_glob, label_glob = self.mapping[dataset_type]

        # 默认变换：仅张量化，通过 _get_transforms() 提供
        self.transforms = self._get_transforms()
        self.config = {"is_rgb": is_rgb, "source": image_glob, "target": label_glob}

        images = [p for p in base_dir.glob(image_glob)]
        masks = [p for p in base_dir.glob(label_glob)]

        # 不再切片，完整使用
        self.images = images
        self.masks = masks
        self.n = len(self.images)

    def __getitem__(self, index: int):
        """按索引返回图像和掩码张量"""
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
        导出为numpy文件
        - 兼容保留 betweens 形参，但内部不再进行切片
        - 统一保存为 save_dir/<dataset_name>/<split>/... 结构
        """
        save_dir = save_dir / ChaseDB1Dataset.name()
        save_dir.mkdir(parents=True, exist_ok=True)

        train_dataset = ChaseDB1Dataset.get_train_dataset(base_dir, **kwargs)
        test_dataset = ChaseDB1Dataset.get_test_dataset(base_dir, **kwargs)

        from torch.utils.data import DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

        for dataloader, dataset_type in ((train_dataloader, 'train'), (test_dataloader, 'test')):
            image_dir = (save_dir / ChaseDB1Dataset.mapping[dataset_type][0]).parent
            mask_dir = (save_dir / ChaseDB1Dataset.mapping[dataset_type][1]).parent
            image_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)

            for i, (image, mask) in enumerate(dataloader):
                image_path = image_dir / f'{i}.npy'
                mask_path = mask_dir / f'{i}.npy'
                np.save(image_path, image.numpy())
                np.save(mask_path, mask.numpy())

        # 配置文件不再写 betweens
        config_file = save_dir / "config.yaml"
        with config_file.open('w', encoding='utf-8') as f:
            yaml.dump({**kwargs}, f, sort_keys=False)

    @staticmethod
    def name():
        return "CHASEDB1"

    @staticmethod
    def get_train_dataset(base_dir: Path, **kwargs):
        """获取训练集实例"""
        return ChaseDB1Dataset(base_dir, 'train', **kwargs)

    @staticmethod
    def get_valid_dataset(base_dir: Path, **kwargs):
        """CHASE_DB1 无验证集，返回 None"""
        return None

    @staticmethod
    def get_test_dataset(base_dir: Path, **kwargs):
        """获取测试集实例"""
        return ChaseDB1Dataset(base_dir, 'test', **kwargs)
