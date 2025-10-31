import torch
from pathlib import Path
import yaml
from PIL import Image
import numpy as np
from torchvision import transforms
from typing import Literal

from ..custom_dataset import CustomDataset, Betweens

# Instance Segmentation Dataset
class BOWL2018Dataset(CustomDataset):
    mapping = {
        'train': ('*/images/*.png', '*/masks/*.png'),
        'valid': ('*/images/*.png', '*/masks/*.png'),
        'test':  ('*/images/*.png', '*/masks/*.png'),
    }

    def _get_transforms(self):
        """
        获取预处理/增强变换；默认仅 ToTensor。
        说明：
            - transforms 在具体数据集类内实现，避免在 CustomDataset 中耦合。
        """
        return transforms.ToTensor()

    def __init__(self, root_dir: Path, split: Literal['train', 'valid', 'test'], is_rgb: bool = False, **kwargs):
        """
        BOWL2018 实例分割数据集
        
        Args:
            root_dir: 数据集根目录
            split: 'train' | 'valid' | 'test'
            is_rgb: 是否以RGB方式读取图像（默认灰度）
            **kwargs: 预留参数，支持传入 source/target 和 n_instance
        """
        super(BOWL2018Dataset, self).__init__(root_dir, split, **kwargs)

        if 'source' in kwargs and 'target' in kwargs:
            image_glob, label_glob = kwargs['source'], kwargs['target']
        else:
            image_glob, label_glob = self.mapping[split]

        # 默认变换：仅张量化，通过 _get_transforms() 提供
        self.transforms = self._get_transforms()
        self.config = {"is_rgb": is_rgb, "source": image_glob, "target": label_glob}
        if "n_instance" in kwargs:
            self.config["n_instance"] = kwargs["n_instance"]
        else:
            from src.config import get_config
            c = get_config()
            self.config["n_instance"] = len(c["classes"])  # 类别数即实例掩码数量

        images = [p for p in root_dir.glob(image_glob)]
        masks = [p for p in root_dir.glob(label_glob)]
        # 将掩码按每个样本的实例数进行分组
        masks = [masks[i:i + self.config["n_instance"]] for i in range(0, len(masks), self.config["n_instance"])]

        # 不再切片，完整使用
        self.images = sorted(images)
        self.masks = sorted(masks)
        self.n = len(self.images)

    def __getitem__(self, index: int):
        """返回图像张量和拼接后的多实例掩码张量"""
        image_path, masks_paths = self.images[index], self.masks[index]

        if self.config['is_rgb']:
            image = Image.open(image_path).convert('RGB')
            masks = [Image.open(masks_paths[j]).convert('RGB') for j in range(self.config['n_instance'])]
        else:
            image = Image.open(image_path).convert('L')
            masks = [Image.open(masks_paths[j]).convert('L') for j in range(self.config['n_instance'])]

        image = self.transforms(image)
        masks = [self.transforms(mask) for mask in masks]
        masks = torch.concat(masks, dim=1)
        return {
            'image': image,  # 图像张量
            'mask': masks,   # 掩码张量
            'metadata': {
                'image_path': str(image_path),
                'mask_paths': [str(p) for p in masks_paths],
                'task_type': 'instance_segmentation',
                'split': self.dataset_type
            }
        }

    @staticmethod
    def to_numpy(save_dir: Path, root_dir: Path, betweens: Betweens, **kwargs):
        """
        导出为numpy文件
        - 兼容保留 betweens 形参，但内部不再进行切片
        - 保持原有保存结构：save_dir/<dataset>/<index>/{images,masks}
        """
        save_dir = save_dir / BOWL2018Dataset.name()
        save_dir.mkdir(parents=True, exist_ok=True)

        train_dataset = BOWL2018Dataset.get_train_dataset(root_dir, **kwargs)
        valid_dataset = BOWL2018Dataset.get_valid_dataset(root_dir, **kwargs)
        test_dataset = BOWL2018Dataset.get_test_dataset(root_dir, **kwargs)

        from torch.utils.data import DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)
        valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

        for dataloader in (train_dataloader, valid_dataloader, test_dataloader):
            for i, (image, masks) in enumerate(dataloader):
                image_dir, mask_dir = save_dir / f'{i}' / "images", save_dir / f'{i}' / "masks"
                image_dir.mkdir(parents=True, exist_ok=True)
                mask_dir.mkdir(parents=True, exist_ok=True)
                image_path = image_dir / f'{i}.npy'
                np.save(image_path, image.numpy())
                for j, mask in enumerate(masks):
                    mask_path = mask_dir / f'{i}_{j}.npy'
                    np.save(mask_path, mask.numpy())

        # 配置文件不再写 betweens
        if "n_instance" in kwargs:
            n_instance = kwargs["n_instance"]
        else:
            from src.config import get_config
            c = get_config()
            n_instance = len(c["classes"])  # 冗余写入，便于读取

        config_file = save_dir / "config.yaml"
        with config_file.open('w', encoding='utf-8') as f:
            yaml.dump({"n_instance": n_instance, **kwargs}, f, sort_keys=False)

    @staticmethod
    def name():
        return "BOWL2018"
    
    @staticmethod
    def metadata(**kwargs):
        """获取BOWL2018数据集元数据"""
        return {
            'num_classes': 2,
            'class_names': ['background', 'nucleus'],
            'task_type': 'segmentation',
            'metrics': ['dice', 'iou', 'accuracy'],
            'num_train': 670,
            'num_test': 65,
            'dataset_name': 'BOWL2018',
            'description': 'Data Science Bowl 2018 Nucleus Segmentation'
        }

    @staticmethod
    def get_train_dataset(root_dir: Path, **kwargs):
        """获取训练集实例"""
        return BOWL2018Dataset(root_dir, 'train', **kwargs)

    @staticmethod
    def get_valid_dataset(root_dir: Path, **kwargs):
        """获取验证集实例"""
        return BOWL2018Dataset(root_dir, 'valid', **kwargs)

    @staticmethod
    def get_test_dataset(root_dir: Path, **kwargs): 
        """获取测试集实例"""
        return BOWL2018Dataset(root_dir, 'test', **kwargs)
