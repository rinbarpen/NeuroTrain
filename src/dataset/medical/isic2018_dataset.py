import torch
from pathlib import Path
from PIL import Image
import numpy as np
import yaml
from typing import Literal
import torchvision.transforms as T

from ..custom_dataset import CustomDataset, Betweens

class ISIC2018Dataset(CustomDataset):
    """ISIC 2018皮肤病变分割数据集
    
    该数据集包含皮肤镜图像及其对应的分割掩码，用于皮肤癌病变的自动分割任务。
    """
    mapping = {"train": ("ISIC2018_Task1-2_Training_Input/*.jpg", "ISIC2018_Task1-2_Training_Input/*.jpg"), 
               "valid": ("ISIC2018_Task1-2_Validation_Input/*.jpg", "ISIC2018_Task1_Validation_GroundTruth/*.jpg"),
               "test":  ("ISIC2018_Task1-2_Test_Input/*.jpg", "ISIC2018_Task1_Test_GroundTruth/*.jpg")}
    
    def __init__(self, root_dir: Path, split: Literal['train', 'test', 'valid'], is_rgb=False, **kwargs):
        """
        初始化ISIC2018数据集
        
        Args:
            root_dir: 数据集根目录
            split: 数据集类型
            is_rgb: 是否以RGB模式加载图像
            **kwargs: 其他配置参数，可包含'source'和'target'自定义路径
        """
        super(ISIC2018Dataset, self).__init__(root_dir, split, **kwargs)

        if 'source' in kwargs.keys() and 'target' in kwargs.keys():
            image_glob, label_glob = kwargs['source'], kwargs['target']
        else:
            image_glob, label_glob = self.mapping[split]

        self.config = {"is_rgb": is_rgb, "split": split}
        self.transforms = self._get_transforms()

        images = [p for p in self.root_dir.glob(image_glob)]
        masks = [p for p in self.root_dir.glob(label_glob)]

        self.images = sorted(images)
        self.masks = sorted(masks)
        self.n = len(self.images)

    def _get_transforms(self):
        """获取ISIC2018数据集的默认变换"""
        return T.ToTensor()

    def __getitem__(self, index):
        """获取指定索引的数据样本"""
        image_path, mask_path = self.images[index], self.masks[index]
        
        if self.config['is_rgb']:
            image, mask = Image.open(image_path).convert('RGB'), Image.open(mask_path).convert('RGB')
        else:
            image, mask = Image.open(image_path).convert('L'), Image.open(mask_path).convert('L')
        
        # 应用变换
        image, mask = self.transforms(image), self.transforms(mask)
        
        return {
            'image': image,  # 图像张量
            'mask': mask,    # 掩码张量
            'metadata': {
                'image_path': str(image_path),
                'mask_path': str(mask_path),
                'task_type': 'segmentation',
                'split': self.dataset_type
            }
        }

    @staticmethod
    def to_numpy(save_dir: Path, root_dir: Path, betweens: Betweens, **kwargs):
        """将数据集转换为numpy格式保存
        
        Args:
            save_dir: 保存目录
            root_dir: 数据集根目录
            betweens: 兼容性参数，实际不再使用
            **kwargs: 其他配置参数
        """
        save_dir = save_dir / ISIC2018Dataset.name()
        save_dir.mkdir(parents=True, exist_ok=True)

        train_dataset = ISIC2018Dataset.get_train_dataset(root_dir, **kwargs)
        valid_dataset = ISIC2018Dataset.get_valid_dataset(root_dir, **kwargs)
        test_dataset  = ISIC2018Dataset.get_test_dataset(root_dir, **kwargs)

        from torch.utils.data import DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)
        valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
        test_dataloader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

        for dataloader, data_dir, dataset_type in zip((train_dataloader, valid_dataloader, test_dataloader), (save_dir, save_dir, save_dir), ('train', 'valid', 'test')):
            for i, (image, mask) in enumerate(dataloader):
                image_path = data_dir / ISIC2018Dataset.mapping[dataset_type][0].replace('*.jpg', f'{i}.npy')
                mask_path = data_dir / ISIC2018Dataset.mapping[dataset_type][1].replace('*.jpg', f'{i}.npy')
                image_path.parent.mkdir(parents=True, exist_ok=True)
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                
                np.save(image_path, image.numpy())
                np.save(mask_path, mask.numpy())

        config_file = save_dir / "config.yaml"
        with config_file.open('w', encoding='utf-8') as f:
            yaml.dump({**kwargs}, f)

    @staticmethod
    def name():
        return "ISIC2018"
    
    @staticmethod
    def get_train_dataset(root_dir: Path, **kwargs):
        """获取训练数据集"""
        return ISIC2018Dataset(root_dir, 'train', **kwargs)
    
    @staticmethod
    def get_valid_dataset(root_dir: Path, **kwargs):
        """获取验证数据集"""
        return ISIC2018Dataset(root_dir, 'valid', **kwargs)
    
    @staticmethod
    def get_test_dataset(root_dir: Path, **kwargs):
        """获取测试数据集"""
        return ISIC2018Dataset(root_dir, 'test', **kwargs)

