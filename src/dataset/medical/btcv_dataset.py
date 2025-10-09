import torch
from pathlib import Path
import yaml
from PIL import Image
import numpy as np
from torchvision import transforms
from typing import Literal
import logging

import nibabel as nib

from ..nnunet_dataset import NNUNetDataset

class BTCVDataset(NNUNetDataset):
    def __init__(self, root_dir: Path, split: Literal['train', 'valid', 'test'], transforms: transforms.Compose|None=transforms.Compose([transforms.Grayscale()]), use_numpy=False, **kwargs):
        # 调用父类NNUNetDataset的构造函数，保持原始参数名
        super(BTCVDataset, self).__init__(root_dir, split, use_numpy=use_numpy, config_filename='dataset_0.json', **kwargs)

        # TODO: complete numpy version
        if use_numpy:
            logging.warning('use_numpy is not supported yet!')

        self.transforms = transforms
        self.n = len(self.config)

    def __getitem__(self, index):
        image_file, label_file = self.root_dir / self.config[index]["image"], self.root_dir / self.config[index]["label"]

        # '.nii.gz'
        image = nib.load(image_file) # (H, W, D)
        mask = nib.load(label_file) # (H, W, D)
        image, mask = image.get_fdata(), mask.get_fdata()

        if self.transforms:
            # (H, W, D) -> (D, H, W)
            image = image.transpose(2, 0, 1)
            mask = mask.transpose(2, 0, 1)
            image_tensor = self.transforms(image)
            mask_tensor = self.transforms(mask)
            image_tensor = image_tensor.transpose(1, 2, 0)
            mask_tensor = mask_tensor.transpose(1, 2, 0)
        else:
            image_tensor = torch.Tensor(image)
            mask_tensor = torch.Tensor(mask)
        
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'metadata': {
                'image_path': str(image_file),
                'mask_path': str(label_file),
                'task_type': 'segmentation',
                'split': self.split
            }
        }

    @staticmethod
    def get_train_dataset(root_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        return BTCVDataset(root_dir, 'train', **kwargs)
    @staticmethod
    def get_valid_dataset(root_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        return BTCVDataset(root_dir, 'valid', **kwargs)
    @staticmethod
    def get_test_dataset(root_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        return BTCVDataset(root_dir, 'test', **kwargs)

    @staticmethod
    def name():
        return "BTCV"
