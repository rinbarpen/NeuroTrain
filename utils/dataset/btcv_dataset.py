import torch
from pathlib import Path
import yaml
from PIL import Image
import numpy as np
from torchvision import transforms
from typing import Literal
import logging

import nibabel as nib

from utils.dataset.nnunet_dataset import NNUNetDataset

class BTCVDataset(NNUNetDataset):
    def __init__(self, base_dir: Path, dataset_type: Literal['train', 'valid', 'test'], transforms: transforms.Compose|None=transforms.Compose(transforms.Grayscale()), use_numpy=False, **kwargs):
        super(BTCVDataset, self).__init__(base_dir=base_dir, dataset_type=dataset_type, use_numpy=use_numpy, config_filename='dataset_0.json')

        # TODO: complete numpy version
        if use_numpy:
            logging.warning('use_numpy is not supported yet!')

        self.transforms = transforms
        self.n = len(self.config)

    def __getitem__(self, index):
        image_file, label_file = self.base_dir / self.config[index]["image"], self.base_dir / self.config[index]["label"]

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
        
        return image_tensor, mask_tensor

    @staticmethod
    def get_train_dataset(base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        return BTCVDataset(base_dir, 'train', **kwargs)
    @staticmethod
    def get_valid_dataset(base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        return BTCVDataset(base_dir, 'valid', **kwargs)
    @staticmethod
    def get_test_dataset(base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        return BTCVDataset(base_dir, 'test', **kwargs)

    @staticmethod
    def name():
        return "BTCV"
