import torch
from pathlib import Path
from torchvision import transforms
import yaml
import numpy as np
from PIL import Image
from typing import Literal

from utils.util import save_numpy_data
from utils.dataset.custom_dataset import CustomDataset, Betweens

class DriveDataset(CustomDataset):
    mapping = {"train": ("training/images/*.png", "training/1st_manual/*.png"), 
               "test":  ("test/images/*.png", "test/1st_manual/*.png")}
    def __init__(self, base_dir: Path, dataset_type: Literal['train', 'test'], between: tuple[float, float]=(0.0, 1.0), transforms: transforms.Compose|None=None, use_numpy=False, is_rgb=False, **kwargs):
        super(DriveDataset, self).__init__(base_dir, between, use_numpy=use_numpy)

        if 'source' in kwargs.keys() and 'target' in kwargs.keys():
            image_glob, label_glob = kwargs['source'], kwargs['target']
        else:
            image_glob, label_glob = self.mapping[dataset_type]

        self.transforms = transforms
        self.config = {"is_rgb": is_rgb, "source": image_glob, "target": label_glob}

        if use_numpy:
            image_glob = image_glob.replace('*.png', '*.npy')
            label_glob = label_glob.replace('*.png', '*.npy')

        images = [p for p in base_dir.glob(image_glob)]
        masks = [p for p in base_dir.glob(label_glob)]

        self.n = len(images)
        bw = (int(self.between[0] * self.n), int(self.between[1] * self.n))
        self.images = images[bw[0]:bw[1]]
        self.masks = masks[bw[0]:bw[1]]

    def __getitem__(self, index):
        image, mask = self.images[index], self.masks[index]
        if self.use_numpy:
            return torch.from_numpy(np.load(image)), torch.from_numpy(np.load(mask))
        
        if self.config['is_rgb']:
            image, mask = Image.open(image).convert('RGB'), Image.open(mask).convert('RGB')
        else:
            image, mask = Image.open(image).convert('L'), Image.open(mask).convert('L')
        
        image, mask = self.transforms(image), self.transforms(mask)
        return image, mask

    @staticmethod
    def to_numpy(save_dir: Path, base_dir: Path, betweens: Betweens, **kwargs):
        save_dir = save_dir / DriveDataset.name()
        save_dir.mkdir(parents=True, exist_ok=True)
        train_dir = base_dir / "training"
        test_dir = base_dir / "test"

        train_dataset = DriveDataset.get_train_dataset(train_dir, between=betweens['train'])
        test_dataset = DriveDataset.get_test_dataset(test_dir, between=betweens['test'])

        for dataset, data_dir, dataset_type in zip((train_dataset, test_dataset), (train_dir, test_dir), ('train', 'test')):
            for i, (image, mask) in enumerate(dataset):
                image_dir = data_dir / DriveDataset.mapping[dataset_type][0].replace('*.png', f'{i}.npy')
                mask_dir = data_dir / DriveDataset.mapping[dataset_type][1].replace('*.png', f'{i}.npy')
                save_numpy_data(image_dir, image)
                save_numpy_data(mask_dir, mask)

        config_file = save_dir / "config.yaml"
        with config_file.open('w', encoding='utf-8') as f:
            yaml.dump({"betweens": betweens, **kwargs}, f)

    @staticmethod
    def name():
        return "DRIVE"

    @staticmethod
    def get_train_dataset(base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        return DriveDataset(base_dir, between, 'train', use_numpy=use_numpy, **kwargs)
        return DriveDataset(base_dir, 'train', between=between, use_numpy=use_numpy, **kwargs)
    @staticmethod
    def get_valid_dataset(base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        return None
    @staticmethod
    def get_test_dataset(base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        return DriveDataset(base_dir, 'test', between=between, use_numpy=use_numpy, **kwargs)
