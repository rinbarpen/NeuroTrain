import torch
from pathlib import Path
from torchvision import transforms
import yaml
import numpy as np
from PIL import Image
from typing import Literal

from .custom_dataset import CustomDataset, Betweens

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

        bw = self.get_slice(len(images))
        self.images = images[bw]
        self.masks = masks[bw]
        self.n = len(self.images)

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

        train_dataset = DriveDataset.get_train_dataset(base_dir, between=betweens['train'], **kwargs)
        test_dataset = DriveDataset.get_test_dataset(base_dir, between=betweens['test'], **kwargs)

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

        config_file = save_dir / "config.yaml"
        with config_file.open('w', encoding='utf-8') as f:
            yaml.dump({"betweens": betweens, **kwargs}, f, sort_keys=False)

    @staticmethod
    def name():
        return "DRIVE"
    @staticmethod
    def get_train_dataset(base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        return DriveDataset(base_dir, 'train', between=between, use_numpy=use_numpy, **kwargs)
    @staticmethod
    def get_valid_dataset(base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        return None
    @staticmethod
    def get_test_dataset(base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        return DriveDataset(base_dir, 'test', between=between, use_numpy=use_numpy, **kwargs)
