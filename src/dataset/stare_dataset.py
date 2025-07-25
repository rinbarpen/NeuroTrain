import torch
from torchvision import transforms
from pathlib import Path
from PIL import Image
import yaml
import numpy as np
from typing import Literal

from .custom_dataset import CustomDataset, Betweens

class StareDataset(CustomDataset):
    mapping = {"train": ("training/images/*.png", "training/1st_labels_ah/*.png"), 
               "test":  ("test/images/*.png", "test/1st_labels_ah/*.png")}
    def __init__(self, base_dir: Path, dataset_type: Literal['train', 'test'], between: tuple[float, float]=(0.0, 1.0), transforms: transforms.Compose|None=None, use_numpy=False, is_rgb=False, **kwargs):
        super(StareDataset, self).__init__(base_dir, dataset_type, between, use_numpy=use_numpy)

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
        save_dir = save_dir / StareDataset.name()
        save_dir.mkdir(parents=True, exist_ok=True)

        train_dataset = StareDataset.get_train_dataset(base_dir, between=betweens['train'])
        test_dataset = StareDataset.get_test_dataset(base_dir, between=betweens['test'])

        from torch.utils.data import DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)
        test_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)

        for dataloader, data_dir, dataset_type in zip((train_dataloader, test_dataloader), (save_dir, save_dir), ('train', 'test')):
            for i, (image, mask) in enumerate(dataloader):
                image_path = data_dir / StareDataset.mapping[dataset_type][0].replace('*.png', f'{i}.npy')
                mask_path = data_dir / StareDataset.mapping[dataset_type][1].replace('*.png', f'{i}.npy')
                image_path.parent.mkdir(parents=True, exist_ok=True)
                mask_path.parent.mkdir(parents=True, exist_ok=True)

                np.save(image_path, image.numpy())
                np.save(mask_path, mask.numpy())

        config_file = save_dir / "config.yaml"
        with config_file.open('w', encoding='utf-8') as f:
            yaml.dump({"betweens": betweens, **kwargs}, f)

    @staticmethod
    def name():
        return "STARE"
    @staticmethod
    def get_train_dataset(base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        return StareDataset(base_dir, 'train', between, use_numpy=use_numpy, **kwargs)
    @staticmethod
    def get_valid_dataset(base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        return None
    @staticmethod
    def get_test_dataset(base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        return StareDataset(base_dir, 'test', between, use_numpy=use_numpy, **kwargs)
