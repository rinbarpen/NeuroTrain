import torch
import yaml
from PIL import Image
import numpy as np
from pathlib import Path
from torchvision import transforms
from typing import Literal

from utils.dataset.custom_dataset import CustomDataset, Betweens
from utils.util import save_numpy_data

class ChaseDB1Dataset(CustomDataset):
    mapping = {"train": ("training/images/*.png", "training/1st_label/*.png"),
               "test":  ("test/images/*.png", "test/1st_label/*.png")}
    def __init__(self, base_dir: Path, dataset_type: Literal['train', 'test', 'valid'], between: tuple[float, float]=(0.0, 1.0), transforms: transforms.Compose|None=None, use_numpy=False, is_rgb=False, **kwargs):
        super(ChaseDB1Dataset, self).__init__(base_dir, dataset_type, between, use_numpy=use_numpy)

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
        save_dir = save_dir / ChaseDB1Dataset.name()
        save_dir.mkdir(parents=True, exist_ok=True)

        image_dir = save_dir / "images"
        mask_dir = save_dir / "1st_label"

        train_dataset = ChaseDB1Dataset.get_train_dataset(base_dir / "training", between=betweens['train'], **kwargs)
        test_dataset  = ChaseDB1Dataset.get_test_dataset(base_dir / "test", between=betweens['test'], **kwargs)

        for i, (image, mask) in enumerate(train_dataset):
            save_numpy_data(image_dir / f'{i}.npy', image)
            save_numpy_data(mask_dir / f'{i}.npy', mask)
        for i, (image, mask) in enumerate(test_dataset):
            save_numpy_data(image_dir / f'{i}.npy', image)
            save_numpy_data(mask_dir / f'{i}.npy', mask)

        config_file = save_dir / "config.yaml"
        with config_file.open('w', encoding='utf-8') as f:
            yaml.dump({"betweens": betweens, **kwargs}, f)

    @staticmethod
    def name():
        return "CHASEDB1"
    @staticmethod
    def get_train_dataset(base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        return ChaseDB1Dataset(base_dir, 'train', between, use_numpy=use_numpy, **kwargs)
    @staticmethod
    def get_valid_dataset(base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        return None
    @staticmethod
    def get_test_dataset(base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        return ChaseDB1Dataset(base_dir, 'test', between, use_numpy=use_numpy, **kwargs)
