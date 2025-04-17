import torch
from pathlib import Path
import yaml
from PIL import Image
import numpy as np
from torchvision import transforms
from typing import Literal

from utils.dataset.custom_dataset import CustomDataset, Betweens
from utils.util import save_numpy_data

# Instance Segmentation Dataset
class BOWL2018Dataset(CustomDataset):
    mapping = {'train': ('*/images/*.png', '*/masks/*.png'), 
               'valid': ('*/images/*.png', '*/masks/*.png'),
               'test':  ('*/images/*.png', '*/masks/*.png')}
    def __init__(self, base_dir: Path, dataset_type: Literal['train', 'valid', 'test'], between: tuple[float, float]=(0.0, 1.0), transforms: transforms.Compose|None=None, use_numpy=False, is_rgb=False, **kwargs):
        super(BOWL2018Dataset, self).__init__(base_dir, dataset_type, between, use_numpy=use_numpy)

        if 'source' in kwargs.keys() and 'target' in kwargs.keys():
            image_glob, label_glob = kwargs['source'], kwargs['target']
        else:
            image_glob, label_glob = self.mapping[dataset_type]

        self.transforms = transforms
        self.config = {"is_rgb": is_rgb, "source": image_glob, "target": label_glob}
        if kwargs.__contains__("n_instance"):
            self.config["n_instance"] = kwargs["n_instance"]
        else:
            from config import get_config
            c = get_config()
            self.config["n_instance"] = len(c["classes"])

        if use_numpy:
            image_glob = image_glob.replace('*.png', '*.npy')
            label_glob = label_glob.replace('*.png', '*.npy')

        images = [p for p in base_dir.glob(image_glob)]
        masks = [p for p in base_dir.glob(label_glob)]
        masks = [masks[i:i+self.config["n_instance"]] for i in range(0, len(masks), self.config["n_instance"])]

        self.n = len(images)
        bw = (int(self.between[0] * self.n), int(self.between[1] * self.n))
        self.images, self.masks = images[bw[0]:bw[1]], masks[bw[0]:bw[1]]

    def __getitem__(self, index):
        image, masks = self.images[index], self.masks[index]
        if self.use_numpy:
            image = torch.from_numpy(np.load(image))
            masks = [torch.from_numpy(np.load(masks[j])) for j in range(self.config['n_instance'])]
            masks = torch.concat(masks, dim=1)
            return image, masks

        if self.config['is_rgb']:
            image = Image.open(image).convert('RGB')
            masks = [Image.open(masks[j]).convert('RGB') for j in range(self.config['n_instance'])]
        else:
            image = Image.open(image).convert('L')
            masks = [Image.open(masks[j]).convert('L') for j in range(self.config['n_instance'])]

        image = self.transforms(image)
        masks = [self.transforms(mask) for mask in masks]
        masks = torch.concat(masks, dim=1)
        return image, masks

    @staticmethod
    def to_numpy(save_dir: Path, base_dir: Path, betweens: Betweens, **kwargs):
        save_dir = save_dir / BOWL2018Dataset.name()
        save_dir.mkdir(parents=True, exist_ok=True)

        train_dataset = BOWL2018Dataset.get_train_dataset(base_dir, between=betweens['train'], **kwargs)
        valid_dataset = BOWL2018Dataset.get_valid_dataset(base_dir, between=betweens['valid'], **kwargs)
        test_dataset  = BOWL2018Dataset.get_test_dataset(base_dir, between=betweens['test'], **kwargs)

        for i, (image, masks) in enumerate(train_dataset):
            image_dir, mask_dir = save_dir / f'{i}' / "images", save_dir / f'{i}' / "masks"
            save_numpy_data(image_dir / f'{i}.npy', image)
            for j, mask in enumerate(masks):
                save_numpy_data(mask_dir / f'{i}_{j}.npy', mask)
        for i, (image, masks) in enumerate(valid_dataset):
            image_dir, mask_dir = save_dir / f'{i}' / "images", save_dir / f'{i}' / "masks"
            save_numpy_data(image_dir / f'{i}.npy', image)
            for j, mask in enumerate(masks):
                save_numpy_data(mask_dir / f'{i}_{j}.npy', mask)
        for i, (image, masks) in enumerate(test_dataset):
            image_dir, mask_dir = save_dir / f'{i}' / "images", save_dir / f'{i}' / "masks"
            save_numpy_data(image_dir / f'{i}.npy', image)
            for j, mask in enumerate(masks):
                save_numpy_data(mask_dir / f'{i}_{j}.npy', mask)

        if kwargs.__contains__("n_instance"):
            n_instance = kwargs["n_instance"]
        else:
            from config import get_config
            c = get_config()
            n_instance = len(c["classes"])

        config_file = save_dir / "config.yaml"
        with config_file.open('w', encoding='utf-8') as f:
            yaml.dump({"betweens": betweens, "n_instance": n_instance, **kwargs}, f)

    @staticmethod
    def name():
        return "BOWL2018"
    @staticmethod
    def get_train_dataset(base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        return BOWL2018Dataset(base_dir, 'train', between, use_numpy=use_numpy, **kwargs)
    @staticmethod
    def get_valid_dataset(base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        return BOWL2018Dataset(base_dir, 'valid', between, use_numpy=use_numpy, **kwargs)
    @staticmethod
    def get_test_dataset(base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        return BOWL2018Dataset(base_dir, 'test', between, use_numpy=use_numpy, **kwargs)
