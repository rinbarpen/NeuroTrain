import torch
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms
import yaml
from typing import Literal

from utils.dataset.custom_dataset import CustomDataset
from utils.util import save_numpy_data

class ISIC2017Dataset(CustomDataset):
    def __init__(self, base_dir: Path, between: tuple[float, float]=(0.0, 1.0), transforms: transforms.Compose|None=None, use_numpy=False, is_rgb=False, *, dataset_type: Literal['train', 'test', 'valid']):
        super(ISIC2017Dataset, self).__init__(base_dir, between, use_numpy=use_numpy)

        self.transforms = transforms
        self.config = {"is_rgb": is_rgb, "dataset_type": dataset_type}

        self.mapping = {"train": ("2017_Training_Data", "2017_Training_Part1_GroundTruth"), 
                        "valid": ("ISIC-2017_Validation_Data", "ISIC-2017_Validation_Part1_GroundTruth"),
                        "test":  ("ISIC-2017_Test_v2_Data", "ISIC-2017_Test_v2_Part1_GroundTruth")}
        image, label = self.mapping[dataset_type]
        images = [p for p in (base_dir / image).glob('*.jpg')] if not use_numpy else [p for p in (base_dir / image).glob('*.npy')]
        masks = [p for p in (base_dir / label).glob('*.jpg')] if not use_numpy else [p for p in (base_dir / label).glob('*.npy')]

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
    def to_numpy(save_dir: Path, base_dir: Path, betweens: dict[str, tuple[float, float]], **kwargs):
        save_dir = save_dir / ISIC2017Dataset.name()
        save_dir.mkdir(parents=True, exist_ok=True)

        train_dataset = ISIC2017Dataset.get_train_dataset(base_dir, between=betweens['train'], **kwargs)
        valid_dataset = ISIC2017Dataset.get_valid_dataset(base_dir, between=betweens['valid'], **kwargs)
        test_dataset  = ISIC2017Dataset.get_test_dataset(base_dir, between=betweens['test'], **kwargs)

        mapping = {"train": ("2017_Training_Data", "2017_Training_Part1_GroundTruth"), 
                   "valid": ("ISIC-2017_Validation_Data", "ISIC-2017_Validation_Part1_GroundTruth"),
                   "test":  ("ISIC-2017_Test_v2_Data", "ISIC-2017_Test_v2_Part1_GroundTruth")}
        for i, (image, mask) in enumerate(train_dataset):
            image_dir, mask_dir = mapping['train']
            save_numpy_data(image_dir / f'{i}.npy', image)
            save_numpy_data(mask_dir / f'{i}.npy', mask)
        for i, (image, mask) in enumerate(valid_dataset):
            image_dir, mask_dir = mapping['valid']
            save_numpy_data(image_dir / f'{i}.npy', image)
            save_numpy_data(mask_dir / f'{i}.npy', mask)
        for i, (image, mask) in enumerate(test_dataset):
            image_dir, mask_dir = mapping['test']
            save_numpy_data(image_dir / f'{i}.npy', image)
            save_numpy_data(mask_dir / f'{i}.npy', mask)

        config_file = save_dir / "config.yaml"
        with config_file.open('w', encoding='utf-8') as f:
            yaml.dump({"betweens": betweens, **kwargs}, f)

    @staticmethod
    def name():
        return "ISIC2017"
