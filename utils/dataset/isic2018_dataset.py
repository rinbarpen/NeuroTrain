import torch
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms
import yaml
from typing import Literal

from utils.dataset.custom_dataset import CustomDataset, Betweens

class ISIC2018Dataset(CustomDataset):
    mapping = {"train": ("ISIC2018_Task1-2_Training_Input/*.jpg", "ISIC2018_Task1-2_Training_Input/*.jpg"), 
               "valid": ("ISIC2018_Task1-2_Validation_Input/*.jpg", "ISIC2018_Task1_Validation_GroundTruth/*.jpg"),
               "test":  ("ISIC2018_Task1-2_Test_Input/*.jpg", "ISIC2018_Task1_Test_GroundTruth/*.jpg")}
    def __init__(self, base_dir: Path, dataset_type: Literal['train', 'test', 'valid'], between: tuple[float, float]=(0.0, 1.0), transforms: transforms.Compose|None=None, use_numpy=False, is_rgb=False, **kwargs):
        super(ISIC2018Dataset, self).__init__(base_dir, dataset_type, between, use_numpy=use_numpy)

        if 'source' in kwargs.keys() and 'target' in kwargs.keys():
            image_glob, label_glob = kwargs['source'], kwargs['target']
        else:
            image_glob, label_glob = self.mapping[dataset_type]

        self.transforms = transforms
        self.config = {"is_rgb": is_rgb, "dataset_type": dataset_type}

        if use_numpy:
            image_glob = image_glob.replace('*.jpg', '*.npy')
            label_glob = label_glob.replace('*.jpg', '*.npy')

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
        save_dir = save_dir / ISIC2018Dataset.name()
        save_dir.mkdir(parents=True, exist_ok=True)

        train_dataset = ISIC2018Dataset.get_train_dataset(base_dir, between=betweens['train'], **kwargs)
        valid_dataset = ISIC2018Dataset.get_valid_dataset(base_dir, between=betweens['valid'], **kwargs)
        test_dataset  = ISIC2018Dataset.get_test_dataset(base_dir, between=betweens['test'], **kwargs)

        for dataset, dataset_type in zip((train_dataset, valid_dataset, test_dataset), ('train', 'valid', 'test')):
            for i, (image, mask) in enumerate(dataset):
                image_path = base_dir / ISIC2018Dataset.mapping[dataset_type][0].replace('*.jpg', f'{i}.npy')
                mask_path = base_dir / ISIC2018Dataset.mapping[dataset_type][1].replace('*.jpg', f'{i}.npy')
                image_path.parent.mkdir(parents=True, exist_ok=True)
                mask_path.parent.mkdir(parents=True, exist_ok=True)

                np.save(image_path, image)
                np.save(mask_path, mask)

        config_file = save_dir / "config.yaml"
        with config_file.open('w', encoding='utf-8') as f:
            yaml.dump({"betweens": betweens, **kwargs}, f)

    @staticmethod
    def name():
        return "ISIC2018"
    @staticmethod
    def get_train_dataset(base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        return ISIC2018Dataset(base_dir, 'train', between, use_numpy=use_numpy, **kwargs)
    @staticmethod
    def get_valid_dataset(base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        return ISIC2018Dataset(base_dir, 'valid', between, use_numpy=use_numpy, **kwargs)
    @staticmethod
    def get_test_dataset(base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        return ISIC2018Dataset(base_dir, 'test', between, use_numpy=use_numpy, **kwargs)

