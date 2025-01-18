import os
import torch
from torch.utils.data import Dataset
from pathlib import Path

import yaml

from utils.transform import to_rgb, to_gray, image_transform
from utils.util import load_numpy_data, save_numpy_data

def get_bowl2018_train_valid_dataset(base_dir: Path, split: float, to_rgb=False):
    base_dir = base_dir / "stage1_train"
    train_dataset = Bowl2018Dataset(base_dir, (0.0, split), to_rgb)
    valid_dataset = Bowl2018Dataset(base_dir, (split, 1.0), to_rgb)
    return train_dataset, valid_dataset

def get_bowl2018_test_dataset(base_dir: Path, to_rgb=False):
    base_dir = base_dir / "stage1_test"
    test_dataset = Bowl2018Dataset(base_dir, (0.0, 1.0), to_rgb)
    return test_dataset

def convert_to_numpy(save_dir: Path, base_dir: Path, to_rgb=False):
    save_dir = save_dir / "BOWL2018"
    os.makedirs(save_dir.absolute(), exist_ok=True)

    train_dataset, _ = get_bowl2018_train_valid_dataset(base_dir, 1.0, to_rgb=to_rgb)
    
    for i, (image, masks) in enumerate(train_dataset):
        image_dir = save_dir / i / "images"
        mask_dir = save_dir / i / "masks"
        save_numpy_data(image_dir / f'{i}.npy', image)
        for j, mask in enumerate(masks):
            save_numpy_data(mask_dir / f'{i}_{j}.npy', mask)

    test_dataset = get_bowl2018_test_dataset(base_dir, to_rgb=to_rgb)
    
    for i, (image, masks) in enumerate(test_dataset):
        image_dir = save_dir / i / "images"
        mask_dir = save_dir / i / "masks"
        save_numpy_data(image_dir / f'{i}.npy', image)
        for j, mask in enumerate(masks):
            save_numpy_data(mask_dir / f'{i}_{j}.npy', mask)

    config_file = save_dir / "config.yaml"
    with config_file.open('w', encoding='utf-8') as f:
        yaml.dump({"to_rgb": to_rgb}, f)


class Bowl2018Dataset(Dataset):
    def __init__(self, base_dir: Path, between: tuple[float, float], to_rgb=False, *, is_numpy=False):
        super(Bowl2018Dataset, self).__init__()

        self.is_numpy = is_numpy
        self.to_rgb = to_rgb
        self.base_dir = base_dir
        paths = [p for p in base_dir.iterdir() if p.is_dir()]
        self.between = (len(paths) * between[0], len(paths) * between[1])
        self.paths = paths[self.between[0]:self.between[1]]

    def __getitem__(self, index):
        path = self.paths[index]
        image_dir, mask_dir = path / "images", path / "masks"
        if self.is_numpy:
            images = [torch.from_numpy(load_numpy_data(p)) for p in image_dir.iterdir()][0] 
            masks = [torch.from_numpy(load_numpy_data(p)) for p in mask_dir.iterdir()]
            return images, masks

        if self.to_rgb:
            images = [to_rgb(image_file) for image_file in image_dir.iterdir()][0]
            masks = [to_rgb(mask_file) for mask_file in mask_dir.iterdir()]
        else:
            images = [to_gray(image_file) for image_file in image_dir.iterdir()][0]
            masks = [to_gray(mask_file) for mask_file in mask_dir.iterdir()]

        images = image_transform(images, size=(512, 512))
        masks = [image_transform(mask, size=(512, 512)) for mask in masks]
        return images, masks

    def __len__(self):
        return len(self.paths)
