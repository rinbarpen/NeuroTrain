import os
import torch
from torch.utils.data import Dataset
from pathlib import Path

import yaml
from utils.transform import image_transform, to_gray, to_rgb
from utils.util import load_numpy_data, save_numpy_data

def get_drive_train_valid_dataset(base_dir: Path, split: float, to_rgb=False):
    base_dir = base_dir / "training"
    train_dataset = DriveDataset(base_dir, (0.0, split), to_rgb)
    valid_dataset = DriveDataset(base_dir, (split, 1.0), to_rgb)
    return train_dataset, valid_dataset

def get_drive_test_dataset(base_dir: Path, to_rgb=False):
    base_dir = base_dir / "test"
    test_dataset = DriveDataset(base_dir, (0.0, 1.0), to_rgb)
    return test_dataset

def convert_to_numpy(save_dir: Path, base_dir: Path, to_rgb=False):
    save_dir = save_dir / "DRIVE"
    os.makedirs(save_dir.absolute(), exist_ok=True)

    train_dataset, _ = get_drive_train_valid_dataset(base_dir, 1.0, to_rgb=to_rgb)
    image_dir = save_dir / "images"
    mask_dir = save_dir / "1st_manual"
    
    for i, (image, mask) in enumerate(train_dataset):
        save_numpy_data(image_dir / f'{i}.npy', image)
        save_numpy_data(mask_dir / f'{i}.npy', mask)

    test_dataset = get_drive_test_dataset(base_dir, to_rgb=to_rgb)
    image_dir = save_dir / "images"
    mask_dir = save_dir / "1st_manual"
    
    for i, (image, mask) in enumerate(test_dataset):
        save_numpy_data(image_dir / f'{i}.npy', image)
        save_numpy_data(mask_dir / f'{i}.npy', mask)

    config_file = save_dir / "config.yaml"
    with config_file.open('w', encoding='utf-8') as f:
        yaml.dump({"to_rgb": to_rgb}, f)


class DriveDataset(Dataset):
    def __init__(self, base_dir: Path, between: tuple[float, float], to_rgb=False, *, is_numpy=False):
        super(DriveDataset, self).__init__()

        self.is_numpy = is_numpy
        self.to_rgb = to_rgb
        self.base_dir = base_dir
        image_paths = [p for p in (base_dir / "images").iterdir()] if not is_numpy else [p for p in (base_dir / "images").iterdir()]
        mask_paths = [p for p in (base_dir / "1st_manual").iterdir()] if not is_numpy else [p for p in (base_dir / "1st_manual").iterdir()]
        self.between = (int(len(image_paths) * between[0]), int(len(image_paths) * between[1]))
        self.image_paths = image_paths[self.between[0]:self.between[1]] 
        self.mask_paths = mask_paths[self.between[0]:self.between[1]]

    def __getitem__(self, index):
        image_file, mask_file = self.image_paths[index], self.mask_paths[index]
        if self.is_numpy:
            return torch.from_numpy(load_numpy_data(image_file)), torch.from_numpy(load_numpy_data(mask_file))

        if self.to_rgb:
            image, mask = to_rgb(image_file), to_rgb(mask_file)
        else:
            image, mask = to_gray(image_file), to_gray(mask_file)

        image, mask = image_transform(image, (512, 512), self.to_rgb), image_transform(mask, (512, 512), self.to_rgb)

        return image, mask

    def __len__(self):
        return len(self.image_paths)
