import os
import torch
from torch.utils.data import Dataset
from pathlib import Path

from utils.transform import to_gray, to_rgb, image_transform
from utils.util import load_numpy_data, save_numpy_data
import yaml

def get_isic2018_train_dataset(base_dir: Path, to_rgb=False):
    train_dataset = ISIC2018Dataset(base_dir / "ISIC2018_Task1-2_Training_Input",
                                    base_dir / "ISIC2018_Task1_Training_GroundTruth",
                                    to_rgb)
    return train_dataset

def get_isic2018_valid_dataset(base_dir: Path, to_rgb=False):
    valid_dataset = ISIC2018Dataset(base_dir / "ISIC2018_Task1-2_Validation_Input",
                                    base_dir / "ISIC2018_Task1_Validation_GroundTruth",
                                    to_rgb)
    return valid_dataset

def get_isic2018_test_dataset(base_dir: Path, *, to_rgb=False):
    test_dataset = ISIC2018Dataset(base_dir / "ISIC2018_Task1-2_Test_Input",
                                   base_dir / "ISIC2018_Task1_Test_GroundTruth",
                                   to_rgb)
    return test_dataset

# output / ISIC2018
def convert_to_numpy(save_dir: Path, base_dir: Path, to_rgb=False):
    save_dir = save_dir / "ISIC2018"
    os.makedirs(save_dir.absolute(), exist_ok=True)

    train_dataset = get_isic2018_train_dataset(base_dir, to_rgb=to_rgb)
    image_dir = save_dir / train_dataset.image_dir.stem
    mask_dir = save_dir / train_dataset.mask_dir.stem
    
    for i, (image, mask) in enumerate(train_dataset):
        save_numpy_data(image_dir / f'{i}.npy', image)
        save_numpy_data(mask_dir / f'{i}.npy', mask)

    valid_dataset = get_isic2018_valid_dataset(base_dir, to_rgb=to_rgb)
    image_dir = save_dir / valid_dataset.image_dir.stem
    mask_dir = save_dir / valid_dataset.mask_dir.stem
    
    for i, (image, mask) in enumerate(valid_dataset):
        save_numpy_data(image_dir / f'{i}.npy', image)
        save_numpy_data(mask_dir / f'{i}.npy', mask)

    test_dataset = get_isic2018_test_dataset(base_dir, to_rgb=to_rgb)
    image_dir = save_dir / test_dataset.image_dir.stem
    mask_dir = save_dir / test_dataset.mask_dir.stem
    
    for i, (image, mask) in enumerate(test_dataset):
        save_numpy_data(image_dir / f'{i}.npy', image)
        save_numpy_data(mask_dir / f'{i}.npy', mask)

    config_file = save_dir / "config.yaml"
    with config_file.open('w', encoding='utf-8') as f:
        yaml.dump({"to_rgb": to_rgb}, f)

class ISIC2018Dataset(Dataset):
    def __init__(self, image_dir: Path, mask_dir: Path, to_rgb=False, *, is_numpy=False):
        super(ISIC2018Dataset, self).__init__()

        self.is_numpy = is_numpy
        self.to_rgb = to_rgb
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_paths = [p for p in image_dir.glob('*.jpg')] if not self.is_numpy else [p for p in image_dir.glob('*.npz')]
        self.mask_paths = [p for p in mask_dir.glob('*.jpg')] if not self.is_numpy else [p for p in image_dir.glob('*.npz')]

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
