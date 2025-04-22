import logging
from pathlib import Path

from utils.dataset.custom_dataset import Betweens
from utils.dataset import drive_dataset, bowl2018_dataset, chasedb1_dataset, isic2017_dataset, isic2018_dataset, stare_dataset
from utils.transform import get_transforms

def get_train_dataset(dataset_name: str, base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
    transforms = get_transforms()

    match dataset_name.lower():
        case 'drive':
            return drive_dataset.DriveDataset.get_train_dataset(base_dir, between, use_numpy, transforms=transforms, **kwargs)
        case 'stare':
            return stare_dataset.StareDataset.get_train_dataset(base_dir, between, use_numpy, transforms=transforms, **kwargs)
        case 'isic2017':
            return isic2017_dataset.ISIC2017Dataset.get_train_dataset(base_dir, between, use_numpy, transforms=transforms, **kwargs)
        case 'isic2018':
            return isic2018_dataset.ISIC2018Dataset.get_train_dataset(base_dir, between, use_numpy, transforms=transforms, **kwargs)
        case 'bowl2018':
            return bowl2018_dataset.BOWL2018Dataset.get_train_dataset(base_dir, between, use_numpy, transforms=transforms, **kwargs)
        case 'chasedb1':
            return chasedb1_dataset.ChaseDB1Dataset.get_train_dataset(base_dir, between, use_numpy, transforms=transforms, **kwargs)
        case _:
            logging.warning(f'No target dataset: {dataset_name}')

    return None

def get_valid_dataset(dataset_name: str, base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
    transforms = get_transforms()

    match dataset_name.lower():
        case 'drive':
            return drive_dataset.DriveDataset.get_valid_dataset(base_dir, between, use_numpy, transforms=transforms, **kwargs)
        case 'stare':
            return stare_dataset.StareDataset.get_valid_dataset(base_dir, between, use_numpy, transforms=transforms, **kwargs)
        case 'isic2017':
            return isic2017_dataset.ISIC2017Dataset.get_valid_dataset(base_dir, between, use_numpy, transforms=transforms, **kwargs)
        case 'isic2018':
            return isic2018_dataset.ISIC2018Dataset.get_valid_dataset(base_dir, between, use_numpy, transforms=transforms, **kwargs)
        case 'bowl2018':
            return bowl2018_dataset.BOWL2018Dataset.get_valid_dataset(base_dir, between, use_numpy, transforms=transforms, **kwargs)
        case 'chasedb1':
            return chasedb1_dataset.ChaseDB1Dataset.get_valid_dataset(base_dir, between, use_numpy, transforms=transforms, **kwargs)
        case _:
            logging.warning(f'No target dataset: {dataset_name}')

    return None

def get_test_dataset(dataset_name: str, base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
    transforms = get_transforms()

    match dataset_name.lower():
        case 'drive':
            return drive_dataset.DriveDataset.get_test_dataset(base_dir, between, use_numpy, transforms=transforms, **kwargs)
        case 'stare':
            return stare_dataset.StareDataset.get_test_dataset(base_dir, between, use_numpy, transforms=transforms, **kwargs)
        case 'isic2017':
            return isic2017_dataset.ISIC2017Dataset.get_test_dataset(base_dir, between, use_numpy, transforms=transforms, **kwargs)
        case 'isic2018':
            return isic2018_dataset.ISIC2018Dataset.get_test_dataset(base_dir, between, use_numpy, transforms=transforms, **kwargs)
        case 'bowl2018':
            return bowl2018_dataset.BOWL2018Dataset.get_test_dataset(base_dir, between, use_numpy, transforms=transforms, **kwargs)
        case 'chasedb1':
            return chasedb1_dataset.ChaseDB1Dataset.get_test_dataset(base_dir, between, use_numpy, transforms=transforms, **kwargs)
        case _:
            logging.warning(f'No target dataset: {dataset_name}')

    return None

def to_numpy(dataset_name: str, save_dir: Path, base_dir: Path, betweens: Betweens, **kwargs):
    transforms = get_transforms()

    match dataset_name.lower():
        case 'drive':
            return drive_dataset.DriveDataset.to_numpy(save_dir, base_dir, betweens, transforms=transforms, **kwargs)
        case 'stare':
            return stare_dataset.StareDataset.to_numpy(save_dir, base_dir, betweens, transforms=transforms, **kwargs)
        case 'isic2017':
            return isic2017_dataset.ISIC2017Dataset.to_numpy(save_dir, base_dir, betweens, transforms=transforms, **kwargs)
        case 'isic2018':
            return isic2018_dataset.ISIC2018Dataset.to_numpy(save_dir, base_dir, betweens, transforms=transforms, **kwargs)
        case 'bowl2018':
            return bowl2018_dataset.BOWL2018Dataset.to_numpy(save_dir, base_dir, betweens, transforms=transforms, **kwargs)
        case 'chasedb1':
            return chasedb1_dataset.ChaseDB1Dataset.to_numpy(save_dir, base_dir, betweens, transforms=transforms, **kwargs)
        case _:
            logging.warning(f'No target dataset: {dataset_name}')
