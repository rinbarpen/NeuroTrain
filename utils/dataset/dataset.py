import logging
from pathlib import Path

from utils.dataset import drive_dataset, bowl2018_dataset, chasedb1_dataset, isic2017_dataset, isic2018_dataset, stare_dataset

def get_train_valid_dataset(dataset_name: str, base_dir: Path, split: float=1.0, **kwargs):
    match dataset_name.lower():
        case 'drive':
            return drive_dataset.get_drive_train_valid_dataset(base_dir, split, **kwargs)
        case 'stare':
            return stare_dataset.get_stare_train_valid_dataset(base_dir, split, **kwargs)
        case 'isic2017':
            return isic2017_dataset.get_isic2017_train_dataset(base_dir, **kwargs), isic2017_dataset.get_isic2017_valid_dataset(base_dir, **kwargs)
        case 'isic2018':
            return isic2018_dataset.get_isic2018_train_dataset(base_dir, **kwargs), isic2018_dataset.get_isic2018_valid_dataset(base_dir, **kwargs)
        case 'bowl2018':
            return bowl2018_dataset.get_bowl2018_train_valid_dataset(base_dir, split, **kwargs)
        case 'chasedb1':
            return chasedb1_dataset.get_chasedb1_train_valid_dataset(base_dir, split, **kwargs)
        case _:
            logging.warning(f'No target dataset: {dataset_name}')
    
    return None, None

def get_test_dataset(dataset_name: str, base_dir: Path, **kwargs):
    match dataset_name.lower():
        case 'drive':
            return drive_dataset.get_drive_test_dataset(base_dir, **kwargs)
        case 'stare':
            return stare_dataset.get_stare_test_dataset(base_dir, **kwargs)
        case 'isic2017':
            return isic2017_dataset.get_isic2017_test_dataset(base_dir, **kwargs)
        case 'isic2018':
            return isic2018_dataset.get_isic2018_test_dataset(base_dir, **kwargs)
        case 'bowl2018':
            return bowl2018_dataset.get_bowl2018_test_dataset(base_dir, **kwargs)
        case 'chasedb1':
            return chasedb1_dataset.get_chasedb1_test_dataset(base_dir, **kwargs)
        case _:
            logging.warning(f'No target dataset: {dataset_name}')

    return None, None

def dataset_to_numpy(dataset_name: str, save_dir: Path, base_dir: Path, **kwargs):
    match dataset_name.lower():
        case 'drive':
            return drive_dataset.convert_to_numpy(save_dir, base_dir, **kwargs)
        case 'stare':
            return stare_dataset.convert_to_numpy(save_dir, base_dir, **kwargs)
        case 'isic2017':
            return isic2017_dataset.convert_to_numpy(save_dir, base_dir, **kwargs)
        case 'isic2018':
            return isic2018_dataset.convert_to_numpy(save_dir, base_dir, **kwargs)
        case 'bowl2018':
            return bowl2018_dataset.convert_to_numpy(save_dir, base_dir, **kwargs)
        case 'chasedb1':
            return chasedb1_dataset.convert_to_numpy(save_dir, base_dir, **kwargs)
        case _:
            logging.warning(f'No target dataset: {dataset_name}')

    return None, None
