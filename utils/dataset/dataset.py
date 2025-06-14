import logging
from pathlib import Path
from torch.utils.data import Dataset
from typing import Literal

from config import get_config_value
from utils.dataset.custom_dataset import Betweens
from utils.dataset import drive_dataset, bowl2018_dataset, chasedb1_dataset, isic2017_dataset, isic2018_dataset, stare_dataset
from utils.transform import get_transforms
from utils.dataset import btcv_dataset

def get_train_dataset(dataset_name: str, base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
    transforms = get_transforms()

    match dataset_name.lower():
        case 'btcv':
            return btcv_dataset.BTCVDataset.get_train_dataset(base_dir, use_numpy=use_numpy, transforms=transforms, **kwargs)
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
        case 'btcv':
            return btcv_dataset.BTCVDataset.get_valid_dataset(base_dir, use_numpy=use_numpy, transforms=transforms, **kwargs)
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
        case 'btcv':
            return btcv_dataset.BTCVDataset.get_test_dataset(base_dir, use_numpy=use_numpy, transforms=transforms, **kwargs)
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

class ChainedDatasets(Dataset):
    def __init__(self, datasets: list[Dataset]|Dataset):
        super(ChainedDatasets, self).__init__()

        if isinstance(datasets, Dataset):
            self.datasets = [datasets]
        else:
            self.datasets = datasets

        self.lens = [len(dataset) for dataset in self.datasets]

    def __getitem__(self, index):
        dataset_index = 0
        while index >= self.lens[dataset_index]:
            index -= self.lens[dataset_index]
            dataset_index += 1
        return self.datasets[dataset_index][index]

    def __len__(self):
        s = 0
        for i in self.lens:
            s += i
        return s

def get_chained_datasets(mode: Literal['train', 'test', 'valid']):
    c_datasets = get_config_value('datasets')
    assert c_datasets is not None

    def get_dataset(mode: Literal['train', 'test', 'valid'], dataset):
        config = dataset['config'] if 'config' in dataset else {}
        match mode:
            case 'train':
                dataset0 = get_train_dataset(dataset['name'], Path(dataset['base_dir']), dataset['betweens']['train'], **config)
            case 'test':
                dataset0 = get_test_dataset(dataset['name'], Path(dataset['base_dir']), dataset['betweens']['test'], **config)
            case 'valid':
                dataset0 = get_valid_dataset(dataset['name'], Path(dataset['base_dir']), dataset['betweens']['valid'], **config)

        if not dataset0:
            logging.error(f"{dataset['name']} is empty!")

        return dataset

    return ChainedDatasets([get_dataset(mode, dataset) for dataset in c_datasets])
