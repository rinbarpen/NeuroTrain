import logging
from pathlib import Path
from typing import Literal
from torch.utils.data import DataLoader


from src.config import get_config_value, get_config
from src.utils.transform import get_transforms
from .custom_dataset import Betweens, CustomDataset

def _get_dataset_by_case(dataset_name: str):
    name = dataset_name.lower()
    if name == 'btcv':
        from .btcv_dataset import BTCVDataset
        return BTCVDataset
    elif name == 'drive':
        from .drive_dataset import DriveDataset
        return DriveDataset
    elif name == 'stare':
        from .stare_dataset import StareDataset
        return StareDataset
    elif name == 'isic2017':
        from .isic2017_dataset import ISIC2017Dataset
        return ISIC2017Dataset
    elif name == 'isic2018':
        from .isic2018_dataset import ISIC2018Dataset
        return ISIC2018Dataset
    elif name == 'bowl2018':
        from .bowl2018_dataset import BOWL2018Dataset
        return BOWL2018Dataset
    elif name == 'chasedb1':
        from .chasedb1_dataset import ChaseDB1Dataset
        return ChaseDB1Dataset
    elif name == 'mnist':
        from .mnist_dataset import MNISTDataset
        return MNISTDataset
    else:
        logging.warning(f'No target dataset: {dataset_name}')
        return None

def get_train_dataset(dataset_name: str, base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
    transforms = get_transforms()
    if dataset_name.lower() == 'cholect50':
        return 
    DatasetClass = _get_dataset_by_case(dataset_name)
    if DatasetClass is None:
        return None
    if dataset_name.lower() == 'btcv':
        return DatasetClass.get_train_dataset(base_dir, use_numpy=use_numpy, transforms=transforms, **kwargs)
    else:
        return DatasetClass.get_train_dataset(base_dir, between, use_numpy, transforms=transforms, **kwargs)

def get_valid_dataset(dataset_name: str, base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
    transforms = get_transforms()
    DatasetClass = _get_dataset_by_case(dataset_name)
    if DatasetClass is None:
        return None
    if dataset_name.lower() == 'btcv':
        return DatasetClass.get_valid_dataset(base_dir, use_numpy=use_numpy, transforms=transforms, **kwargs)
    else:
        return DatasetClass.get_valid_dataset(base_dir, between, use_numpy, transforms=transforms, **kwargs)

def get_test_dataset(dataset_name: str, base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
    transforms = get_transforms()
    DatasetClass = _get_dataset_by_case(dataset_name)
    if DatasetClass is None:
        return None
    if dataset_name.lower() == 'btcv':
        return DatasetClass.get_test_dataset(base_dir, use_numpy=use_numpy, transforms=transforms, **kwargs)
    else:
        return DatasetClass.get_test_dataset(base_dir, between, use_numpy, transforms=transforms, **kwargs)

def to_numpy(dataset_name: str, save_dir: Path, base_dir: Path, betweens: Betweens, **kwargs):
    transforms = get_transforms()
    DatasetClass = _get_dataset_by_case(dataset_name)
    if DatasetClass is None:
        return None
    return DatasetClass.to_numpy(save_dir, base_dir, betweens, transforms=transforms, **kwargs)

def get_dataset(mode: Literal['train', 'test', 'valid']):
    c_dataset = get_config_value('dataset')
    assert c_dataset is not None, "Dataset configuration is not defined in the config file."

    config = c_dataset.get('config', {})
    match mode:
        case 'train':
            dataset0 = get_train_dataset(c_dataset['name'], Path(c_dataset['base_dir']), c_dataset.get('betweens', {'train': (0, 1)})['train'], **config)
        case 'test':
            dataset0 = get_test_dataset(c_dataset['name'], Path(c_dataset['base_dir']), c_dataset.get('betweens', {'test': (0, 1)})['test'], **config)
        case 'valid':
            dataset0 = get_valid_dataset(c_dataset['name'], Path(c_dataset['base_dir']), c_dataset.get('betweens', {'valid': (0, 1)})['valid'], **config)

    if not dataset0:
        logging.error(f"{c_dataset['name']} is empty!")

    return dataset0


def random_sample(dataset: CustomDataset, sample_ratio: float=0.1, generator=None):
    from torch.utils.data import RandomSampler
    num_samples = int(sample_ratio * len(dataset))
    return RandomSampler(dataset, num_samples=num_samples, generator=generator)

def get_train_valid_test_dataloader(use_valid=False):
    c = get_config()

    train_dataset = get_dataset('train')
    test_dataset = get_dataset('test')
    num_workers = c["dataloader"]["num_workers"]
    shuffle = c["dataloader"]["shuffle"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=c["train"]["batch_size"],
        pin_memory=True,
        num_workers=num_workers,
        shuffle=shuffle,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=c["test"]["batch_size"],
        pin_memory=True,
        num_workers=num_workers,
        shuffle=shuffle,
    )
    if use_valid:
        valid_dataset = get_dataset('valid')
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=c["valid"]["batch_size"],
            pin_memory=True,
            num_workers=num_workers,
            shuffle=shuffle,
        )

        return train_loader, valid_loader, test_loader

    return train_loader, None, test_loader
