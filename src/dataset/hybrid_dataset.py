from .custom_dataset import CustomDataset
import numpy as np
import torch

from typing import Sequence

class HybridDataset(CustomDataset):
    def __init__(self, datasets: list[CustomDataset]):
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.cumulative_lengths = np.cumsum(self.lengths)
    
    def __len__(self):
        return self.cumulative_lengths[-1]
    
    def __getitem__(self, index: int):
        dataset_index = np.searchsorted(self.cumulative_lengths, index, side='right')
        local_index = index if dataset_index == 0 else index - self.cumulative_lengths[dataset_index - 1]
        return self.datasets[dataset_index][local_index]
    
    def name(self):
        return f'HybridDataset({", ".join([dataset.name() for dataset in self.datasets])})'

    def get_train_dataset(self, datasets: list[CustomDataset]):
        ...

    def get_valid_dataset(self, datasets: list[CustomDataset]):
        ...
    
    def get_test_dataset(self, datasets: list[CustomDataset]):
        ...
    
    def get_dataset(self, datasets: list[CustomDataset], dataset_type: str|Sequence[str]):
        if isinstance(dataset_type, str):
            dataset_type = [dataset_type]
        if 'train' in dataset_type:
            train_dataset = self.get_train_dataset(datasets)
        else:
            train_dataset = None
        if 'valid' in dataset_type:
            valid_dataset = self.get_valid_dataset(datasets)
        else:
            valid_dataset = None
        if 'test' in dataset_type:
            test_dataset = self.get_test_dataset(datasets)
        else:
            test_dataset = None
        
        output_dataset = []
        for dt in dataset_type:
            if dt == 'train':
                output_dataset.append(train_dataset)
            elif dt == 'valid':
                output_dataset.append(valid_dataset)
            elif dt == 'test':
                output_dataset.append(test_dataset)
        return tuple(output_dataset)
    
    def get_train_valid_test_dataset(self, datasets: list[CustomDataset]):
        return self.get_dataset(datasets, ['train', 'valid', 'test'])
    
    def get_train_valid_dataset(self, datasets: list[CustomDataset]):
        return self.get_dataset(datasets, ['train', 'valid'])
    
    def get_train_test_dataset(self, datasets: list[CustomDataset]):
        return self.get_dataset(datasets, ['train', 'test'])
