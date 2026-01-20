import pandas as pd
from pathlib import Path
from fastparquet import ParquetFile
from typing import Literal, Union
from datasets import load_dataset
from .custom_dataset import CustomDataset

class Flickr30kDataset(CustomDataset):
    mapping = {
        'train': 'train',
        'val': 'val',
        'test': 'test'
    }
    def __init__(self, root_dir: Union[str, Path], split: Literal['train', 'val', 'test']):
        super(Flickr30kDataset, self).__init__(root_dir, split)
        self.samples = self._load_samples()
        self.n = len(self.samples)

    def _load_samples(self):
        ds = load_dataset("nlphuji/flickr30k", cache_dir='data', split='test')
        return ds[ds['split'] == self.mapping[self.split]]

if __name__ == '__main__':
    ds = Flickr30kDataset(root_dir='data', split='test')
    print(len(ds))
    print(ds[0])