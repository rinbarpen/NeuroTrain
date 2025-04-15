from torch.utils.data import Dataset
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super(Dataset, self).__init__()
    
    def __getitem__(self, index):
        return super().__getitem__(index)


class ImageClassificationCustomDataset(Dataset):
    def __init__(self, base_dir: Path, between: tuple[float, float], transforms: transforms.Compose|None=None, *, is_rgb: bool, is_numpy: bool=False, **kwargs):
        super(ImageClassificationCustomDataset, self).__init__()

        self.is_rgb = is_rgb
        self.is_numpy = is_numpy
        self.transforms = transforms

        source_pattern: str = kwargs['source']
        target_pattern: str = kwargs['target']

        if is_numpy:
            source_pattern = source_pattern.replace(source_pattern[source_pattern.rfind('.'):], ".npy")

        self.raws   = [p for p in base_dir.glob(source_pattern)] 
        self.labels = self._read_data_file(base_dir / target_pattern)
        self.n = len(self.raws)
        between = (int(between[0] * self.n), int(between[1] * self.n))
        self.raws   = self.raws[between[0]:between[1]]
        self.labels = self.labels[between[0]:between[1]]

    def _read_data_file(self, data_file: Path):
        match data_file.suffix:
            case '.csv':
                df = pd.read_csv(data_file)
            case '.xlsx':
                df = pd.read_excel(data_file)
            case '.parquet':
                df = pd.read_parquet(data_file)
            case '.hdf':
                df = pd.read_hdf(data_file)
            case _:
                return pd.DataFrame([])
        return df

    def __getitem__(self, index):
        source, target = self.raws[index], self.labels[index]
        if self.is_numpy:
            source = torch.from_numpy(np.load(source))
        else:
            if self.is_rgb:
                source = Image.open(source).convert('RGB')
            else:
                source = Image.open(source).convert('L')

            if self.transforms:
                source = self.transforms(source)

        return source, target

    def __len__(self):
        return self.n

class ImageSegmentCustomDataset(Dataset):
    def __init__(self, base_dir: Path, between: tuple[float, float], transforms: transforms.Compose|None=None, *, is_rgb: bool, is_numpy: bool=False, **kwargs):
        super(ImageSegmentCustomDataset, self).__init__()

        self.is_rgb = is_rgb
        self.is_numpy = is_numpy
        self.transforms = transforms

        source_pattern: str = kwargs['source']
        target_pattern: str = kwargs['target']

        if is_numpy:
            source_pattern = source_pattern.replace(source_pattern[source_pattern.rfind('.'):], ".npy")
            target_pattern = target_pattern.replace(target_pattern[target_pattern.rfind('.'):], ".npy")

        self.raws   = [p for p in base_dir.glob(source_pattern)] 
        self.labels = [p for p in base_dir.glob(target_pattern)]
        self.n = len(self.raws)
        between = (int(between[0] * self.n), int(between[1] * self.n))
        self.raws   = self.raws[between[0]:between[1]]
        self.labels = self.labels[between[0]:between[1]]

    def __getitem__(self, index):
        source, target = self.raws[index], self.labels[index]
        if self.is_numpy:
            source, target = torch.from_numpy(np.load(source)), torch.from_numpy(np.load(target))
        else:
            if self.is_rgb:
                source = Image.open(source).convert('RGB')
                target = Image.open(target).convert('RGB')
            else:
                source = Image.open(source).convert('L')
                target = Image.open(target).convert('L')

            if self.transforms:
                source, target = self.transforms(source), self.transforms(target)

        return source, target

    def __len__(self):
        return self.n
