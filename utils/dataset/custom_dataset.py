from torch.utils.data import Dataset
from pathlib import Path
from abc import abstractmethod
from typing import Literal, TypedDict, Tuple

class Betweens(TypedDict):
    train: tuple[float, float]
    valid: tuple[float, float]
    test: tuple[float, float]

class CustomDataset(Dataset):
    mapping=... # {'train': (), 'valid': (), 'test': ()}
    def __init__(self, base_dir: Path, dataset_type: Literal['train', 'test', 'valid'], between: tuple[float, float]=(0.0, 1.0), use_numpy: bool=False):
        super(Dataset, self).__init__()

        self.dataset_type = dataset_type
        self.base_dir = base_dir
        self.between = between
        self.use_numpy = use_numpy
        self.n = 0

    def __len__(self):
        return self.n

    @classmethod
    @abstractmethod
    def __getitem__(self, index):
        ...
        # raise NotImplementedError(f"{self.__name__} hasn't been implemented.")

    @staticmethod
    @abstractmethod
    def to_numpy(save_dir: Path, base_dir: Path, betweens: Betweens, **kwargs):
        ...
        # raise NotImplementedError(f"{self.__name__} hasn't been implemented.")

    @staticmethod
    @abstractmethod
    def name():
        ...
        # raise NotImplementedError(f"{self.__name__} hasn't been implemented.")

    @staticmethod
    @abstractmethod
    def get_train_dataset(base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        ...
        # raise NotImplementedError(f"{self.__name__} hasn't been implemented.")
    @staticmethod
    @abstractmethod
    def get_valid_dataset(base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        ...
        # raise NotImplementedError(f"{self.__name__} hasn't been implemented.")
    @staticmethod
    @abstractmethod
    def get_test_dataset(base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        ...
        # raise NotImplementedError(f"{self.__name__} hasn't been implemented.")
