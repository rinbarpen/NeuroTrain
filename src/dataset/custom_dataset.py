from torch.utils.data import Dataset, Subset
from pathlib import Path
from abc import abstractmethod
from typing import Literal, TypedDict, Tuple, Optional, List, Sequence
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

# 保留Betweens类型定义，因为其他地方可能还在使用
class Betweens(TypedDict):
    train: tuple[float, float]
    valid: tuple[float, float]
    test: tuple[float, float]

class CustomDataset(Dataset):
    """数据集基类，提供标准化的数据集接口"""
    mapping = ... # {'train': (), 'valid': (), 'test': ()}
    
    def __init__(self, root_dir: Path, split: str, desired_n: int = 0, **kwargs):
        """
        初始化数据集基类
        
        Args:
            root_dir: 数据集根目录
            split: 数据集类型 ('train', 'test', 'valid', 'val', 'xx')
            desired_n: 期望的样本数量，默认0
            **kwargs: 其他扩展参数
        """
        super(Dataset, self).__init__()

        self.split = split
        self.root_dir = root_dir
        self.n = desired_n  # 数据集样本数量

        for k,v in kwargs.items():
            setattr(self, k, v)
        
        self.samples = []

    def __len__(self):
        """返回数据集样本数量"""
        return self.n

    @abstractmethod
    def __getitem__(self, index):
        """获取指定索引的数据样本"""
        ...

    @staticmethod
    @abstractmethod
    def to_numpy(save_dir: Path, root_dir: Path, betweens: Betweens, **kwargs):
        """将数据集转换为numpy格式保存"""
        ...

    @staticmethod
    @abstractmethod
    def name() -> str:
        """返回数据集名称"""
        ...

    @staticmethod
    @abstractmethod
    def get_train_dataset(root_dir: Path, **kwargs):
        """获取训练数据集"""
        ...
    
    @staticmethod
    @abstractmethod
    def get_valid_dataset(root_dir: Path, **kwargs):
        """获取验证数据集"""
        ...
    
    @staticmethod
    @abstractmethod
    def get_test_dataset(root_dir: Path, **kwargs):
        """获取测试数据集"""
        ...

    # ------------------------------
    # 通用划分与交叉验证工具
    # ------------------------------
    def train_valid_test_split(
        self,
        train: float = 0.8,
        valid: float = 0.1,
        test: float = 0.1,
        *,
        shuffle: bool = True,
        random_state: Optional[int] = None,
    ) -> tuple[Optional[Subset], Optional[Subset], Optional[Subset]]:
        """将当前数据集按比例切分为 train/valid/test 三个子集
        
        注意：
        - 该函数基于索引切分并返回 `torch.utils.data.Subset`，不会复制底层数据。
        - 适用于通用数据集，具体任务的数据增强与读取逻辑仍在各自数据集中实现。
        
        参数:
            train: 训练集比例，取值范围[0, 1]。
            valid: 验证集比例，取值范围[0, 1]。
            test: 测试集比例，取值范围[0, 1]。若 train + valid < 1，则 test 自动补足剩余比例；若三者之和>1将抛出异常。
            shuffle: 是否在切分前打乱索引。
            random_state: 随机种子，便于复现实验。
        
        返回:
            (train_subset, valid_subset, test_subset) 三元组，其中某一比例为0时对应位置返回 None。
        """
        n = len(self)
        if n <= 0:
            return None, None, None

        # 比例检查与自动补全
        eps = 1e-8
        s = train + valid + test
        if s > 1.0 + eps:
            raise ValueError(f"train/valid/test 比例之和不能超过1：当前={s}")
        if s < 1.0 - eps:
            # 剩余归入 test
            test = 1.0 - (train + valid)

        rng = np.random.default_rng(random_state)
        indices = np.arange(n)
        if shuffle:
            rng.shuffle(indices)

        train_size = int(np.floor(train * n))
        valid_size = int(np.floor(valid * n))
        # 余数全部划入 test，确保三者之和 == n
        test_size = n - train_size - valid_size

        train_idx = indices[:train_size].tolist() if train_size > 0 else []
        valid_idx = indices[train_size:train_size + valid_size].tolist() if valid_size > 0 else []
        test_idx = indices[train_size + valid_size:].tolist() if test_size > 0 else []

        train_subset = Subset(self, train_idx) if train_idx else None
        valid_subset = Subset(self, valid_idx) if valid_idx else None
        test_subset = Subset(self, test_idx) if test_idx else None
        return train_subset, valid_subset, test_subset

    @staticmethod
    def kfold(
        dataset: "CustomDataset",
        n_splits: int = 5,
        *,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        include_test: bool = False,
        test_ratio: float = 0.0,
    ) -> List[tuple[Subset, Subset, Optional[Subset]]]:
        """对给定数据集执行 K 折交叉验证划分
        
        说明：
        - 默认返回长度为 `n_splits` 的列表，每个元素是 (train_subset, valid_subset, test_subset) 三元组；
          当 include_test=False 时，test_subset 恒为 None。
        - 不依赖外部库（如 sklearn），通过 numpy 实现索引级别的折分。
        
        参数:
            dataset: 需要划分的完整数据集实例。
            n_splits: 折数，必须 >= 2。
            shuffle: 是否在折分前打乱索引。
            random_state: 随机种子，便于复现。
            include_test: 是否在每折中为测试集预留固定比例的样本（从训练索引中再划分）。
            test_ratio: 每折中从训练索引里抽取的测试集比例，取值范围[0, 1)。当 include_test=True 且 test_ratio>0 时生效。
        
        返回:
            列表，每个元素为 (train_subset, valid_subset, test_subset)。
        """
        if n_splits < 2:
            raise ValueError("n_splits 必须 >= 2")

        n = len(dataset)
        if n == 0:
            return []

        rng = np.random.default_rng(random_state)
        indices = np.arange(n)
        if shuffle:
            rng.shuffle(indices)

        # 将索引均匀切分为 n_splits 份，最后一份可能更长
        folds = np.array_split(indices, n_splits)

        results: List[tuple[Subset, Subset, Optional[Subset]]] = []
        for i in range(n_splits):
            valid_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i])

            test_subset = None
            if include_test and test_ratio > 0:
                # 从 train_idx 内部再划分一部分作为 test
                t_n = len(train_idx)
                t_test = int(np.floor(test_ratio * t_n))
                if t_test > 0:
                    if shuffle:
                        rng.shuffle(train_idx)
                    test_idx = train_idx[:t_test]
                    train_idx = train_idx[t_test:]
                    test_subset = Subset(dataset, test_idx.tolist())

            train_subset = Subset(dataset, train_idx.tolist()) if len(train_idx) > 0 else None
            valid_subset = Subset(dataset, valid_idx.tolist()) if len(valid_idx) > 0 else None

            results.append((train_subset, valid_subset, test_subset))

        return results

    def mininalize(self, dataset_size: float|None = 0.1, random_sample: bool = True):
        """将数据集转换为最小化形式，移除所有元数据
        
        修复说明：
        - 非随机采样不再尝试对 SequentialSampler 进行切片（该对象不可下标）。
        - 改为使用顺序采样前 dataset_size 个样本的索引，并更新 self.n。
        """
        # 计算目标样本数
        if dataset_size is None or dataset_size <= 1.0:
            dataset_size = int(dataset_size * len(self))
        else:
            dataset_size = int(dataset_size)
        # 边界保护
        dataset_size = max(0, min(dataset_size, len(self)))
        # 更新当前数据集的可见长度
        self.n = dataset_size

        if random_sample:
            # 随机采样指定数量样本
            self.sampler = RandomSampler(self, num_samples=dataset_size)
        else:
            # 顺序采样前 dataset_size 个样本（0..dataset_size-1）
            # 注意：SequentialSampler 会按 data_source 的长度生成 0..len-1 的索引，
            # 这里使用 range(dataset_size) 即得到前 N 个样本的顺序索引。
            self.sampler = SequentialSampler(range(dataset_size))
        return self

    def dataloader(self, batch_size: int, shuffle: bool = True, num_workers: int = 0, pin_memory: bool=True, drop_last: bool=False, collate_fn=None) -> DataLoader:
        # 当提供 sampler 时，必须禁用 shuffle 以避免冲突
        sampler = getattr(self, 'sampler', None)
        if sampler is not None and shuffle:
            shuffle = False
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, collate_fn=collate_fn)

    def get_dataset(self, splits: str|Sequence[str], **kwargs):
        if isinstance(splits, str):
            splits = [splits]
        
        datasets = []
        for dt in splits:
            if dt == 'train':
                dataset = self.get_train_dataset(**kwargs['train'])
            elif dt == 'valid' or dt == 'val':
                dataset = self.get_valid_dataset(**kwargs['valid'])
            elif dt == 'test':
                dataset = self.get_test_dataset(**kwargs['test'])
            datasets.append(dataset)
        return datasets

    def get_train_valid_test_dataset(self, **kwargs):
        return self.get_dataset(['train', 'valid', 'test'], **kwargs)
    
    def get_train_valid_dataset(self, **kwargs):
        return self.get_dataset(['train', 'valid'], **kwargs)
    
    def get_train_test_dataset(self, **kwargs):
        return self.get_dataset(['train', 'test'], **kwargs)

class TransformersDataset(CustomDataset):
    def __init__(self, root_dir: Path, split: str, desired_n: int=0, processor=None, tokenizer=None, **kwargs):
        self.tokenizer = tokenizer
        self.processor = processor
        self.tokenizer_args = kwargs.pop('tokenizer', {})
        self.processor_args = kwargs.pop('processor', {})
        super().__init__(root_dir, split, desired_n=desired_n, **kwargs)
    
    def get_dataset(self, splits: str|Sequence[str], processor=None, tokenizer=None, **kwargs):
        if isinstance(splits, str):
            splits = [splits]
        
        datasets = []
        for dt in splits:
            if dt == 'train':
                dataset = self.get_train_dataset(**kwargs['train'], tokenizer=tokenizer, processor=processor)
            elif dt == 'valid' or dt == 'val':
                dataset = self.get_valid_dataset(**kwargs['valid'], tokenizer=tokenizer, processor=processor)
            elif dt == 'test':
                dataset = self.get_test_dataset(**kwargs['test'], tokenizer=tokenizer, processor=processor)
            datasets.append(dataset)
        return datasets
