import pandas as pd
import numpy as np
import torch
from typing import Literal, List, Sequence, TextIO, Iterator, Union, Optional
from pathlib import Path
import torch.distributed as dist

from src.utils.ndict import NDict
# from utils.db import DB  # 暂时注释掉，避免依赖问题
# from visualizer.painter import Plot  # 暂时注释掉，避免依赖问题
# from .data_saver import DataSaver  # 暂时注释掉，避免依赖问题

class MiniMeter:
    def __init__(self, name: str, fmt: str=':4f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: Union[float, np.ndarray, torch.Tensor], n: int=1):
        if isinstance(val, torch.Tensor):
            val = val.item()
        elif isinstance(val, np.ndarray):
            val = val.item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def __repr__(self):
        return self.__str__()
    
    def sync(self):
        if not dist.is_available() or not dist.is_initialized():
            return

        t = torch.tensor([self.sum, self.count], dtype=torch.float32, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        self.sum, self.count = t.tolist()
        self.avg = (self.sum / self.count).item()
        self.sum = self.sum.item()
        self.count = self.count.item()
        self.name = self.name + f'_{dist.get_rank()}'

    def complete(self) -> NDict:
        self.sync()
        return NDict({
            'avg': self.avg,
            'sum': self.sum,
            'count': self.count
        })

_meter_manager: dict[str, 'Meter'] = {}
class Meter:
    def __init__(self, name: str, fmt: str=':4f'):
        self.name = name
        self.fmt = fmt
        self.reset()

        _meter_manager.update({name: self})

    def reset(self):
        self._vals = []
    
    @property
    def val(self) -> np.float64:
        return np.float64(self._vals[-1])
    
    @property
    def vals(self) -> np.ndarray:
        return np.array(self._vals, dtype=np.float64)

    @property
    def avg(self) -> np.float64:
        return np.float64(np.mean(self._vals))
    
    @property
    def sum(self) -> np.float64:
        return np.float64(np.sum(self._vals))
    
    @property
    def count(self) -> int:
        return len(self._vals)
    
    def update(self, val: Union[float, np.ndarray, torch.Tensor], n: int=1):
        if isinstance(val, torch.Tensor):
            val = val.item()
        elif isinstance(val, np.ndarray):
            val = val.item()
        self._vals.extend([val] * n)

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '} ({sum' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def __repr__(self):
        return self.__str__()
    
    def sync(self):
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        sum, count = t.tolist()
        avg = (sum / count).item()
        sum = sum.item()
        count = count.item()
        self.name = self.name + f'_{dist.get_rank()}'
    
    def complete(self):
        self.sync()
        return {'avg': self.avg, 'sum': self.sum, 'count': self.count}
    
    @staticmethod
    def instance(name: str) -> Optional['Meter']:
        return _meter_manager.get(name, None)
    
    @staticmethod
    def available_instances() -> list[str]:
        return list(_meter_manager.keys())
    
    def paint(self, filename: Path, title: Optional[str]=None, xlabel: Optional[str]=None, ylabel: Optional[str]=None, xlim: Optional[tuple[int, int]]=None, ylim: Optional[tuple[int, int]]=None):
        # 暂时注释掉绘图功能，避免依赖问题
        # if title is None: 
        #     title = self.name
        # if xlim is None: 
        #     xlim = (0, self.count)
        # if ylim is None: 
        #     ylim = (0, np.max(self.vals))

        # if not filename.parent.exists():
        #     filename.mkdir(parents=True)

        # plot = Plot()
        # subplot = plot.subplot()
        # subplot = subplot.plot(np.arange(1, self.count + 1, dtype=np.int32), np.array(self.vals, dtype=np.float32))
        # subplot = subplot.title(title)
        # if xlabel is not None:
        #     subplot = subplot.xlabel(xlabel)
        # if ylabel is not None:
        #     subplot = subplot.ylabel(ylabel)
        # subplot = subplot.xlim(xlim[0], xlim[1])
        # subplot = subplot.ylim(ylim[0], ylim[1])
        # plot = subplot.complete()
        # plot.save(filename)
        pass

    # filename: {name}.csv, {name}.parquet
    def save(self, filename: Path, name: Optional[str]=None, to_csv=True, to_parquet=True):
        if name is None:
            name = self.name
        df = pd.DataFrame({name: self.vals})
        if to_csv:
            df.to_csv(filename.with_suffix('.csv'), index=False)
        if to_parquet:
            df.to_parquet(filename.with_suffix('.parquet'), index=False)
    
    def save_to_db(self, db_name: str):
        # 暂时注释掉DB功能，避免依赖问题
        # with DB(db_name) as db:
        #     db.update(self.name, self.vals.tolist())
        pass

    def save_as(self, filename: Path, name: Optional[str]=None):
        if name is None:
            name = self.name
        # send to DataSaver 
        # DataSaver.save(filename, name, self.vals.tolist())  # 暂时注释掉，避免依赖问题
        pass


