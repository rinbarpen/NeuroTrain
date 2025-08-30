import pandas as pd
import numpy as np
import torch
from typing import Literal, List, Sequence, TextIO, Iterator
from pathlib import Path

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

    def update(self, val: float|np.ndarray|torch.Tensor, n: int=1):
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
    
    def all_reduce(self):
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(torch.Tensor([self.sum]))
            torch.distributed.all_reduce(torch.Tensor([self.count]))
            self.avg = (self.sum / self.count).item()
            self.sum = self.sum.item()
            self.count = self.count.item()
            self.name = self.name + f'_{torch.distributed.get_rank()}'

    def complete(self):
        self.all_reduce()
        return self.avg, self.sum, self.count


class Meter:
    def __init__(self, name: str, fmt: str=':4f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.vals = []
    
    @property
    def avg(self):
        return np.mean(self.vals)
    
    @property
    def sum(self):
        return np.sum(self.vals)
    
    @property
    def count(self):
        return len(self.vals)
    
    def update(self, val: float|np.ndarray|torch.Tensor, n: int=1):
        if isinstance(val, torch.Tensor):
            val = val.item()
        elif isinstance(val, np.ndarray):
            val = val.item()
        self.vals.extend([val] * n)

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '} ({sum' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def __repr__(self):
        return self.__str__()

    def paint(self, filename: Path, title: str|None=None, xlabel: str|None=None, ylabel: str|None=None, xlim: tuple[int, int]|None=None, ylim: tuple[int, int]|None=None):
        from utils.painter import Plot
        if title is None: 
            title = self.name
        if xlim is None: 
            xlim = (0, self.count)
        if ylim is None: 
            ylim = (0, np.max(self.vals))

        if not filename.parent.exists():
            filename.mkdir(parents=True)

        plot = Plot()
        subplot = plot.subplot()
        subplot = subplot.plot(np.arange(1, self.count + 1, dtype=np.int32), np.array(self.vals, dtype=np.float32))
        subplot = subplot.title(title)
        if xlabel is not None:
            subplot = subplot.xlabel(xlabel)
        if ylabel is not None:
            subplot = subplot.ylabel(ylabel)
        subplot = subplot.lim(xlim, ylim)
        plot = subplot.complete()
        plot.save(filename)

    # filename: {name}.csv, {name}.parquet
    def save(self, filename: Path, name: str|None=None):
        if name is None:
            name = self.name
        df = pd.DataFrame({name: self.vals})
        df.to_csv(filename.with_suffix('.csv'), index=False)
        df.to_parquet(filename.with_suffix('.parquet'), index=False)

    def save_as(self, filename: Path, name: str|None=None):
        if name is None:
            name = self.name
        # send to DataSaver 
        DataSaver.save(filename, name, self.vals)


import queue
import threading
class DataSaver:
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self._queue = queue.Queue()

        self._mapping: dict[str, pd.DataFrame] = {}

        self._thread = threading.Thread(target=self._run)
        self._running = True
        self._thread.start()

    def _run(self):
        while self._running:
            try:
                filename, df = self._queue.get()
                self._mapping[filename] = pd.concat([self._mapping.get(filename, pd.DataFrame()), df], axis=1, inplace=True, ignore_index=True, sort=False)
            except queue.Empty:
                pass

    @staticmethod
    def complete():
        DataSaver._instance._running = False
        DataSaver._instance._queue.join()
        DataSaver._instance._thread.join()
    
    @staticmethod
    def save_to_local():
        for filename, df in DataSaver._instance._mapping.items():
            df.to_csv(filename.with_suffix('.csv'), index=False)
            df.to_parquet(filename.with_suffix('.parquet'), index=False)
        DataSaver._instance._mapping.clear()

    @staticmethod
    def save(filename: Path, name: str, vals: Sequence[float]):
        df = pd.DataFrame({name: vals})
        DataSaver._queue.put((filename, df))
