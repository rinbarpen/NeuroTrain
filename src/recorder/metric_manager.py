from typing import TypedDict, OrderedDict, Union, Optional
import numpy as np
import torch

from .meter import Meter

class MetricManager:
    def __init__(self):
        self.meters: OrderedDict[str, Meter] = OrderedDict()

    def update(self, name: str, val: Union[float, np.ndarray, torch.Tensor], n: int=1):
        if isinstance(val, torch.Tensor):
            val = val.item()
        elif isinstance(val, np.ndarray):
            val = val.item()

        if name not in self.meters:
            self.meters[name] = Meter(name)

        self.meters[name].update(val, n)
        return

    def complete(self):
        for meter in self.meters.values():
            meter.complete()
        return self.meters
    
    def reset_meter(self, name: str):
        self.meters[name].reset()

    def reset_all(self):
        for meter in self.meters.values():
            meter.reset()

    def __str__(self):
        return '\n'.join([str(meter) for meter in self.meters.values()])
    
    def __repr__(self):
        return self.__str__()
    
    def __getitem__(self, name: str) -> Optional[Meter]:
        return self.meters.get(name, None)
    
    def __iter__(self):
        return iter(self.meters.values())
    
    def __contains__(self, name: str) -> bool:
        return name in self.meters
    
    def __len__(self):
        return len(self.meters)
