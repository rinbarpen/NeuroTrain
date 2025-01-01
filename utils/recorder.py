import os.path

import numpy as np
import pandas as pd
from lancedb.dependencies import torch

from config import CONFIG, IS_INFERENCE, IS_TRAIN, IS_TEST


# import wandb

LIST_TYPE = list|np.ndarray|torch.Tensor|tuple

class Recorder:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls

    @staticmethod
    def record_loss(losses: LIST_TYPE):
        losses = Recorder._to_list(losses)
        df = pd.DataFrame()
        df['epoch'] = [i+1 for i in range(len(losses))]
        df['loss'] = losses

        mode = Recorder._mode()
        path = os.path.join(CONFIG['output'], CONFIG['task'], CONFIG['task_class'], mode)
        os.makedirs(path, exist_ok=True)
        df.to_csv(os.path.join(path, 'loss.csv'))
        df.to_parquet(os.path.join(path, 'loss.parquet'))

    # TODO
    @staticmethod
    def record_metrics(metrics: dict[str, LIST_TYPE]):
        df = pd.DataFrame()
        for k, v in metrics.items():
            df[k] = Recorder._to_list(v)

        mode = Recorder._mode()
        path = os.path.join(CONFIG['output'], CONFIG['task'], CONFIG['task_class'], mode)
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def _to_list(l: LIST_TYPE):
        if isinstance(l, np.ndarray):
            l = list(l)
        elif isinstance(l, torch.Tensor):
            l = list(l.detach().cpu().numpy())
        elif isinstance(l, tuple):
            l = list(tuple)
        return l

    @staticmethod
    def _mode():
        if IS_TRAIN:
            return 'train'
        if IS_TEST:
            return 'test'
        if IS_INFERENCE:
            return 'inference'
        raise ValueError('Not supported!')
