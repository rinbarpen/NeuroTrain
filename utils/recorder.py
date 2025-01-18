import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from config import get_config, is_predict, is_test, is_train

LIST_TYPE = list|np.ndarray|torch.Tensor|tuple

def to_numpy(l: LIST_TYPE):
    if isinstance(l, list):
        l = np.array(l)
    elif isinstance(l, torch.Tensor):
        l = l.detach().cpu().numpy()
    elif isinstance(l, tuple):
        l = np.array(l)
    return l

class Recorder:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls) 
        return cls

    @staticmethod
    def record_losses(losses: dict[str, LIST_TYPE], path: Path):
        losses = to_numpy(losses)
        df = pd.DataFrame()
        for k, v in losses.items():
            df[k] = to_numpy(v)
            # df['loss'] = losses

        df.to_csv(path / 'loss.csv')
        df.to_parquet(path / 'loss.parquet')
        logging.info(f'Save loss data under the {path}')

    @staticmethod
    def record_loss(loss: LIST_TYPE, path: Path):
        loss = to_numpy(loss)
        df = pd.DataFrame()
        df['loss'] = loss

        df.to_csv(path / 'loss.csv')
        df.to_parquet(path / 'loss.parquet')
        logging.info(f'Save loss data under the {path}')


    @staticmethod
    def record_metrics(metrics: dict[str, LIST_TYPE], path: Path):
        # metrics: {'f1': [epoch f1], 'recall': [epoch recall]}
        df = pd.DataFrame()
        for k, v in metrics.items():
            df[k] = to_numpy(v)

        df.to_csv(path / 'metrics.csv')
        df.to_parquet(path / 'metrics.parquet')
        logging.info(f'Save metric data under the {path}')
