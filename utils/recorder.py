import os.path
import torch
import numpy as np
import pandas as pd
from config import get_config, is_predict, is_test, is_train

LIST_TYPE = list|np.ndarray|torch.Tensor|tuple

def to_list(l: LIST_TYPE):
    if isinstance(l, np.ndarray):
        l = list(l)
    elif isinstance(l, torch.Tensor):
        l = list(l.detach().cpu().numpy())
    elif isinstance(l, tuple):
        l = list(l)
    return l

class Recorder:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls) 
        return cls

    @staticmethod
    def record_losses(losses: dict[str, LIST_TYPE]):
        losses = to_list(losses)
        df = pd.DataFrame()
        for k, v in losses.items():
            df[k] = to_list(v)
            # df['loss'] = losses

        path = Recorder._save_path()
        df.to_csv(os.path.join(path, 'loss.csv'))
        df.to_parquet(os.path.join(path, 'loss.parquet'))

    @staticmethod
    def record_loss(loss: LIST_TYPE):
        loss = to_list(loss)
        df = pd.DataFrame()
        df['loss'] = loss

        path = Recorder._save_path()
        df.to_csv(os.path.join(path, 'loss.csv'))
        df.to_parquet(os.path.join(path, 'loss.parquet'))


    @staticmethod
    def record_metrics(metrics: dict[str, LIST_TYPE]):
        # metrics: {'f1': [epoch f1], 'recall': [epoch recall]}
        df = pd.DataFrame()
        for k, v in metrics.items():
            df[k] = to_list(v)

        path = Recorder._save_path()
        df.to_csv(os.path.join(path, 'metrics.csv'))
        df.to_parquet(os.path.join(path, 'metrics.parquet'))

    @staticmethod
    def _save_path():
        CONFIG = get_config()
        if is_train():
            mode = 'train'
        elif is_test():
            mode = 'test'
        elif is_predict():
            mode = 'predict'
        else:
            raise ValueError('Not supported!')

        path = os.path.join(CONFIG['output_dir'], CONFIG['run_id'], CONFIG['task'], mode)
        os.makedirs(path, exist_ok=True)
        return path
