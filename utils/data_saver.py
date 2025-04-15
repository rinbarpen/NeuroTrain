import threading
from concurrent.futures import ThreadPoolExecutor 
import numpy as np
import pandas as pd
import fastparquet

from utils.typed import (ClassLabelManyScoreDict,   ClassLabelOneScoreDict, ClassMetricOneScoreDict,
    MetricClassManyScoreDict, MetricClassOneScoreDict, MetricLabelOneScoreDict)
from typing import Literal
from pathlib import Path

# sync mode
class _DataSaver:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def save_loss(self, losses: list[np.ndarray], label: Literal['train', 'valid']):
        csv_filename, parquet_filename = self._get_filenames(f"{label}/loss")

        df_prev = self._prev_df(csv_filename, parquet_filename)

        df = pd.DataFrame({"loss": losses})
        df = pd.concat([df_prev, df], ignore_index=True)

        self._save_dataframe(df, csv_filename, parquet_filename)

    def save_mean_metric(self, metric_mean_score: MetricLabelOneScoreDict, label: Literal['train', 'test', 'valid']):
        csv_filename, parquet_filename = self._get_filenames(f"{label}/mean_metric")

        df = pd.DataFrame({k: [v] for k, v in metric_mean_score.items()})
        self._save_dataframe(df, csv_filename, parquet_filename)

    # for every class
    def save_mean_metric_by_class(self, class_metric_mean_score: ClassMetricOneScoreDict, label: Literal['train', 'test', 'valid']):
        for class_label, metric_mean in class_metric_mean_score.items():
            csv_filename, parquet_filename = self._get_filenames(f"{label}/{class_label}/mean_metric")

            df = pd.DataFrame({k: [v] for k, v in metric_mean.items()})
            self._save_dataframe(df, csv_filename, parquet_filename)

    def save_all_metric_by_class(
        self, class_metric_all_score: ClassLabelManyScoreDict, label: Literal['train', 'test', 'valid']
    ):
        for class_label, metric_all in class_metric_all_score.items():
            csv_filename, parquet_filename = self._get_filenames(f"{label}/{class_label}/all_metric")

            df_prev = self._prev_df(csv_filename, parquet_filename)

            df = pd.DataFrame(metric_all)
            df = pd.concat([df_prev, df], ignore_index=True)

            self._save_dataframe(df, csv_filename, parquet_filename)

    def _prev_df(self, csv_filename: str, parquet_filename: str):
        try:
            pf = fastparquet.ParquetFile(parquet_filename)
            df_prev = pf.to_pandas()
        except Exception:
            df_prev = pd.read_csv(csv_filename)
        return df_prev

    def _save_dataframe(
        self, df: pd.DataFrame, csv_filename: str, parquet_filename: str
    ):
        df.to_csv(csv_filename)
        fastparquet.write(parquet_filename, df)

    def _get_filenames(self, back: str):
        csv_filename = self.base_dir / (back + ".csv")
        parquet_filename = self.base_dir / (back + ".parquet")
        csv_filename.mkdir(parents=True, exist_ok=True)
        parquet_filename.mkdir(parents=True, exist_ok=True)
        return csv_filename, parquet_filename

# async mode
class DataSaver:
    def __init__(self, base_dir: Path, num_threads: int=1, *, async_mode=True):
        self.executor = ThreadPoolExecutor(num_threads)
        self.saver = _DataSaver(base_dir)
        self.async_mode = async_mode

    def save_loss(self, losses: list[np.ndarray], label: Literal['train', 'valid']):
        if self.async_mode:
            self.executor.submit(self.saver.save_loss, losses, label)
        else:
            self.saver.save_loss(losses, label)         

    def save_mean_metric(self, metric_mean_score: MetricLabelOneScoreDict, label: Literal['train', 'test', 'valid']):
        if self.async_mode:
            self.executor.submit(self.saver.save_mean_metric, metric_mean_score, label)
        else:
            self.saver.save_mean_metric(metric_mean_score, label)

    # for every class
    def save_mean_metric_by_class(self, class_metric_mean_score: ClassMetricOneScoreDict, label: Literal['train', 'test', 'valid']):
        if self.async_mode:
            self.executor.submit(self.saver.save_mean_metric_by_class, class_metric_mean_score, label)
        else:
            self.saver.save_mean_metric_by_class(class_metric_mean_score, label)

    def save_all_metric_by_class(
        self, class_metric_all_score: ClassLabelManyScoreDict, label: Literal['train', 'test', 'valid']
    ):
        if self.async_mode:
            self.executor.submit(self.saver.save_all_metric_by_class, class_metric_all_score, label)
        else:
            self.saver.save_all_metric_by_class(class_metric_all_score, label)
