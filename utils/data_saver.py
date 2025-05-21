import numpy as np
import pandas as pd
import fastparquet
from concurrent.futures import ThreadPoolExecutor
from typing import Literal
from pathlib import Path

from utils.typed import (ClassLabelManyScoreDict, ClassLabelOneScoreDict, ClassMetricOneScoreDict, MetricAfterDict, MetricClassManyScoreDict, MetricClassOneScoreDict, MetricLabelOneScoreDict, ClassMetricManyScoreDict)

# sync mode
class _DataSaver:
    def __init__(self, base_dir: Path):
        base_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir = base_dir

    def save_loss(self, losses: list[np.ndarray], label: Literal["train", "valid"]):
        csv_filename, parquet_filename = self._get_filenames(f"{label}_loss")

        df_prev = self._prev_df(csv_filename, parquet_filename)

        df = pd.DataFrame({"loss": losses})
        df = pd.concat([df_prev, df], ignore_index=True)

        self._save_dataframe(df, csv_filename, parquet_filename)

    def save_mean_metric(self, metric_mean_score: MetricLabelOneScoreDict):
        csv_filename, parquet_filename = self._get_filenames("mean_metric")

        df = pd.DataFrame({k: [v] for k, v in metric_mean_score.items()})
        self._save_dataframe(df, csv_filename, parquet_filename)
    def save_std_metric(self, metric_std_score: MetricLabelOneScoreDict):
        csv_filename, parquet_filename = self._get_filenames("std_metric")

        df = pd.DataFrame({k: [v] for k, v in metric_std_score.items()})
        self._save_dataframe(df, csv_filename, parquet_filename)

    # for every class
    def save_mean_metric_by_class(
        self, class_metric_mean_score: ClassMetricOneScoreDict
    ):
        for class_label, metric_mean in class_metric_mean_score.items():
            csv_filename, parquet_filename = self._get_filenames(
                f"{class_label}/mean_metric"
            )

            df = pd.DataFrame({k: [v] for k, v in metric_mean.items()})
            self._save_dataframe(df, csv_filename, parquet_filename)
    def save_std_metric_by_class(
        self, class_metric_std_score: ClassMetricOneScoreDict
    ):
        for class_label, metric_std in class_metric_std_score.items():
            csv_filename, parquet_filename = self._get_filenames(
                f"{class_label}/std_metric"
            )

            df = pd.DataFrame({k: [v] for k, v in metric_std.items()})
            self._save_dataframe(df, csv_filename, parquet_filename)

    def save_all_metric_by_class(self, class_metric_all_score: ClassMetricManyScoreDict):
        for class_label, metric_all in class_metric_all_score.items():
            csv_filename, parquet_filename = self._get_filenames(
                f"{class_label}/all_metric"
            )

            df_prev = self._prev_df(csv_filename, parquet_filename)

            df = pd.DataFrame(metric_all)
            df = pd.concat([df_prev, df], ignore_index=True)

            self._save_dataframe(df, csv_filename, parquet_filename)
    def save_argmaxmin(self, metric_after_dict: MetricAfterDict):
        csv_filename, parquet_filename = self._get_filenames("argmaxmin_class")

        df_argmax = pd.DataFrame(metric_after_dict["argmax"])
        df_argmin = pd.DataFrame(metric_after_dict["argmin"])
        df = pd.concat([df_argmax, df_argmin], ignore_index=True)

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
        return csv_filename, parquet_filename

# async mode
class DataSaver:
    def __init__(self, base_dir: Path, num_threads: int = 1, *, async_mode=True):
        self.executor = ThreadPoolExecutor(num_threads)
        self.saver = _DataSaver(base_dir)
        self.async_mode = async_mode

    def save_train_loss(self, losses: list[np.ndarray]):
        if self.async_mode:
            self.executor.submit(self.saver.save_loss, losses, 'train')
        else:
            self.saver.save_loss(losses, 'train')
    def save_valid_loss(self, losses: list[np.ndarray]):
        if self.async_mode:
            self.executor.submit(self.saver.save_loss, losses, 'valid')
        else:
            self.saver.save_loss(losses, 'valid')

    def save_mean_metric(self, metric_mean_score: MetricLabelOneScoreDict):
        if self.async_mode:
            self.executor.submit(self.saver.save_mean_metric, metric_mean_score)
        else:
            self.saver.save_mean_metric(metric_mean_score)
    def save_std_metric(self, metric_std_score: MetricLabelOneScoreDict):
        if self.async_mode:
            self.executor.submit(self.saver.save_std_metric, metric_std_score)
        else:
            self.saver.save_std_metric(metric_std_score)

    # for every class
    def save_mean_metric_by_class(
        self, class_metric_mean_score: ClassMetricOneScoreDict
    ):
        if self.async_mode:
            self.executor.submit(
                self.saver.save_mean_metric_by_class, class_metric_mean_score
            )
        else:
            self.saver.save_mean_metric_by_class(class_metric_mean_score)
    def save_std_metric_by_class(
        self, class_metric_std_score: ClassMetricOneScoreDict
    ):
        if self.async_mode:
            self.executor.submit(
                self.saver.save_std_metric_by_class, class_metric_std_score
            )
        else:
            self.saver.save_std_metric_by_class(class_metric_std_score)

    def save_all_metric_by_class(self, class_metric_all_score: ClassMetricManyScoreDict):
        if self.async_mode:
            self.executor.submit(
                self.saver.save_all_metric_by_class, class_metric_all_score
            )
        else:
            self.saver.save_all_metric_by_class(class_metric_all_score)

    def save_argmaxmin(self, metric_after_dict: MetricAfterDict):
        if self.async_mode:
            self.executor.submit(
                self.saver.save_argmaxmin, metric_after_dict)
        else:
            self.saver.save_argmaxmin(metric_after_dict)
