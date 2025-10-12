import numpy as np
import pandas as pd
import fastparquet
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Sequence, Optional, Dict, Any
from pathlib import Path

from src.utils.typed import (
    ClassMetricOneScoreDict, 
    MetricAfterDict, 
    MetricLabelOneScoreDict, 
    ClassMetricManyScoreDict
)


class DataSaver:
    """
    统一的数据保存器
    
    特性：
    1. 支持同步和异步模式
    2. 支持多种数据类型保存
    3. 支持CSV和Parquet格式
    4. 内置数据验证和过滤
    5. 支持Meter的save_as方法
    6. 线程安全
    """
    
    def __new__(cls, *args, **kwargs):
        """单例模式，确保全局只有一个DataSaver实例"""
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, num_threads: int = 1, *, async_mode: bool = True):
        """
        初始化DataSaver
        
        Args:
            base_dir: 基础保存目录
            num_threads: 线程池大小
            async_mode: 是否使用异步模式
        """
        # 如果已经初始化且参数相同，则跳过
        if hasattr(self, '_initialized') and self.async_mode == async_mode:
            return
            
        self.async_mode = async_mode
        
        # 异步处理组件
        self._queue = queue.Queue()
        self._mapping: Dict[str, pd.DataFrame] = {}
        self._running = True
        
        # 线程池用于复杂操作
        self.executor = ThreadPoolExecutor(max_workers=num_threads) if async_mode else None
        
        # 启动队列处理线程
        self._queue_thread = threading.Thread(target=self._run_queue, daemon=True)
        self._queue_thread.start()
        
        self._initialized = True
    
    def _run_queue(self):
        """队列处理线程"""
        while self._running:
            try:
                filename, df = self._queue.get(timeout=1.0)
                if filename in self._mapping:
                    # 合并数据
                    self._mapping[filename] = pd.concat(
                        [self._mapping[filename], df], 
                        axis=1, 
                        ignore_index=True, 
                        sort=False
                    )
                else:
                    self._mapping[filename] = df
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"DataSaver queue error: {e}")
    
    def _save_dataframe(self, df: pd.DataFrame, csv_filename: str, parquet_filename: str):
        """保存DataFrame到CSV和Parquet文件"""
        try:
            # 确保目录存在
            csv_path = Path(csv_filename)
            parquet_path = Path(parquet_filename)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 检查CSV文件是否存在，如果存在则不写入header
            write_header = not csv_path.exists() or csv_path.stat().st_size == 0
            
            # 保存CSV
            df.to_csv(csv_filename, mode='a', header=write_header, index=False)
            
            # 保存Parquet（修复append问题：若不存在则写入新文件，否则append）
            try:
                if not parquet_path.exists():
                    df.to_parquet(parquet_filename, index=False)
                else:
                    # 读取旧数据再追加（兼容性最好）
                    try:
                        old_df = pd.read_parquet(parquet_filename)
                        df = pd.concat([old_df, df], ignore_index=True)
                    except Exception:
                        pass
                    df.to_parquet(parquet_filename, index=False)
            except Exception:
                # 回退到fastparquet写入
                fastparquet.write(parquet_filename, df, append=True)
            
        except Exception as e:
            print(f"Error saving dataframe: {e}")
    
    def _get_filenames(self, filename: str) -> tuple[str, str]:
        """获取CSV和Parquet文件名，保留目录结构"""
        p = Path(filename)
        csv_filename = p.with_suffix('.csv')
        parquet_filename = p.with_suffix('.parquet')
        return csv_filename.as_posix(), parquet_filename.as_posix()
    
    def _filter_valid_data(self, data: Dict[str, Any], allow_zero: bool = True) -> Dict[str, Any]:
        """过滤有效数据"""
        if not data:
            return {}
        
        filtered = {}
        for k, v in data.items():
            if allow_zero:
                # 允许零值，只过滤NaN和无穷大
                if np.isfinite(v):
                    filtered[k] = v
            else:
                # 过滤NaN、无穷大和零值
                if not (np.isnan(v) or np.isinf(v) or v == 0):
                    filtered[k] = v
        
        return filtered
    
    def _submit_or_execute(self, func, *args):
        """提交到线程池或直接执行"""
        if self.async_mode and self.executor:
            self.executor.submit(func, *args)
        else:
            func(*args)
    
    # ==================== 损失函数保存 ====================
    def save_train_loss(self, losses: np.ndarray, filename: str = "train_loss.csv"):
        """保存训练损失"""
        csv_filename, parquet_filename = self._get_filenames(filename)
        df = pd.DataFrame({"loss": losses})
        self._submit_or_execute(self._save_dataframe, df, csv_filename, parquet_filename)
    
    def save_valid_loss(self, losses: np.ndarray, filename: str = "valid_loss.csv"):
        """保存验证损失"""
        csv_filename, parquet_filename = self._get_filenames(filename)
        df = pd.DataFrame({"loss": losses})
        self._submit_or_execute(self._save_dataframe, df, csv_filename, parquet_filename)
    
    # ==================== Step 损失与学习率保存 ====================
    def save_train_step_loss(self, step_losses: np.ndarray, filename: str = "train_step_loss.csv"):
        csv_filename, parquet_filename = self._get_filenames(filename)
        df = pd.DataFrame({"loss": step_losses})
        self._submit_or_execute(self._save_dataframe, df, csv_filename, parquet_filename)
    
    def save_lr_by_epoch(self, lrs: np.ndarray, filename: str = "lr_by_epoch.csv"):
        csv_filename, parquet_filename = self._get_filenames(filename)
        df = pd.DataFrame({"lr": lrs})
        self._submit_or_execute(self._save_dataframe, df, csv_filename, parquet_filename)
    
    def save_lr(self, lrs: np.ndarray, filename: str = "lr.csv"):
        csv_filename, parquet_filename = self._get_filenames(filename)
        df = pd.DataFrame({"lr": lrs})
        self._submit_or_execute(self._save_dataframe, df, csv_filename, parquet_filename)
    
    # ==================== 指标保存 ====================
    
    def save_mean_metric(self, metric_mean_score: MetricLabelOneScoreDict, filename: str = "mean_metric.csv"):
        """保存平均指标"""
        filtered_scores = self._filter_valid_data(metric_mean_score, allow_zero=False)
        if not filtered_scores:
            return
        
        csv_filename, parquet_filename = self._get_filenames(filename)
        df = pd.DataFrame({k: [v] for k, v in filtered_scores.items()})
        self._submit_or_execute(self._save_dataframe, df, csv_filename, parquet_filename)
    
    def save_std_metric(self, metric_std_score: MetricLabelOneScoreDict, filename: str = "std_metric.csv"):
        """保存标准差指标"""
        filtered_scores = self._filter_valid_data(metric_std_score, allow_zero=True)
        if not filtered_scores:
            return
        
        csv_filename, parquet_filename = self._get_filenames(filename)
        df = pd.DataFrame({k: [v] for k, v in filtered_scores.items()})
        self._submit_or_execute(self._save_dataframe, df, csv_filename, parquet_filename)
    
    def save_mean_metric_by_class(self, class_metric_mean_score: ClassMetricOneScoreDict, filename: str = "{class_label}/mean_metric.csv"):
        """按类别保存平均指标"""
        for class_label, metric_mean in class_metric_mean_score.items():
            filtered_metric_mean = self._filter_valid_data(metric_mean, allow_zero=False)
            if not filtered_metric_mean:
                continue
            
            # 处理{class_label}占位符
            filename = filename.format(class_label=class_label) if "{class_label}" in filename else filename
            csv_filename, parquet_filename = self._get_filenames(filename)
            df = pd.DataFrame({k: [v] for k, v in filtered_metric_mean.items()})
            self._submit_or_execute(self._save_dataframe, df, csv_filename, parquet_filename)
    
    def save_std_metric_by_class(self, class_metric_std_score: ClassMetricOneScoreDict, filename: str = "{class_label}/std_metric.csv"):
        """按类别保存标准差指标"""
        for class_label, metric_std in class_metric_std_score.items():
            filtered_metric_std = self._filter_valid_data(metric_std, allow_zero=True)
            if not filtered_metric_std:
                continue
            
            # 处理{class_label}占位符
            filename = filename.format(class_label=class_label) if "{class_label}" in filename else filename
            csv_filename, parquet_filename = self._get_filenames(filename)
            df = pd.DataFrame({k: [v] for k, v in filtered_metric_std.items()})
            self._submit_or_execute(self._save_dataframe, df, csv_filename, parquet_filename)
    
    def save_all_metric_by_class(self, class_metric_all_score: ClassMetricManyScoreDict, filename: str = "{class_label}/all_metric.csv"):
        """按类别保存所有指标"""
        for class_label, metric_all in class_metric_all_score.items():
            # 处理{class_label}占位符
            filename = filename.format(class_label=class_label) if "{class_label}" in filename else filename
            csv_filename, parquet_filename = self._get_filenames(filename)
            df = pd.DataFrame(metric_all)
            self._submit_or_execute(self._save_dataframe, df, csv_filename, parquet_filename)
    
    def save_argmaxmin(self, metric_after_dict: MetricAfterDict):
        """保存argmax和argmin数据"""
        csv_filename, parquet_filename = self._get_filenames("argmaxmin_class")
        
        df_argmax = pd.DataFrame(metric_after_dict["argmax"])
        df_argmin = pd.DataFrame(metric_after_dict["argmin"])
        df = pd.concat([df_argmax, df_argmin], ignore_index=True)
        
        self._submit_or_execute(self._save_dataframe, df, csv_filename, parquet_filename)
    
    # ==================== Meter支持方法 ====================

    @staticmethod
    def save(filename: Path, name: str, vals: Sequence[float]):
        """Meter的save_as方法调用的静态方法"""
        instance = DataSaver()
        df = pd.DataFrame({name: vals})
        instance._queue.put((filename, df))
    
    def save_to_local(self):
        """将队列中的数据保存到本地文件"""
        # 等待队列处理完成
        self._queue.join()
        
        # 保存所有数据
        for filename, df in self._mapping.items():
            try:
                path = Path(filename)
                df.to_csv(path.with_suffix('.csv'), index=False)
                df.to_parquet(path.with_suffix('.parquet'), index=False)
            except Exception as e:
                print(f"Error saving {filename}: {e}")
        
        # 清空映射
        self._mapping.clear()
    
    def complete(self):
        """完成所有保存操作"""
        # 停止队列处理
        self._running = False
        
        # 等待队列处理完成
        if self._queue_thread.is_alive():
            self._queue_thread.join(timeout=5.0)
        
        # 等待线程池完成
        if self.executor:
            self.executor.shutdown(wait=True)
        
        # 保存剩余数据
        self.save_to_local()
    
    def __del__(self):
        """析构函数，确保资源清理"""
        try:
            self.complete()
        except Exception:
            pass