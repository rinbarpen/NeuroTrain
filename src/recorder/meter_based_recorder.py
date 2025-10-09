"""
基于Meter的简化版指标记录器
结合了Meter的简洁性和MetricRecorder的多类别支持
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from collections import defaultdict

from .meter import Meter, _meter_manager
from .data_saver import DataSaver
from src.utils.typed import *
from src.metrics import get_metric_fns, many_metrics


class MeterBasedRecorder:
    """
    基于Meter的简化版指标记录器
    
    主要改进：
    1. 使用Meter管理单个指标，简化数据结构
    2. 支持多类别指标计算
    3. 保持与现有可视化系统的兼容性
    4. 提供更简洁的API
    """
    
    def __init__(
        self,
        output_dir: FilePath,
        class_labels: ClassLabelsList,
        metric_labels: MetricLabelsList,
        *,
        logger=None,
        saver: DataSaver,
        prefix: str = ""
    ):
        self.logger = logger or logging.getLogger()
        self.saver = saver
        self.output_dir = Path(output_dir)
        self.class_labels = class_labels
        self.metric_labels = metric_labels
        self.prefix = prefix
        
        # 为每个指标-类别组合创建Meter
        self.batch_meters: Dict[str, Dict[str, Meter]] = {}
        self.epoch_meters: Dict[str, Dict[str, Meter]] = {}
        
        # 初始化Meter
        self._init_meters()
        
        # 创建输出目录
        for class_label in class_labels:
            class_dir = self.output_dir / class_label
            class_dir.mkdir(parents=True, exist_ok=True)
    
    def _init_meters(self):
        """初始化所有需要的Meter"""
        for metric in self.metric_labels:
            self.batch_meters[metric] = {}
            self.epoch_meters[metric] = {}
            
            for class_label in self.class_labels:
                # 批次级别的Meter
                batch_name = f"{self.prefix}batch_{metric}_{class_label}"
                self.batch_meters[metric][class_label] = Meter(batch_name)
                
                # Epoch级别的Meter
                epoch_name = f"{self.prefix}epoch_{metric}_{class_label}"
                self.epoch_meters[metric][class_label] = Meter(epoch_name)
    
    def finish_one_batch(self, targets: np.ndarray, outputs: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        处理一个批次的数据
        
        Args:
            targets: 真实标签 shape: (batch_size, num_classes)
            outputs: 模型输出 shape: (batch_size, num_classes)
            
        Returns:
            当前批次的指标结果 {metric: {class: score}}
        """
        # 获取指标函数
        metric_fns = get_metric_fns(self.metric_labels)
        
        # 计算指标
        metrics = many_metrics(
            metric_fns, 
            targets, 
            outputs, 
            class_split=True,
            class_axis=1
        )
        
        # 创建名称映射
        name_mapping = {}
        for i, metric_fn in enumerate(metric_fns):
            actual_name = getattr(metric_fn, '__name__', str(metric_fn))
            expected_name = self.metric_labels[i]
            name_mapping[expected_name] = actual_name
        
        # 更新Meter并返回结果
        result = {}
        for metric_label in self.metric_labels:
            result[metric_label] = {}
            actual_name = name_mapping[metric_label]
            metric_scores = metrics[actual_name]
            
            for j, class_label in enumerate(self.class_labels):
                score = metric_scores[j]
                result[metric_label][class_label] = score
                
                # 更新批次Meter
                self.batch_meters[metric_label][class_label].update(score)
        
        return result
    
    def finish_one_epoch(self) -> Dict[str, Dict[str, float]]:
        """
        完成一个epoch的处理
        
        Returns:
            当前epoch的平均指标结果 {metric: {class: avg_score}}
        """
        result = {}
        
        for metric_label in self.metric_labels:
            result[metric_label] = {}
            
            for class_label in self.class_labels:
                # 获取批次平均值
                batch_meter = self.batch_meters[metric_label][class_label]
                avg_score = batch_meter.avg
                
                # 更新epoch Meter
                self.epoch_meters[metric_label][class_label].update(avg_score)
                result[metric_label][class_label] = avg_score
                
                # 重置批次Meter
                batch_meter.reset()
        
        return result
    
    def get_epoch_scores(self) -> MetricClassManyScoreDict:
        """
        获取所有epoch的分数，用于兼容现有的可视化系统
        
        Returns:
            格式化的分数字典 {metric: {class: [scores]}}
        """
        result = {}
        
        for metric_label in self.metric_labels:
            result[metric_label] = {}
            
            for class_label in self.class_labels:
                epoch_meter = self.epoch_meters[metric_label][class_label]
                result[metric_label][class_label] = epoch_meter.vals.tolist()
        
        return result
    
    def record_epochs(self, epoch: int, n_epochs: int):
        """
        记录epoch级别的指标并生成可视化
        
        Args:
            epoch: 当前epoch
            n_epochs: 总epoch数
        """
        # 延迟导入以避免循环导入
        from src.visualizer.painter import Plot
        from .metric_recorder import ScoreAggregator
        
        # 获取epoch分数用于可视化
        epoch_scores = self.get_epoch_scores()
        
        # 使用现有的ScoreAggregator进行数据处理和可视化
        scores_agg = ScoreAggregator(epoch_scores)
        
        # 保存数据
        self.saver.save_all_metric_by_class(scores_agg.cmm)
        self.saver.save_mean_metric_by_class(scores_agg.cm1_mean)
        self.saver.save_std_metric_by_class(scores_agg.cm1_std)
        self.saver.save_mean_metric(scores_agg.ml1_mean)
        self.saver.save_std_metric(scores_agg.ml1_std)
        
        # 生成可视化图表
        self._generate_visualizations(epoch, n_epochs, scores_agg)
    
    def _generate_visualizations(self, epoch: int, n_epochs: int, scores_agg):
        """生成各种可视化图表"""
        from src.visualizer.painter import Plot
        
        # 1. 每个类别的指标曲线
        epoch_metrics_image = self.output_dir / "epoch_metrics_curve_per_classes.png"
        n = len(self.metric_labels)
        if n > 4:
            nrows, ncols = (n + 2) // 3, 3
        elif n == 4:
            nrows, ncols = 2, 2
        else:
            nrows, ncols = 1, n

        plot = Plot(nrows, ncols)
        epoch_scores = self.get_epoch_scores()
        for metric in self.metric_labels:
            plot.subplot().many_epoch_metrics_by_class(
                epoch, 
                n_epochs,
                epoch_scores[metric],
                self.class_labels,
                title=metric,
            ).complete()
        plot.save(epoch_metrics_image)
        
        # 2. 全局所有metric
        epoch_metrics_image = self.output_dir / "epoch_metrics_curve.png"
        plot = Plot(1, 1)
        plot.subplot().many_epoch_metrics(
            epoch, 
            n_epochs, 
            scores_agg.m2_mean, 
            metric_labels=self.metric_labels, 
            title="All Metrics"
        ).complete()
        plot.save(epoch_metrics_image)
        
        # 3. 每个类别的指标
        for label in self.class_labels:
            epoch_metric_image = self.output_dir / label / "metrics.png"
            plot = Plot(1, 1)
            plot.subplot().many_epoch_metrics(
                epoch, n_epochs, scores_agg.cmm[label], 
                metric_labels=self.metric_labels, title=label
            ).complete()
            plot.save(epoch_metric_image)
        
        # 4. 每个类别每个metric的详细图表
        for metric in self.metric_labels:
            for label in self.class_labels:
                epoch_meter = self.epoch_meters[metric][label]
                m_scores = epoch_meter.vals
                
                epoch_metric_image = self.output_dir / label / f"{metric}.png"
                plot = Plot(1, 1)
                plot.subplot().epoch_metrics(
                    epoch, 
                    n_epochs, 
                    m_scores, 
                    label, 
                    title=f"Epoch-{label}-{metric}"
                ).complete()
                plot.save(epoch_metric_image)
    
    def reset(self):
        """重置所有Meter"""
        for metric_meters in self.batch_meters.values():
            for meter in metric_meters.values():
                meter.reset()
        
        for metric_meters in self.epoch_meters.values():
            for meter in metric_meters.values():
                meter.reset()
    
    def get_current_metrics(self) -> Dict[str, float]:
        """
        获取当前的平均指标（跨所有类别）
        
        Returns:
            {metric_name: avg_value}
        """
        result = {}
        
        for metric_label in self.metric_labels:
            scores = []
            for class_label in self.class_labels:
                batch_meter = self.batch_meters[metric_label][class_label]
                if batch_meter.count > 0:
                    scores.append(batch_meter.avg)
            
            if scores:
                result[metric_label] = np.mean(scores)
            else:
                result[metric_label] = 0.0
        
        return result
    
    def save_meters(self, suffix: str = ""):
        """
        保存所有Meter的数据
        
        Args:
            suffix: 文件名后缀
        """
        for metric_label in self.metric_labels:
            for class_label in self.class_labels:
                # 保存批次数据
                batch_meter = self.batch_meters[metric_label][class_label]
                batch_filename = self.output_dir / f"batch_{metric_label}_{class_label}{suffix}"
                batch_meter.save(batch_filename)
                
                # 保存epoch数据
                epoch_meter = self.epoch_meters[metric_label][class_label]
                epoch_filename = self.output_dir / f"epoch_{metric_label}_{class_label}{suffix}"
                epoch_meter.save(epoch_filename)