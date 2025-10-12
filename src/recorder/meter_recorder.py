"""
基于Meter的完整功能指标记录器
完全替代MetricRecorder，充分利用Meter的功能
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Callable, Union, Optional

from .meter import Meter
from .data_saver import DataSaver
from src.utils.typed import *
from src.metrics import get_metric_fns, many_metrics


class MeterRecorder:
    """
    基于Meter的完整功能指标记录器
    
    主要特性：
    1. 完全基于Meter管理所有指标数据
    2. 支持MetricRecorder的所有功能
    3. 不使用ScoreAggregator，直接使用Meter的统计功能
    4. 提供更简洁和高效的API
    5. 支持批次和epoch级别的数据记录
    """
    
    def __init__(
        self,
        class_labels: ClassLabelsList,
        metric_labels: MetricLabelsList,
        *,
        logger=None,
        saver: DataSaver,
        prefix: str = ""
    ):
        self.logger = logger or logging.getLogger()
        self.saver = saver
        self.class_labels = class_labels
        self.metric_labels = metric_labels
        self.prefix = prefix
        
        # 为每个指标-类别组合创建Meter
        # batch_meters: 存储每个批次的指标值
        # epoch_meters: 存储每个epoch的平均指标值
        self.batch_meters: Dict[str, Dict[str, Meter]] = {}
        self.epoch_meters: Dict[str, Dict[str, Meter]] = {}
        
        # 初始化Meter
        self._init_meters()
    
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
    
    def _compute_mc1(self, func: Callable[[np.ndarray], float]) -> MetricClassOneScoreDict:
        """
        计算MetricClassOneScoreDict (mean or std)
        使用Meter的统计功能替代ScoreAggregator
        """
        result: MetricClassOneScoreDict = {}
        for metric_label in self.metric_labels:
            result[metric_label] = {}
            for class_label in self.class_labels:
                meter = self.epoch_meters[metric_label][class_label]
                if meter.count > 0:
                    if func == np.mean:
                        result[metric_label][class_label] = FLOAT(meter.avg)
                    elif func == np.std:
                        result[metric_label][class_label] = FLOAT(np.std(meter.vals))
                    else:
                        result[metric_label][class_label] = FLOAT(func(meter.vals))
                else:
                    result[metric_label][class_label] = FLOAT(0.0)
        return result
    
    def _compute_cmm(self) -> ClassMetricManyScoreDict:
        """
        计算ClassMetricManyScoreDict
        使用Meter的vals属性获取历史数据
        """
        result: ClassMetricManyScoreDict = {}
        for metric_label in self.metric_labels:
            for class_label in self.class_labels:
                if class_label not in result:
                    result[class_label] = {}
                meter = self.epoch_meters[metric_label][class_label]
                result[class_label][metric_label] = meter.vals.tolist()
        return result
    
    def _compute_cm1(self, func: Callable[[np.ndarray], float]) -> ClassMetricOneScoreDict:
        """
        计算ClassMetricOneScoreDict (mean or std)
        从epoch_meters直接计算
        """
        result: ClassMetricOneScoreDict = {}
        for class_label in self.class_labels:
            result[class_label] = {}
            for metric_label in self.metric_labels:
                meter = self.epoch_meters[metric_label][class_label]
                if meter.count > 0:
                    if func == np.mean:
                        result[class_label][metric_label] = FLOAT(meter.avg)
                    elif func == np.std:
                        result[class_label][metric_label] = FLOAT(np.std(meter.vals))
                    else:
                        result[class_label][metric_label] = FLOAT(func(meter.vals))
                else:
                    result[class_label][metric_label] = FLOAT(0.0)
        return result
    
    def _compute_ml1(self, func: Callable[[np.ndarray], float]) -> MetricLabelOneScoreDict:
        """
        计算MetricLabelOneScoreDict (mean or std)
        跨所有类别聚合每个指标的分数
        """
        result: MetricLabelOneScoreDict = {}
        for metric_label in self.metric_labels:
            all_scores = []
            for class_label in self.class_labels:
                meter = self.epoch_meters[metric_label][class_label]
                if meter.count > 0:
                    all_scores.extend(meter.vals.tolist())
            
            if all_scores:
                result[metric_label] = FLOAT(func(np.array(all_scores)))
            else:
                result[metric_label] = FLOAT(0.0)
        return result
    
    def _compute_m2(self) -> MetricLabelManyScoreDict:
        """
        计算MetricLabelManyScoreDict
        按epoch聚合所有类别的平均分数
        """
        result: MetricLabelManyScoreDict = {}
        
        # 获取最大epoch数
        max_epochs = 0
        for metric_label in self.metric_labels:
            for class_label in self.class_labels:
                meter = self.epoch_meters[metric_label][class_label]
                max_epochs = max(max_epochs, meter.count)
        
        for metric_label in self.metric_labels:
            result[metric_label] = []
            for epoch_idx in range(max_epochs):
                epoch_scores = []
                for class_label in self.class_labels:
                    meter = self.epoch_meters[metric_label][class_label]
                    if epoch_idx < meter.count:
                        epoch_scores.append(meter.vals[epoch_idx])
                
                if epoch_scores:
                    result[metric_label].append(FLOAT(np.mean(epoch_scores)))
                else:
                    result[metric_label].append(FLOAT(0.0))
        
        return result
    
    def _compute_m2_std(self) -> MetricLabelManyScoreDict:
        """
        计算MetricLabelManyScoreDict的标准差
        按epoch计算所有类别的标准差
        """
        result: MetricLabelManyScoreDict = {}
        
        # 获取最大epoch数
        max_epochs = 0
        for metric_label in self.metric_labels:
            for class_label in self.class_labels:
                meter = self.epoch_meters[metric_label][class_label]
                max_epochs = max(max_epochs, meter.count)
        
        for metric_label in self.metric_labels:
            result[metric_label] = []
            for epoch_idx in range(max_epochs):
                epoch_scores = []
                for class_label in self.class_labels:
                    meter = self.epoch_meters[metric_label][class_label]
                    if epoch_idx < meter.count:
                        epoch_scores.append(meter.vals[epoch_idx])
                
                if len(epoch_scores) > 1:
                    result[metric_label].append(FLOAT(np.std(epoch_scores)))
                else:
                    result[metric_label].append(FLOAT(0.0))
        
        return result
    
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
                # metric_scores 可能是 numpy array 或 list
                if isinstance(metric_scores, (list, np.ndarray)):
                    score = float(metric_scores[j])
                else:
                    score = float(metric_scores)
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
                avg_score = batch_meter.avg if batch_meter.count > 0 else 0.0
                
                # 更新epoch Meter
                self.epoch_meters[metric_label][class_label].update(avg_score)
                result[metric_label][class_label] = avg_score
                
                # 重置批次Meter
                batch_meter.reset()
        
        return result
    
    def _compute_cmm_from_batch(self) -> ClassMetricManyScoreDict:
        """从batch_meters计算ClassMetricManyScoreDict"""
        result: ClassMetricManyScoreDict = {}
        for class_label in self.class_labels:
            result[class_label] = {}
            for metric_label in self.metric_labels:
                meter = self.batch_meters[metric_label][class_label]
                if meter.count > 0:
                    result[class_label][metric_label] = meter.vals.tolist()
                else:
                    result[class_label][metric_label] = []
        return result
    
    def _compute_cm1_from_batch(self, func: Callable[[np.ndarray], float]) -> ClassMetricOneScoreDict:
        """从batch_meters计算ClassMetricOneScoreDict"""
        result: ClassMetricOneScoreDict = {}
        for class_label in self.class_labels:
            result[class_label] = {}
            for metric_label in self.metric_labels:
                meter = self.batch_meters[metric_label][class_label]
                if meter.count > 0:
                    if func == np.mean:
                        result[class_label][metric_label] = FLOAT(meter.avg)
                    elif func == np.std:
                        result[class_label][metric_label] = FLOAT(np.std(meter.vals))
                    else:
                        result[class_label][metric_label] = FLOAT(func(meter.vals))
                else:
                    result[class_label][metric_label] = FLOAT(0.0)
        return result
    
    def _compute_ml1_from_batch(self, func: Callable[[np.ndarray], float]) -> MetricLabelOneScoreDict:
        """从batch_meters计算MetricLabelOneScoreDict"""
        result: MetricLabelOneScoreDict = {}
        for metric_label in self.metric_labels:
            all_scores = []
            for class_label in self.class_labels:
                meter = self.batch_meters[metric_label][class_label]
                if meter.count > 0:
                    all_scores.extend(meter.vals.tolist())
            
            if all_scores:
                result[metric_label] = FLOAT(func(np.array(all_scores)))
            else:
                result[metric_label] = FLOAT(0.0)
        return result
    
    def record_batches(self, output_dir: Optional[Union[str, Path]] = None, filenames: dict={
        'all_metric_by_class': '{class_label}/all_metric.csv',
        'mean_metric_by_class': '{class_label}/mean_metric.csv',
        'std_metric_by_class': '{class_label}/std_metric.csv',
        'mean_metric': 'mean_metric.csv',
        'std_metric': 'std_metric.csv',
    }):
        """
        记录批次级别的指标并生成可视化
        使用Meter的功能替代ScoreAggregator
        
        Args:
            output_dir: 输出目录，用于保存可视化图表
            filenames: CSV文件名字典
        """
        # 计算统计数据 - 使用batch_meters的当前状态
        cmm = self._compute_cmm_from_batch()
        cm1_mean = self._compute_cm1_from_batch(np.mean)
        cm1_std = self._compute_cm1_from_batch(np.std)
        ml1_mean = self._compute_ml1_from_batch(np.mean)
        ml1_std = self._compute_ml1_from_batch(np.std)
        
        # 保存数据（传入完整路径）
        self.saver.save_all_metric_by_class(cmm, str(filenames['all_metric_by_class']))
        self.saver.save_mean_metric_by_class(cm1_mean, str(filenames['mean_metric_by_class']))
        self.saver.save_std_metric_by_class(cm1_std, str(filenames['std_metric_by_class']))
        self.saver.save_mean_metric(ml1_mean, str(filenames['mean_metric']))
        self.saver.save_std_metric(ml1_std, str(filenames['std_metric']))
        
        # 生成可视化图表
        if output_dir is not None:
            self._generate_batch_visualizations(cm1_mean, Path(output_dir))
    
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
    
    def record_epochs(self, epoch: int, n_epochs: int, output_dir: Optional[Union[str, Path]] = None, filenames: dict={
        'all_metric_by_class': '{class_label}/all_metric.csv',
        'mean_metric_by_class': '{class_label}/mean_metric.csv',
        'std_metric_by_class': '{class_label}/std_metric.csv',
        'mean_metric': 'mean_metric.csv',
        'std_metric': 'std_metric.csv',
    }):
        """
        记录epoch级别的指标并生成可视化
        
        Args:
            epoch: 当前epoch
            n_epochs: 总epoch数
            output_dir: 输出目录，用于保存可视化图表
            filenames: CSV文件名字典
        """
        # 计算统计数据
        cmm = self._compute_cmm()
        cm1_mean = self._compute_cm1(np.mean)
        cm1_std = self._compute_cm1(np.std)
        ml1_mean = self._compute_ml1(np.mean)
        ml1_std = self._compute_ml1(np.std)
        m2_mean = self._compute_m2()
        
        # 保存数据（传入完整路径）
        self.saver.save_all_metric_by_class(cmm, str(filenames['all_metric_by_class']))
        self.saver.save_mean_metric_by_class(cm1_mean, str(filenames['mean_metric_by_class']))
        self.saver.save_std_metric_by_class(cm1_std, str(filenames['std_metric_by_class']))
        self.saver.save_mean_metric(ml1_mean, str(filenames['mean_metric']))
        self.saver.save_std_metric(ml1_std, str(filenames['std_metric']))
        
        # 生成可视化图表
        if output_dir is not None:
            self._generate_epoch_visualizations(epoch, n_epochs, cmm, cm1_mean, m2_mean, Path(output_dir))
    
    def _generate_batch_visualizations(self, cm1_mean: ClassMetricOneScoreDict, output_dir: Path):
        """生成批次级别的可视化图表"""
        from src.visualizer.painter import Plot
        
        # 1. 所有类别的平均指标
        mean_metrics_image = output_dir / "mean_metrics_per_classes.png"
        Plot(1, 1).subplot().many_metrics(cm1_mean).complete().save(mean_metrics_image)
        
        # 2. 每个类别的平均指标
        for class_label in self.class_labels:
            if class_label in cm1_mean:
                mean_image = output_dir / class_label / "mean_metric.png"
                mean_image.parent.mkdir(parents=True, exist_ok=True)
                Plot(1, 1).subplot().metrics(cm1_mean[class_label], class_label).complete().save(mean_image)
        
        # 3. 全局平均指标
        ml1_mean = self._compute_ml1_from_batch(np.mean)
        mean_metrics_image = output_dir / "mean_metrics.png"
        Plot(1, 1).subplot().metrics(ml1_mean).complete().save(mean_metrics_image)
    
    def _generate_epoch_visualizations(self, epoch: int, n_epochs: int, cmm: ClassMetricManyScoreDict, cm1_mean: ClassMetricOneScoreDict, m2_mean: MetricLabelManyScoreDict, output_dir: Path):
        """生成epoch级别的可视化图表"""
        from src.visualizer.painter import Plot
        
        # 检查是否有数据
        if not m2_mean or len(m2_mean) == 0:
            self.logger.warning(f"No data available for visualization at epoch {epoch}")
            return
        
        # 1. 每个类别的指标曲线
        epoch_metrics_image = output_dir / "epoch_metrics_curve_per_classes.png"
        n = len(self.metric_labels)
        if n > 4:
            nrows, ncols = (n + 2) // 3, 3
        elif n == 4:
            nrows, ncols = 2, 2
        else:
            nrows, ncols = 1, n

        plot = Plot(nrows, ncols)
        epoch_scores = self.get_epoch_scores()
        
        # 检查epoch_scores是否有数据
        if not epoch_scores:
            self.logger.warning(f"No epoch scores available for visualization at epoch {epoch}")
            return
            
        for metric in self.metric_labels:
            if metric in epoch_scores and epoch_scores[metric]:
                plot.subplot().many_epoch_metrics_by_class(
                    epoch, 
                    n_epochs,
                    epoch_scores[metric],
                    self.class_labels,
                    title=metric,
                ).complete()
            else:
                self.logger.warning(f"No data for metric {metric} at epoch {epoch}")
        plot.save(str(epoch_metrics_image))
        
        # 2. 全局所有metric（带标准差）
        # 计算标准差
        m2_std = self._compute_m2_std()
        
        epoch_metrics_image = output_dir / "epoch_metrics_curve.png"
        plot = Plot(1, 1)
        plot.subplot().many_epoch_metrics(
            epoch, 
            n_epochs, 
            m2_mean, 
            metric_labels=self.metric_labels, 
            title="All Metrics",
            std_scores=m2_std
        ).complete()
        plot.save(str(epoch_metrics_image))
        
        # 3. 每个类别的指标
        for label in self.class_labels:
            if label in cmm:
                epoch_metric_image = output_dir / label / "metrics.png"
                epoch_metric_image.parent.mkdir(parents=True, exist_ok=True)
                plot = Plot(1, 1)
                plot.subplot().many_epoch_metrics(
                    epoch, n_epochs, cmm[label], 
                    metric_labels=self.metric_labels, title=label
                ).complete()
                plot.save(epoch_metric_image)
            else:
                self.logger.warning(f"No data for class {label} at epoch {epoch}")
        
        # 4. 每个类别每个metric的详细图表
        for metric in self.metric_labels:
            for label in self.class_labels:
                if metric in self.epoch_meters and label in self.epoch_meters[metric]:
                    epoch_meter = self.epoch_meters[metric][label]
                    m_scores = epoch_meter.vals
                    
                    if len(m_scores) > 0:  # 检查是否有数据
                        epoch_metric_image = output_dir / label / f"{metric}.png"
                        epoch_metric_image.parent.mkdir(parents=True, exist_ok=True)
                        plot = Plot(1, 1)
                        plot.subplot().epoch_metrics(
                            epoch, 
                            n_epochs, 
                            m_scores, 
                            label, 
                            title=f"Epoch-{label}-{metric}"
                        ).complete()
                        plot.save(epoch_metric_image)
                    else:
                        self.logger.warning(f"No data for {metric}-{label} at epoch {epoch}")
                else:
                    self.logger.warning(f"Meter not found for {metric}-{label}")
    
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
    
    def get_epoch_metrics(self) -> Dict[str, float]:
        """
        获取epoch级别的平均指标（跨所有类别）
        
        Returns:
            {metric_name: avg_value}
        """
        result = {}
        
        for metric_label in self.metric_labels:
            scores = []
            for class_label in self.class_labels:
                epoch_meter = self.epoch_meters[metric_label][class_label]
                if epoch_meter.count > 0:
                    scores.append(epoch_meter.avg)
            
            if scores:
                result[metric_label] = np.mean(scores)
            else:
                result[metric_label] = 0.0
        
        return result
    
    def get_meter_stats(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        获取所有Meter的统计信息
        
        Returns:
            {metric: {class: {stat_name: value}}}
        """
        result = {}
        
        for metric_label in self.metric_labels:
            result[metric_label] = {}
            for class_label in self.class_labels:
                batch_meter = self.batch_meters[metric_label][class_label]
                epoch_meter = self.epoch_meters[metric_label][class_label]
                
                result[metric_label][class_label] = {
                    'batch_count': batch_meter.count,
                    'batch_avg': float(batch_meter.avg),
                    'batch_sum': float(batch_meter.sum),
                    'epoch_count': epoch_meter.count,
                    'epoch_avg': float(epoch_meter.avg),
                    'epoch_sum': float(epoch_meter.sum),
                }
        
        return result
    
    def save_meters(self, output_dir: Union[str, Path], suffix: str = ""):
        """
        保存所有Meter的数据
        
        Args:
            output_dir: 输出目录
            suffix: 文件名后缀
        """
        output_path = Path(output_dir)
        for metric_label in self.metric_labels:
            for class_label in self.class_labels:
                # 保存批次数据
                batch_meter = self.batch_meters[metric_label][class_label]
                batch_filename = output_path / f"batch_{metric_label}_{class_label}{suffix}"
                batch_meter.save(batch_filename)
                
                # 保存epoch数据
                epoch_meter = self.epoch_meters[metric_label][class_label]
                epoch_filename = output_path / f"epoch_{metric_label}_{class_label}{suffix}"
                epoch_meter.save(epoch_filename)
    
    def paint_meters(self, output_dir: Union[str, Path], suffix: str = ""):
        """
        使用Meter的paint方法生成可视化图表
        
        Args:
            output_dir: 输出目录
            suffix: 文件名后缀
        """
        output_path = Path(output_dir)
        for metric_label in self.metric_labels:
            for class_label in self.class_labels:
                # 绘制批次数据图表
                batch_meter = self.batch_meters[metric_label][class_label]
                if batch_meter.count > 0:
                    batch_image = output_path / f"batch_{metric_label}_{class_label}{suffix}.png"
                    batch_meter.paint(
                        batch_image,
                        title=f"Batch {metric_label} - {class_label}",
                        xlabel="Batch",
                        ylabel=metric_label
                    )
                
                # 绘制epoch数据图表
                epoch_meter = self.epoch_meters[metric_label][class_label]
                if epoch_meter.count > 0:
                    epoch_image = output_path / f"epoch_{metric_label}_{class_label}{suffix}.png"
                    epoch_meter.paint(
                        epoch_image,
                        title=f"Epoch {metric_label} - {class_label}",
                        xlabel="Epoch",
                        ylabel=metric_label
                    )
    
    def export_data(self, output_dir: Union[str, Path], format: str = "csv") -> Dict[str, Dict[str, str]]:
        """
        导出所有Meter的数据
        
        Args:
            output_dir: 输出目录
            format: 导出格式 ("csv", "parquet", "both")
            
        Returns:
            导出的文件路径字典
        """
        output_path = Path(output_dir)
        exported_files = {}
        
        for metric_label in self.metric_labels:
            exported_files[metric_label] = {}
            for class_label in self.class_labels:
                exported_files[metric_label][class_label] = {}
                
                # 导出批次数据
                batch_meter = self.batch_meters[metric_label][class_label]
                if batch_meter.count > 0:
                    batch_filename = output_path / f"batch_{metric_label}_{class_label}"
                    batch_meter.save(batch_filename, to_csv=(format in ["csv", "both"]), to_parquet=(format in ["parquet", "both"]))
                    exported_files[metric_label][class_label]["batch"] = str(batch_filename)
                
                # 导出epoch数据
                epoch_meter = self.epoch_meters[metric_label][class_label]
                if epoch_meter.count > 0:
                    epoch_filename = output_path / f"epoch_{metric_label}_{class_label}"
                    epoch_meter.save(epoch_filename, to_csv=(format in ["csv", "both"]), to_parquet=(format in ["parquet", "both"]))
                    exported_files[metric_label][class_label]["epoch"] = str(epoch_filename)
        
        return exported_files