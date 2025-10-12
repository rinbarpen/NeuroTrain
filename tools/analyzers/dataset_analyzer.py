# TODO: 完善数据集分析器模块
"""
数据集分析器模块

该模块提供全面的数据集分析功能，包括：
- 数据集类别分布分析
- 数据质量检查与统计
- 数据集特征分析
- 可视化与报告生成

Author: NeuroTrain Team
Date: 2024
"""

import os
import json
import uuid
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Literal, Callable, Tuple
from collections import Counter, defaultdict
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader

# 导入NeuroTrain的数据集和工具模块
try:
    from src.dataset import get_dataset, get_train_dataset, get_test_dataset, get_valid_dataset
    from src.utils import DataSaver
    from src.visualizer.painter import Plot, Font
    from src.visualizer.presets import CmapPresets, ThemePresets
except ImportError:
    print("Warning: Could not import NeuroTrain dataset and utils modules")

# 定义函数类型
LabelExtractorType = Callable[[Any], Optional[np.ndarray]]
ImageExtractorType = Callable[[Any], Optional[np.ndarray]]
LabelProcessorType = Callable[[np.ndarray], List[int]]
ImageStatsCalculatorType = Callable[[np.ndarray], Optional[Dict[str, Any]]]


class DatasetAnalyzer:
    """数据集分析器类
    
    提供全面的数据集分析功能，包括类别分布、数据质量检查、
    特征统计、可视化和报告生成等功能。
    """
    
    def __init__(self, 
                 dataset_name: str,
                 dataset_config: Optional[Dict[str, Any]] = None,
                 output_dir: str = "output/analysis",
                 report_id: Optional[str] = None,
                 logger: Optional[logging.Logger] = None,
                 label_extractor: Optional[LabelExtractorType] = None,
                 image_extractor: Optional[ImageExtractorType] = None,
                 label_processor: Optional[LabelProcessorType] = None,
                 image_stats_calculator: Optional[ImageStatsCalculatorType] = None):
        """
        初始化数据集分析器
        
        Args:
            dataset_name: 数据集名称
            dataset_config: 数据集配置字典
            output_dir: 输出目录
            report_id: 报告ID，如果为None则自动生成
            logger: 日志记录器，如果为None则创建新的
            label_extractor: 标签提取函数
            image_extractor: 图像提取函数
            label_processor: 标签处理函数
            image_stats_calculator: 图像统计计算函数
        """
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config or {}
        self.report_id = report_id or self._generate_report_id()
        
        # 创建输出目录 (必须在logger初始化之前)
        self.output_dir = Path(output_dir) / self.report_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化logger
        self.logger = logger or self._setup_logger()
        
        # 设置自定义函数或使用默认函数
        self.label_extractor = label_extractor or self._default_label_extractor
        self.image_extractor = image_extractor or self._default_image_extractor
        self.label_processor = label_processor or self._default_label_processor
        self.image_stats_calculator = image_stats_calculator or self._default_image_stats_calculator
        
        # 初始化数据存储
        self.analysis_results = {}
        self.datasets = {}
        self.statistics = {}
        
        # 设置绘图样式
        self._setup_plotting_style()
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(f"DatasetAnalyzer.{self.dataset_name}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            file_handler = logging.FileHandler(self.output_dir / "dataset_analysis.log")
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
        
        return logger
    
    def _setup_plotting_style(self):
        """设置绘图样式"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    
    def _default_label_extractor(self, sample: Any) -> Optional[np.ndarray]:
        """默认标签提取函数
        
        Args:
            sample: 数据集样本
            # 假设样本是一个元组或列表，标签是第二个元素
            # 假设标签是一个Dict，label、class
            
        Returns:
            提取的标签，如果提取失败返回None
        """
        try:
            if isinstance(sample, (tuple, list)) and len(sample) >= 2:
                label = sample[1]  # 假设标签是第二个元素
                if torch.is_tensor(label):
                    label = label.numpy()
                return label
            elif isinstance(sample, dict):
                for label_name in ["label", "class"]:
                    if label_name in sample:
                        if torch.is_tensor(sample[label_name]):
                            sample[label_name] = sample[label_name].numpy()
                        return sample[label_name]
            else:
                self.logger.warning(f"Sample format unexpected: {type(sample)}")
                return None
                
        except Exception as e:
            self.logger.warning(f"Error extracting label: {e}")
            return None
    
    def _default_image_extractor(self, sample: Any) -> Optional[np.ndarray]:
        """默认图像提取函数
        
        Args:
            sample: 数据集样本
            
        Returns:
            提取的图像，如果提取失败返回None
        """
        try:
            if isinstance(sample, (tuple, list)):
                image = sample[0]  # 假设图像是第一个元素
                
                if torch.is_tensor(image):
                    image = image.numpy()
                
                return image
            elif isinstance(sample, dict):
                for image_name in ["image", "img"]:
                    if image_name in sample:
                        image = sample[image_name]
                        if torch.is_tensor(image):
                            image = image.numpy()
                        return image
            else:
                self.logger.warning(f"Sample format unexpected: {type(sample)}")
                return None
                
        except Exception as e:
            self.logger.warning(f"Error extracting image: {e}")
            return None
    
    def _default_label_processor(self, label: np.ndarray) -> List[int]:
        """默认标签处理函数
        
        Args:
            label: 标签数组
            
        Returns:
            处理后的标签列表
        """
        processed_labels = []
        
        try:
            # 分析不同类型的标签
            if label.ndim == 0:  # 标量标签（分类）
                processed_labels.append(int(label))
            elif label.ndim >= 2:  # 分割掩码
                unique_values = np.unique(label)
                processed_labels.extend(unique_values.tolist())
            elif label.ndim == 1:  # 一维标签（多标签分类等）
                if len(label) == 1:
                    processed_labels.append(int(label[0]))
                else:
                    processed_labels.extend(label.tolist())
                    
        except Exception as e:
            self.logger.warning(f"Error processing label: {e}")
            
        return processed_labels
    
    def _default_image_stats_calculator(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """默认图像统计计算函数
        
        Args:
            image: 图像数组
            
        Returns:
            图像统计信息字典
        """
        try:
            return {
                'mean': float(np.mean(image)),
                'std': float(np.std(image)),
                'min': float(np.min(image)),
                'max': float(np.max(image)),
                'shape': image.shape,
                'dtype': str(image.dtype)
            }
        except Exception as e:
            self.logger.warning(f"Error calculating image stats: {e}")
            return None
    
    def _generate_report_id(self) -> str:
        """生成报告ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"{self.dataset_name}_{timestamp}_{unique_id}"
    
    def load_datasets(self, splits: List[str] = ['train', 'valid', 'test']) -> Dict[str, Any]:
        """加载数据集
        
        Args:
            splits: 要加载的数据集分割
            
        Returns:
            加载的数据集字典
        """
        self.logger.info(f"Loading datasets: {self.dataset_name}")
        
        for split in splits:
            try:
                if split == 'train':
                    dataset = get_train_dataset(**self.dataset_config)
                elif split == 'valid' or split == 'val':
                    dataset = get_valid_dataset(**self.dataset_config)
                elif split == 'test':
                    dataset = get_test_dataset(**self.dataset_config)
                else:
                    dataset = get_dataset(split=split, **self.dataset_config)
                
                self.datasets[split] = dataset
                self.logger.info(f"  {split}: {len(dataset)} samples")
                
            except Exception as e:
                self.logger.warning(f"  Cannot load {split} dataset: {e}")
                
        return self.datasets
    
    def load_custom_dataset(self, dataset: Dataset, split_name: str = 'custom') -> None:
        """加载自定义数据集
        
        Args:
            dataset: PyTorch数据集对象
            split_name: 数据集分割名称
        """
        self.datasets[split_name] = dataset
        self.logger.info(f"Loaded custom dataset '{split_name}': {len(dataset)} samples")
    
    def analyze_class_distribution(self, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """分析类别分布
        
        Args:
            max_samples: 最大分析样本数，None表示分析全部
            
        Returns:
            类别分布分析结果
        """
        self.logger.info("Analyzing class distribution...")
        
        class_stats = {}
        
        for split_name, dataset in self.datasets.items():
            self.logger.info(f"  Analyzing {split_name} dataset...")
            
            # 获取标签信息
            labels = []
            label_shapes = []
            
            # 确定分析的样本数量
            num_samples = len(dataset)
            if max_samples is not None:
                num_samples = min(num_samples, max_samples)
            
            # 遍历数据集
            for idx in range(num_samples):
                try:
                    sample = dataset[idx]
                    
                    # 使用自定义标签提取函数
                    label = self.label_extractor(sample)
                    if label is None:
                        continue
                    
                    # 记录标签形状（对于标量标签记录为 (1,)）
                    if label.ndim == 0:
                        label_shapes.append((1,))  # 标量标签记录为 (1,)
                    else:
                        label_shapes.append(label.shape)
                    
                    # 使用自定义标签处理函数
                    processed_labels = self.label_processor(label)
                    labels.extend(processed_labels)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing sample {idx}: {e}")
                    continue
            
            # 统计类别分布
            if labels:
                label_counter = Counter(labels)
                total_labels = len(labels)
                
                class_stats[split_name] = {
                    'total_samples': len(dataset),
                    'analyzed_samples': num_samples,
                    'total_labels': total_labels,
                    'unique_classes': len(label_counter),
                    'class_counts': dict(label_counter),
                    'class_percentages': {k: (v/total_labels)*100 for k, v in label_counter.items()},
                    'label_shapes': label_shapes,
                    'most_common_shape': Counter(label_shapes).most_common(1)[0] if label_shapes else None
                }
                
                self.logger.info(f"    Found {len(label_counter)} unique classes")
            else:
                self.logger.warning(f"    No valid labels found in {split_name}")
                class_stats[split_name] = {
                    'total_samples': len(dataset),
                    'analyzed_samples': num_samples,
                    'error': 'No valid labels found'
                }
        
        self.analysis_results['class_distribution'] = class_stats
        return class_stats
    
    def analyze_data_quality(self, sample_size: int = 1000) -> Dict[str, Any]:
        """分析数据质量
        
        Args:
            sample_size: 采样分析的样本数量
            
        Returns:
            数据质量分析结果
        """
        self.logger.info("Analyzing data quality...")
        
        quality_stats = {}
        
        for split_name, dataset in self.datasets.items():
            self.logger.info(f"  Analyzing {split_name} dataset quality...")
            
            # 确定采样数量
            num_samples = min(len(dataset), sample_size)
            indices = np.random.choice(len(dataset), num_samples, replace=False)
            
            # 质量指标
            valid_samples = 0
            corrupted_samples = 0
            missing_labels = 0
            missing_images = 0
            image_stats_list = []
            
            for idx in indices:
                try:
                    sample = dataset[idx]
                    
                    # 检查图像
                    image = self.image_extractor(sample)
                    if image is None:
                        missing_images += 1
                        continue
                    
                    # 检查标签
                    label = self.label_extractor(sample)
                    if label is None:
                        missing_labels += 1
                        continue
                    
                    # 计算图像统计信息
                    image_stats = self.image_stats_calculator(image)
                    if image_stats:
                        image_stats_list.append(image_stats)
                    
                    valid_samples += 1
                    
                except Exception as e:
                    corrupted_samples += 1
                    self.logger.warning(f"Corrupted sample at index {idx}: {e}")
            
            # 汇总质量统计
            quality_stats[split_name] = {
                'total_samples': len(dataset),
                'analyzed_samples': num_samples,
                'valid_samples': valid_samples,
                'corrupted_samples': corrupted_samples,
                'missing_labels': missing_labels,
                'missing_images': missing_images,
                'quality_score': (valid_samples / num_samples) * 100 if num_samples > 0 else 0,
                'image_statistics': self._aggregate_image_stats(image_stats_list)
            }
            
            self.logger.info(f"    Quality score: {quality_stats[split_name]['quality_score']:.2f}%")
        
        self.analysis_results['data_quality'] = quality_stats
        return quality_stats
    
    def _aggregate_image_stats(self, stats_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """聚合图像统计信息"""
        if not stats_list:
            return {}
        
        # 收集所有数值统计
        means = [s['mean'] for s in stats_list if 'mean' in s]
        stds = [s['std'] for s in stats_list if 'std' in s]
        mins = [s['min'] for s in stats_list if 'min' in s]
        maxs = [s['max'] for s in stats_list if 'max' in s]
        shapes = [s['shape'] for s in stats_list if 'shape' in s]
        
        aggregated = {}
        
        if means:
            aggregated['mean_stats'] = {
                'mean': np.mean(means),
                'std': np.std(means),
                'min': np.min(means),
                'max': np.max(means)
            }
        
        if stds:
            aggregated['std_stats'] = {
                'mean': np.mean(stds),
                'std': np.std(stds),
                'min': np.min(stds),
                'max': np.max(stds)
            }
        
        if mins and maxs:
            aggregated['range_stats'] = {
                'min_value': np.min(mins),
                'max_value': np.max(maxs),
                'mean_range': np.mean([mx - mn for mn, mx in zip(mins, maxs)])
            }
        
        if shapes:
            shape_counter = Counter([str(shape) for shape in shapes])
            aggregated['shape_distribution'] = dict(shape_counter)
            aggregated['most_common_shape'] = shape_counter.most_common(1)[0]
        
        return aggregated
    
    def analyze_dataset_balance(self) -> Dict[str, Any]:
        """分析数据集平衡性"""
        self.logger.info("Analyzing dataset balance...")
        
        balance_stats = {}
        
        if 'class_distribution' not in self.analysis_results:
            self.analyze_class_distribution()
        
        class_dist = self.analysis_results['class_distribution']
        
        for split_name, stats in class_dist.items():
            if 'class_counts' not in stats:
                continue
            
            class_counts = stats['class_counts']
            counts = list(class_counts.values())
            
            if len(counts) < 2:
                balance_stats[split_name] = {'balance_score': 100.0, 'note': 'Single class'}
                continue
            
            # 计算平衡性指标
            min_count = min(counts)
            max_count = max(counts)
            mean_count = np.mean(counts)
            std_count = np.std(counts)
            
            # 平衡性得分（基于变异系数）
            cv = std_count / mean_count if mean_count > 0 else float('inf')
            balance_score = max(0, 100 - cv * 100)
            
            # 不平衡比率
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            balance_stats[split_name] = {
                'balance_score': balance_score,
                'imbalance_ratio': imbalance_ratio,
                'coefficient_of_variation': cv,
                'min_class_count': min_count,
                'max_class_count': max_count,
                'mean_class_count': mean_count,
                'std_class_count': std_count,
                'gini_coefficient': self._calculate_gini_coefficient(counts)
            }
        
        self.analysis_results['dataset_balance'] = balance_stats
        return balance_stats
    
    def _calculate_gini_coefficient(self, counts: List[int]) -> float:
        """计算基尼系数"""
        if not counts or len(counts) < 2:
            return 0.0
        
        counts = sorted(counts)
        n = len(counts)
        cumsum = np.cumsum(counts)
        
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    def visualize_class_distribution(self, save_plots: bool = True) -> List[Path]:
        """可视化类别分布
        
        Args:
            save_plots: 是否保存图片
            
        Returns:
            生成的图片文件路径列表
        """
        if 'class_distribution' not in self.analysis_results:
            self.analyze_class_distribution()
        
        saved_plots = []
        class_dist = self.analysis_results['class_distribution']
        
        for split_name, stats in class_dist.items():
            if 'class_counts' not in stats:
                continue
            
            # 创建条形图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            class_counts = stats['class_counts']
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            
            # 条形图
            bars = ax1.bar(range(len(classes)), counts, color='skyblue', alpha=0.7)
            ax1.set_title(f'Class Distribution - {split_name.upper()}', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Class ID')
            ax1.set_ylabel('Sample Count')
            ax1.set_xticks(range(len(classes)))
            ax1.set_xticklabels(classes, rotation=45)
            
            # 添加数值标签
            for bar, count in zip(bars, counts):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                        str(count), ha='center', va='bottom', fontsize=10)
            
            # 饼图
            ax2.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90)
            ax2.set_title(f'Class Proportion - {split_name.upper()}', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            if save_plots:
                plot_path = self.output_dir / f'class_distribution_{split_name}.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                saved_plots.append(plot_path)
                plt.close()
            else:
                plt.show()
        
        return saved_plots
    
    def visualize_data_quality(self, save_plots: bool = True) -> List[Path]:
        """可视化数据质量
        
        Args:
            save_plots: 是否保存图片
            
        Returns:
            生成的图片文件路径列表
        """
        if 'data_quality' not in self.analysis_results:
            self.analyze_data_quality()
        
        saved_plots = []
        quality_stats = self.analysis_results['data_quality']
        
        # 创建质量概览图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dataset Quality Analysis', fontsize=16, fontweight='bold')
        
        splits = list(quality_stats.keys())
        quality_scores = [quality_stats[split]['quality_score'] for split in splits]
        
        # 质量得分条形图
        ax1 = axes[0, 0]
        bars = ax1.bar(splits, quality_scores, color='lightgreen', alpha=0.7)
        ax1.set_title('Quality Scores by Split')
        ax1.set_ylabel('Quality Score (%)')
        ax1.set_ylim(0, 100)
        
        for bar, score in zip(bars, quality_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{score:.1f}%', ha='center', va='bottom')
        
        # 样本状态堆叠条形图
        ax2 = axes[0, 1]
        valid_counts = [quality_stats[split]['valid_samples'] for split in splits]
        corrupted_counts = [quality_stats[split]['corrupted_samples'] for split in splits]
        missing_labels = [quality_stats[split]['missing_labels'] for split in splits]
        missing_images = [quality_stats[split]['missing_images'] for split in splits]
        
        width = 0.6
        ax2.bar(splits, valid_counts, width, label='Valid', color='green', alpha=0.7)
        ax2.bar(splits, corrupted_counts, width, bottom=valid_counts, 
               label='Corrupted', color='red', alpha=0.7)
        ax2.bar(splits, missing_labels, width, 
               bottom=[v+c for v, c in zip(valid_counts, corrupted_counts)],
               label='Missing Labels', color='orange', alpha=0.7)
        
        ax2.set_title('Sample Status Distribution')
        ax2.set_ylabel('Sample Count')
        ax2.legend()
        
        # 图像统计信息（如果有的话）
        ax3 = axes[1, 0]
        if splits and 'image_statistics' in quality_stats[splits[0]]:
            # 显示图像均值分布
            for i, split in enumerate(splits):
                img_stats = quality_stats[split].get('image_statistics', {})
                if 'mean_stats' in img_stats:
                    mean_stats = img_stats['mean_stats']
                    ax3.bar(i, mean_stats['mean'], alpha=0.7, label=split)
            
            ax3.set_title('Average Image Intensity by Split')
            ax3.set_ylabel('Mean Intensity')
            ax3.set_xticks(range(len(splits)))
            ax3.set_xticklabels(splits)
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No image statistics available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Image Statistics')
        
        # 隐藏第四个子图或用于其他信息
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # 添加文本摘要
        summary_text = "Dataset Quality Summary:\n\n"
        for split in splits:
            stats = quality_stats[split]
            summary_text += f"{split.upper()}:\n"
            summary_text += f"  Quality Score: {stats['quality_score']:.1f}%\n"
            summary_text += f"  Valid Samples: {stats['valid_samples']}\n"
            summary_text += f"  Issues: {stats['corrupted_samples'] + stats['missing_labels'] + stats['missing_images']}\n\n"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.output_dir / 'data_quality_analysis.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            saved_plots.append(plot_path)
            plt.close()
        else:
            plt.show()
        
        return saved_plots
    
    def generate_analysis_report(self, include_recommendations: bool = True) -> str:
        """生成分析报告
        
        Args:
            include_recommendations: 是否包含改进建议
            
        Returns:
            报告内容字符串
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("NEUROTRAIN DATASET ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Dataset: {self.dataset_name}")
        report_lines.append(f"Report ID: {self.report_id}")
        report_lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 数据集概览
        if self.datasets:
            report_lines.append("DATASET OVERVIEW")
            report_lines.append("-" * 40)
            for split_name, dataset in self.datasets.items():
                report_lines.append(f"{split_name.upper()}: {len(dataset)} samples")
            report_lines.append("")
        
        # 类别分布分析
        if 'class_distribution' in self.analysis_results:
            report_lines.append("CLASS DISTRIBUTION ANALYSIS")
            report_lines.append("-" * 40)
            
            class_dist = self.analysis_results['class_distribution']
            for split_name, stats in class_dist.items():
                if 'unique_classes' in stats:
                    report_lines.append(f"\n{split_name.upper()}:")
                    report_lines.append(f"  Total samples: {stats['total_samples']}")
                    report_lines.append(f"  Unique classes: {stats['unique_classes']}")
                    report_lines.append(f"  Total labels: {stats['total_labels']}")
                    
                    # 显示前5个最常见的类别
                    class_counts = stats['class_counts']
                    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
                    report_lines.append("  Top 5 classes:")
                    for i, (class_id, count) in enumerate(sorted_classes[:5], 1):
                        percentage = stats['class_percentages'][class_id]
                        report_lines.append(f"    {i}. Class {class_id}: {count} samples ({percentage:.1f}%)")
        
        # 数据质量分析
        if 'data_quality' in self.analysis_results:
            report_lines.append("\n\nDATA QUALITY ANALYSIS")
            report_lines.append("-" * 40)
            
            quality_stats = self.analysis_results['data_quality']
            for split_name, stats in quality_stats.items():
                report_lines.append(f"\n{split_name.upper()}:")
                report_lines.append(f"  Quality Score: {stats['quality_score']:.2f}%")
                report_lines.append(f"  Valid samples: {stats['valid_samples']}")
                report_lines.append(f"  Corrupted samples: {stats['corrupted_samples']}")
                report_lines.append(f"  Missing labels: {stats['missing_labels']}")
                report_lines.append(f"  Missing images: {stats['missing_images']}")
        
        # 数据集平衡性分析
        if 'dataset_balance' in self.analysis_results:
            report_lines.append("\n\nDATASET BALANCE ANALYSIS")
            report_lines.append("-" * 40)
            
            balance_stats = self.analysis_results['dataset_balance']
            for split_name, stats in balance_stats.items():
                if 'balance_score' in stats:
                    report_lines.append(f"\n{split_name.upper()}:")
                    report_lines.append(f"  Balance Score: {stats['balance_score']:.2f}")
                    report_lines.append(f"  Imbalance Ratio: {stats['imbalance_ratio']:.2f}")
                    report_lines.append(f"  Gini Coefficient: {stats['gini_coefficient']:.3f}")
        
        # 改进建议
        if include_recommendations:
            recommendations = self._generate_recommendations()
            if recommendations:
                report_lines.append("\n\nRECOMMENDATIONS")
                report_lines.append("-" * 40)
                for rec in recommendations:
                    report_lines.append(f"• {rec}")
        
        report_lines.append("\n" + "=" * 80)
        
        # 保存报告
        report_content = "\n".join(report_lines)
        report_path = self.output_dir / 'dataset_analysis_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"Report saved to: {report_path}")
        return report_content
    
    def _generate_recommendations(self) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于数据质量的建议
        if 'data_quality' in self.analysis_results:
            quality_stats = self.analysis_results['data_quality']
            for split_name, stats in quality_stats.items():
                if stats['quality_score'] < 90:
                    recommendations.append(
                        f"Consider cleaning {split_name} dataset (quality score: {stats['quality_score']:.1f}%)"
                    )
                
                if stats['corrupted_samples'] > 0:
                    recommendations.append(
                        f"Remove or fix {stats['corrupted_samples']} corrupted samples in {split_name}"
                    )
        
        # 基于数据平衡性的建议
        if 'dataset_balance' in self.analysis_results:
            balance_stats = self.analysis_results['dataset_balance']
            for split_name, stats in balance_stats.items():
                if 'balance_score' in stats and stats['balance_score'] < 70:
                    recommendations.append(
                        f"Consider data augmentation or resampling for {split_name} (balance score: {stats['balance_score']:.1f})"
                    )
                
                if 'imbalance_ratio' in stats and stats['imbalance_ratio'] > 10:
                    recommendations.append(
                        f"High class imbalance detected in {split_name} (ratio: {stats['imbalance_ratio']:.1f}:1)"
                    )
        
        return recommendations
    
    def export_results(self, format: str = 'json') -> Path:
        """导出分析结果
        
        Args:
            format: 导出格式 ('json', 'csv', 'excel')
            
        Returns:
            导出文件路径
        """
        if format == 'json':
            export_path = self.output_dir / 'analysis_results.json'
            export_data = {
                'metadata': {
                    'dataset_name': self.dataset_name,
                    'report_id': self.report_id,
                    'generated_at': datetime.now().isoformat(),
                    'analyzer_version': '1.0.0'
                },
                'results': self.analysis_results
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        elif format == 'csv':
            export_path = self.output_dir / 'analysis_results.csv'
            # 将结果转换为表格格式
            rows = []
            for analysis_type, data in self.analysis_results.items():
                if isinstance(data, dict):
                    for split_name, split_data in data.items():
                        if isinstance(split_data, dict):
                            for metric, value in split_data.items():
                                rows.append({
                                    'analysis_type': analysis_type,
                                    'split': split_name,
                                    'metric': metric,
                                    'value': value
                                })
            
            df = pd.DataFrame(rows)
            df.to_csv(export_path, index=False)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Results exported to: {export_path}")
        return export_path
    
    def analyze_dataset(self, dataset: Optional[Dataset] = None, 
                       splits: List[str] = ['train', 'valid', 'test'],
                       max_samples: Optional[int] = None) -> Dict[str, Any]:
        """分析数据集（兼容性方法）
        
        Args:
            dataset: 可选的自定义数据集对象
            splits: 要分析的数据集分割
            max_samples: 最大分析样本数
            
        Returns:
            分析结果字典
        """
        if dataset is not None:
            self.load_custom_dataset(dataset, 'custom')
            splits = ['custom']
        
        return self.run_full_analysis(splits=splits, max_samples=max_samples)

    def run_full_analysis(self, 
                         splits: List[str] = ['train', 'valid', 'test'],
                         save_plots: bool = True,
                         max_samples: Optional[int] = None) -> Dict[str, Any]:
        """运行完整的数据集分析
        
        Args:
            splits: 要分析的数据集分割
            save_plots: 是否保存图片
            max_samples: 最大分析样本数
            
        Returns:
            完整的分析结果
        """
        self.logger.info("Starting full dataset analysis...")
        
        # 1. 加载数据集
        self.load_datasets(splits)
        
        # 2. 分析类别分布
        self.analyze_class_distribution(max_samples)
        
        # 3. 分析数据质量
        self.analyze_data_quality()
        
        # 4. 分析数据集平衡性
        self.analyze_dataset_balance()
        
        # 5. 生成可视化
        if save_plots:
            self.visualize_class_distribution(save_plots)
            self.visualize_data_quality(save_plots)
        
        # 6. 生成报告
        self.generate_analysis_report()
        
        # 7. 导出结果
        self.export_results('json')
        
        self.logger.info(f"Analysis completed. Results saved to: {self.output_dir}")
        
        return {
            'analysis_results': self.analysis_results,
            'output_directory': self.output_dir,
            'report_id': self.report_id
        }


def analyze_dataset(dataset_name: str,
                   dataset_config: Optional[Dict[str, Any]] = None,
                   dataset_object: Optional[Dataset] = None,
                   output_dir: Optional[str] = None,
                   splits: List[str] = ['train', 'valid', 'test']) -> Dict[str, Any]:
    """便捷函数：分析数据集
    
    Args:
        dataset_name: 数据集名称
        dataset_config: 数据集配置（用于加载NeuroTrain数据集）
        dataset_object: 自定义数据集对象
        output_dir: 输出目录
        splits: 要分析的数据集分割
        
    Returns:
        分析结果字典
    """
    analyzer = DatasetAnalyzer(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        output_dir=output_dir or f"output/analysis/{dataset_name}"
    )
    
    # 如果提供了自定义数据集对象，直接加载
    if dataset_object is not None:
        analyzer.load_custom_dataset(dataset_object, 'custom')
        splits = ['custom']
    
    # 运行完整分析
    return analyzer.run_full_analysis(splits=splits)


# 示例用法
if __name__ == "__main__":
    # 示例1: 分析NeuroTrain数据集
    config = {
        'dataset_name': 'CIFAR10',
        'data_dir': './data',
        'batch_size': 32
    }
    
    results = analyze_dataset(
        dataset_name='CIFAR10',
        dataset_config=config,
        splits=['train', 'test']
    )
    
    print(f"Analysis completed. Results saved to: {results['output_directory']}")
    
    # 示例2: 分析自定义数据集
    # from torch.utils.data import TensorDataset
    # import torch
    # 
    # # 创建示例数据集
    # X = torch.randn(1000, 3, 32, 32)
    # y = torch.randint(0, 10, (1000,))
    # custom_dataset = TensorDataset(X, y)
    # 
    # results = analyze_dataset(
    #     dataset_name='CustomDataset',
    #     dataset_object=custom_dataset
    # )