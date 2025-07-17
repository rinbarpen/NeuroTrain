import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision.models import resnet50, resnet34

# data analyzer
import os
import json
import uuid
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Literal, Callable
from collections import Counter, defaultdict
from datetime import datetime

# 导入项目中的数据集和绘图工具
from src.dataset import get_dataset, get_train_dataset, get_test_dataset, get_valid_dataset
from src.utils import Plot, Font, CmapPresets, ThemePresets, DataSaver

from typing import Dict, List, Any, Optional, Union, Literal, Callable

# 定义函数类型
LabelExtractorType = Callable[[Any], Optional[np.ndarray]]
ImageExtractorType = Callable[[Any], Optional[np.ndarray]]
LabelProcessorType = Callable[[np.ndarray], List[int]]
ImageStatsCalculatorType = Callable[[np.ndarray], Optional[Dict[str, Any]]]

class DatasetAnalyzer:
    """数据集分析器类
    
    功能包括：
    1. 数据集类别分布分析
    2. 数据统计信息分析
    3. 图表可视化
    4. 分析报告生成和保存
    """
    
    def __init__(self, 
                 dataset_name: str,
                 dataset_config: Dict[str, Any],
                 output_dir: str = "output/analyzer",
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
        self.dataset_config = dataset_config
        self.report_id = report_id or self._generate_report_id()
        
        # 初始化logger
        self.logger = logger or logging.getLogger(f"DatasetAnalyzer.{self.dataset_name}")
        
        # 设置自定义函数或使用默认函数
        self.label_extractor = label_extractor or self._default_label_extractor
        self.image_extractor = image_extractor or self._default_image_extractor
        self.label_processor = label_processor or self._default_label_processor
        self.image_stats_calculator = image_stats_calculator or self._default_image_stats_calculator
        
        # 创建输出目录
        self.output_dir = Path(output_dir) / self.report_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据存储
        self.analysis_results = {}
        self.datasets = {}
        self.statistics = {}
        
        # 设置绘图主题
        ThemePresets.apply_custom_theme('scientific')
        
    def _default_label_extractor(self, sample: Any) -> Optional[np.ndarray]:
        """默认标签提取函数
        
        Args:
            sample: 数据集样本
            
        Returns:
            提取的标签，如果提取失败返回None
        """
        try:
            if isinstance(sample, (tuple, list)) and len(sample) >= 2:
                label = sample[1]  # 假设标签是第二个元素
                
                if torch.is_tensor(label):
                    label = label.numpy()
                
                return label
            else:
                self.logger.warning(f"样本格式不符合预期: {type(sample)}")
                return None
                
        except Exception as e:
            self.logger.warning(f"提取标签时出错: {e}")
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
            else:
                self.logger.warning(f"样本格式不符合预期: {type(sample)}")
                return None
                
        except Exception as e:
            self.logger.warning(f"提取图像时出错: {e}")
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
            self.logger.warning(f"处理标签时出错: {e}")
            
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
                'shape': image.shape
            }
        except Exception as e:
            self.logger.warning(f"计算图像统计信息时出错: {e}")
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
        self.logger.info(f"正在加载数据集: {self.dataset_name}")
        
        for split in splits:
            try:
                if split == 'train':
                    dataset = get_train_dataset(**self.dataset_config)
                elif split == 'valid':
                    dataset = get_valid_dataset(**self.dataset_config)
                elif split == 'test':
                    dataset = get_test_dataset(**self.dataset_config)
                else:
                    dataset = get_dataset(split=split, **self.dataset_config)
                
                self.datasets[split] = dataset
                self.logger.info(f"  {split}: {len(dataset)} 样本")
                
            except Exception as e:
                self.logger.warning(f"  无法加载 {split} 数据集: {e}")
                
        return self.datasets
    
    def analyze_class_distribution(self) -> Dict[str, Any]:
        """分析类别分布"""
        self.logger.info("正在分析类别分布...")
        
        class_stats = {}
        
        for split_name, dataset in self.datasets.items():
            self.logger.info(f"  分析 {split_name} 数据集...")
            
            # 获取标签信息
            labels = []
            label_shapes = []
            
            # 遍历整个数据集
            for idx in range(len(dataset)):
                try:
                    sample = dataset[idx]
                    
                    # 使用自定义标签提取函数
                    label = self.label_extractor(sample)
                    if label is None:
                        continue
                    
                    # 记录标签形状
                    label_shapes.append(label.shape)
                    
                    # 使用自定义标签处理函数
                    processed_labels = self.label_processor(label)
                    labels.extend(processed_labels)
                    
                except Exception as e:
                    self.logger.warning(f"    处理样本 {idx} 时出错: {e}")
                    continue
            
            # 统计类别分布
            if labels:
                class_counts = Counter(labels)
                total_samples = len(labels)
                
                class_stats[split_name] = {
                    'total_samples': len(dataset),
                    'analyzed_samples': len(dataset),
                    'class_counts': dict(class_counts),
                    'class_distribution': {k: v/total_samples for k, v in class_counts.items()},
                    'num_classes': len(class_counts),
                    'label_shapes': Counter([str(shape) for shape in label_shapes]),
                    'most_common_classes': class_counts.most_common(10)
                }
            
        self.analysis_results['class_distribution'] = class_stats
        return class_stats
    
    def analyze_data_statistics(self) -> Dict[str, Any]:
        """分析数据统计信息"""
        self.logger.info("正在分析数据统计信息...")
        
        data_stats = {}
        
        for split_name, dataset in self.datasets.items():
            self.logger.info(f"  分析 {split_name} 数据集统计信息...")
            
            # 遍历整个数据集
            image_shapes = []
            image_stats = []
            
            for idx in range(len(dataset)):
                try:
                    sample = dataset[idx]
                    
                    # 使用自定义图像提取函数
                    image = self.image_extractor(sample)
                    if image is None:
                        continue
                    
                    image_shapes.append(image.shape)
                    
                    # 使用自定义图像统计计算函数
                    stats = self.image_stats_calculator(image)
                    if stats is not None:
                        image_stats.append(stats)
                        
                except Exception as e:
                    self.logger.warning(f"    处理样本 {idx} 时出错: {e}")
                    continue
            
            if image_stats:
                # 汇总统计信息
                data_stats[split_name] = {
                    'total_samples': len(dataset),
                    'analyzed_samples': len(dataset),
                    'image_shapes': Counter([str(shape) for shape in image_shapes]),
                    'mean_intensity': np.mean([s['mean'] for s in image_stats]),
                    'std_intensity': np.mean([s['std'] for s in image_stats]),
                    'intensity_range': {
                        'min': min([s['min'] for s in image_stats]),
                        'max': max([s['max'] for s in image_stats])
                    }
                }
        
        self.analysis_results['data_statistics'] = data_stats
        return data_stats
    
    def visualize_class_distribution(self, save_plots: bool = True) -> None:
        """可视化类别分布"""
        self.logger.info("正在生成类别分布图表...")
        
        if 'class_distribution' not in self.analysis_results:
            self.logger.error("请先运行 analyze_class_distribution()")
            return
        
        class_stats = self.analysis_results['class_distribution']
        
        # 为每个数据集分割创建图表
        for split_name, stats in class_stats.items():
            if 'class_counts' not in stats:
                continue
                
            class_counts = stats['class_counts']
            
            # 创建条形图
            plot = Plot(figsize=(12, 8))
            ax = plot.subplot()
            
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            
            # 使用painter绘制条形图
            bars = ax.bar(range(len(classes)), counts, 
                         color=CmapPresets.get_cmap('classification', 'multi_class'))
            
            # 设置标题和标签
            ax.set_title(f'{self.dataset_name} - {split_name.title()} 数据集类别分布', 
                        fontdict=Font.title_font().build())
            ax.set_xlabel('类别', fontdict=Font.label_font().build())
            ax.set_ylabel('样本数量', fontdict=Font.label_font().build())
            
            # 设置x轴标签
            ax.set_xticks(range(len(classes)))
            ax.set_xticklabels([f'Class {c}' for c in classes], rotation=45)
            
            # 添加数值标签
            for i, (bar, count) in enumerate(zip(bars, counts)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                       f'{count}', ha='center', va='bottom',
                       fontdict=Font.small_font().build())
            
            plot.tight_layout()
            
            if save_plots:
                plot_path = self.output_dir / f'class_distribution_{split_name}.png'
                plot.savefig(str(plot_path), dpi=300, bbox_inches='tight')
                self.logger.info(f"  保存图表: {plot_path}")
            
            plot.show()
    
    def visualize_data_overview(self, save_plots: bool = True) -> None:
        """可视化数据概览"""
        self.logger.info("正在生成数据概览图表...")
        
        # 数据集大小对比
        plot = Plot(figsize=(10, 6))
        ax = plot.subplot()
        
        splits = list(self.datasets.keys())
        sizes = [len(dataset) for dataset in self.datasets.values()]
        
        bars = ax.bar(splits, sizes, color=CmapPresets.get_cmap('analysis', 'heatmap_blue'))
        
        ax.set_title(f'{self.dataset_name} 数据集大小对比', 
                    fontdict=Font.title_font().build())
        ax.set_ylabel('样本数量', fontdict=Font.label_font().build())
        
        # 添加数值标签
        for bar, size in zip(bars, sizes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(sizes)*0.01,
                   f'{size}', ha='center', va='bottom',
                   fontdict=Font.small_font().build())
        
        plot.tight_layout()
        
        if save_plots:
            plot_path = self.output_dir / 'dataset_overview.png'
            plot.savefig(str(plot_path), dpi=300, bbox_inches='tight')
            self.logger.info(f"  保存图表: {plot_path}")
        
        plot.show()
    
    def generate_report(self) -> Dict[str, Any]:
        """生成分析报告"""
        self.logger.info("正在生成分析报告...")
        
        report = {
            'metadata': {
                'report_id': self.report_id,
                'dataset_name': self.dataset_name,
                'analysis_time': datetime.now().isoformat(),
                'dataset_config': self.dataset_config
            },
            'summary': {
                'total_datasets': len(self.datasets),
                'dataset_splits': list(self.datasets.keys()),
                'total_samples': sum(len(ds) for ds in self.datasets.values())
            },
            'analysis_results': self.analysis_results
        }
        
        # 保存JSON报告
        report_path = self.output_dir / 'analysis_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"  保存报告: {report_path}")
        
        # 生成Markdown报告
        self._generate_markdown_report(report)
        
        return report
    
    def _generate_markdown_report(self, report: Dict[str, Any]) -> None:
        """生成Markdown格式的报告"""
        md_content = f"""# 数据集分析报告

## 基本信息
- **数据集名称**: {report['metadata']['dataset_name']}
- **报告ID**: {report['metadata']['report_id']}
- **分析时间**: {report['metadata']['analysis_time']}
- **数据集分割**: {', '.join(report['summary']['dataset_splits'])}
- **总样本数**: {report['summary']['total_samples']}

## 数据集概览
"""
        
        # 添加类别分布信息
        if 'class_distribution' in self.analysis_results:
            md_content += "\n### 类别分布\n\n"
            for split, stats in self.analysis_results['class_distribution'].items():
                md_content += f"#### {split.title()} 数据集\n"
                md_content += f"- 总样本数: {stats['total_samples']}\n"
                md_content += f"- 类别数量: {stats['num_classes']}\n"
                md_content += f"- 最常见类别: {stats['most_common_classes'][:3]}\n\n"
        
        # 添加数据统计信息
        if 'data_statistics' in self.analysis_results:
            md_content += "\n### 数据统计\n\n"
            for split, stats in self.analysis_results['data_statistics'].items():
                md_content += f"#### {split.title()} 数据集\n"
                md_content += f"- 平均强度: {stats['mean_intensity']:.4f}\n"
                md_content += f"- 强度标准差: {stats['std_intensity']:.4f}\n"
                md_content += f"- 强度范围: [{stats['intensity_range']['min']:.4f}, {stats['intensity_range']['max']:.4f}]\n\n"
        
        # 保存Markdown报告
        md_path = self.output_dir / 'analysis_report.md'
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        self.logger.info(f"  保存Markdown报告: {md_path}")
    
    def run_full_analysis(self, 
                         splits: List[str] = ['train', 'valid', 'test'],
                         save_plots: bool = True) -> Dict[str, Any]:
        """运行完整的数据集分析
        
        Args:
            splits: 要分析的数据集分割
            save_plots: 是否保存图表
            
        Returns:
            完整的分析报告
        """
        self.logger.info(f"开始对数据集 '{self.dataset_name}' 进行完整分析")
        self.logger.info(f"报告ID: {self.report_id}")
        self.logger.info(f"输出目录: {self.output_dir}")
        
        try:
            # 1. 加载数据集
            self.load_datasets(splits)
            
            # 2. 分析类别分布
            self.analyze_class_distribution()
            
            # 3. 分析数据统计信息
            self.analyze_data_statistics()
            
            # 4. 生成可视化图表
            self.visualize_class_distribution(save_plots)
            self.visualize_data_overview(save_plots)
            
            # 5. 生成报告
            report = self.generate_report()
            
            self.logger.info(f"✅ 分析完成！结果保存在: {self.output_dir}")
            return report
            
        except Exception as e:
            self.logger.error(f"❌ 分析过程中出现错误: {e}")
            raise


# 使用示例
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s | %(name)s | %(message)s'
    )
    
    # 自定义标签提取函数示例
    def custom_label_extractor(sample):
        """自定义标签提取函数"""
        if isinstance(sample, dict):
            return sample.get('label')
        elif isinstance(sample, (tuple, list)) and len(sample) >= 2:
            return sample[1]
        return None
    
    # 自定义标签处理函数示例
    def custom_label_processor(label):
        """自定义标签处理函数"""
        if label.ndim == 0:
            return [int(label)]
        elif label.ndim == 2:  # 分割掩码
            unique_values = np.unique(label)
            return unique_values[unique_values > 0].tolist()  # 排除背景类
        return []
    
    # 示例配置
    dataset_config = {
        'dataset_name': 'drive',
        'base_dir': Path('data/DRIVE'),
        'transforms': None,
        'use_numpy': False
    }
    
    # 创建分析器，使用自定义函数
    analyzer = DatasetAnalyzer(
        dataset_name='DRIVE',
        dataset_config=dataset_config,
        output_dir='output/analyzer',
        label_extractor=custom_label_extractor,
        label_processor=custom_label_processor
    )
    
    # 运行完整分析
    report = analyzer.run_full_analysis(
        splits=['train', 'test'],
        save_plots=True
    )
    
    analyzer.logger.info("分析报告摘要:")
    analyzer.logger.info(f"- 数据集分割: {report['summary']['dataset_splits']}")
    analyzer.logger.info(f"- 总样本数: {report['summary']['total_samples']}")

