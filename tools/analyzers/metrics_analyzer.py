"""
数据指标分析器模块

该模块提供全面的数据指标分析功能，包括：
- 模型性能指标计算与评估
- 多类别指标统计分析
- 指标对比与可视化
- 分析报告生成

Author: NeuroTrain Team
Date: 2024
"""

import os
import re
import json
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Sequence

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch

# Reporting utilities (optional dependency handled in the module)
try:
from tools.reporting import (
    HTMLReportGenerator,
    ReportRenderingError,
    ReportSummary,
    build_report_context,
)

REPORTING_AVAILABLE = True
except Exception:  # pragma: no cover - optional feature fallback
    HTMLReportGenerator = None  # type: ignore
    ReportRenderingError = None  # type: ignore
    ReportSummary = None  # type: ignore
    build_report_context = None  # type: ignore
    REPORTING_AVAILABLE = False

# 导入NeuroTrain的指标模块
try:
    from src.metrics import (
        dice, dice_coefficient, normalized_surface_dice, nsd,
        accuracy, recall, f1, precision, auc,
        iou_seg, iou_bbox, many_metrics,
        at_threshold, at_accuracy_threshold, at_recall_threshold,
        at_precision_threshold, at_f1_threshold, at_auc_threshold,
        mAP_at_iou_bbox, mAP_at_iou_seg, mF1_at_iou_bbox, mF1_at_iou_seg
    )
except ImportError:
    print("Warning: Could not import NeuroTrain metrics module")

# 尝试导入PDF生成库
try:
# pylint: disable=import-error
from reportlab.lib.pagesizes import letter, A4  # type: ignore
from reportlab.platypus import (  # type: ignore
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # type: ignore
from reportlab.lib.units import inch  # type: ignore
from reportlab.lib import colors  # type: ignore
REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class MetricsAnalyzer:
    """数据指标分析器
    
    提供全面的模型性能指标分析功能，包括指标计算、统计分析、
    可视化和报告生成等功能。
    """
    
    def __init__(self, result_dir: Optional[Path] = None):
        """初始化指标分析器
        
        Args:
            result_dir: 结果保存目录，默认为当前目录下的runs文件夹
        """
        if result_dir is None:
            result_dir = Path("output/analysis") / f"metrics_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("MetricsAnalyzer")
        
        # 支持的指标类型
        self.supported_metrics = {
            'classification': ['accuracy', 'precision', 'recall', 'f1', 'auc'],
            'segmentation': ['dice', 'iou_seg', 'nsd'],
            'detection': ['iou_bbox', 'mAP', 'mF1'],
            'regression': ['mse', 'mae', 'rmse', 'r2']
        }
    
    def calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                       y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """计算分类任务指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_prob: 预测概率（可选，用于计算AUC）
            
        Returns:
            包含各项指标的字典
        """
        metrics = {}
        
        try:
            metrics['accuracy'] = accuracy(y_true, y_pred)
            metrics['precision'] = precision(y_true, y_pred)
            metrics['recall'] = recall(y_true, y_pred)
            metrics['f1'] = f1(y_true, y_pred)
            
            if y_prob is not None:
                metrics['auc'] = auc(y_true, y_prob)
                
        except Exception as e:
            print(f"Error calculating classification metrics: {e}")
            
        return metrics
    
    def calculate_segmentation_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     spacing: Optional[Tuple] = None) -> Dict[str, float]:
        """计算分割任务指标
        
        Args:
            y_true: 真实分割掩码
            y_pred: 预测分割掩码
            spacing: 像素间距（用于NSD计算）
            
        Returns:
            包含各项指标的字典
        """
        metrics = {}
        
        try:
            metrics['dice'] = dice(y_true, y_pred)
            metrics['dice_coefficient'] = dice_coefficient(y_true, y_pred)
            metrics['iou_seg'] = iou_seg(y_true, y_pred)
            
            if spacing is not None:
                metrics['nsd'] = normalized_surface_dice(y_true, y_pred, spacing)
                
        except Exception as e:
            print(f"Error calculating segmentation metrics: {e}")
            
        return metrics
    
    def calculate_detection_metrics(self, y_true_boxes: List, y_pred_boxes: List,
                                  iou_threshold: float = 0.5) -> Dict[str, float]:
        """计算检测任务指标
        
        Args:
            y_true_boxes: 真实边界框列表
            y_pred_boxes: 预测边界框列表
            iou_threshold: IoU阈值
            
        Returns:
            包含各项指标的字典
        """
        metrics = {}
        
        try:
            # 计算IoU相关指标
            iou_scores = []
            for true_box, pred_box in zip(y_true_boxes, y_pred_boxes):
                iou_score = iou_bbox(true_box, pred_box)
                iou_scores.append(iou_score)
            
            metrics['mean_iou'] = np.mean(iou_scores)
            metrics['mAP'] = mAP_at_iou_bbox(y_true_boxes, y_pred_boxes, iou_threshold)
            metrics['mF1'] = mF1_at_iou_bbox(y_true_boxes, y_pred_boxes, iou_threshold)
            
        except Exception as e:
            print(f"Error calculating detection metrics: {e}")
            
        return metrics
    
    def analyze_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """分析分类指标（兼容性方法）
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            class_names: 类别名称列表
            
        Returns:
            分类指标分析结果
        """
        # 计算基础分类指标
        metrics = self.calculate_classification_metrics(y_true, y_pred)
        
        # 添加详细分析
        analysis_result = {
            'basic_metrics': metrics,
            'num_samples': len(y_true),
            'num_classes': len(np.unique(y_true)),
            'class_distribution': {
                'true': dict(zip(*np.unique(y_true, return_counts=True))),
                'pred': dict(zip(*np.unique(y_pred, return_counts=True)))
            }
        }
        
        if class_names:
            analysis_result['class_names'] = class_names
            
        return analysis_result

    def calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """计算回归任务指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            包含各项指标的字典
        """
        metrics = {}
        
        try:
            mse = np.mean((y_true - y_pred) ** 2)
            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(mse)
            
            # R²计算
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            metrics.update({
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            })
            
        except Exception as e:
            print(f"Error calculating regression metrics: {e}")
            
        return metrics
    
    def analyze_class_metrics(self, class_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """分析多类别指标
        
        Args:
            class_metrics: 每个类别的指标字典
            
        Returns:
            包含统计分析结果的字典
        """
        analysis = {
            'class_count': len(class_metrics),
            'metrics_summary': {},
            'class_ranking': {},
            'outliers': {}
        }
        
        if not class_metrics:
            return analysis
        
        # 获取所有指标名称
        all_metrics = set()
        for metrics in class_metrics.values():
            all_metrics.update(metrics.keys())
        
        # 对每个指标进行统计分析
        for metric_name in all_metrics:
            values = []
            for class_name, metrics in class_metrics.items():
                if metric_name in metrics:
                    values.append(metrics[metric_name])
            
            if values:
                analysis['metrics_summary'][metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
                
                # 类别排名（按该指标）
                class_scores = [(class_name, metrics.get(metric_name, 0)) 
                              for class_name, metrics in class_metrics.items()]
                class_scores.sort(key=lambda x: x[1], reverse=True)
                analysis['class_ranking'][metric_name] = class_scores
                
                # 检测异常值（使用IQR方法）
                q1, q3 = np.percentile(values, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = []
                for class_name, metrics in class_metrics.items():
                    if metric_name in metrics:
                        value = metrics[metric_name]
                        if value < lower_bound or value > upper_bound:
                            outliers.append((class_name, value))
                
                if outliers:
                    analysis['outliers'][metric_name] = outliers
        
        return analysis
    
    def visualize_metrics(self, metrics_data: Dict[str, Any], 
                         save_path: Optional[Path] = None) -> List[Path]:
        """可视化指标数据
        
        Args:
            metrics_data: 指标数据
            save_path: 保存路径
            
        Returns:
            生成的图片文件路径列表
        """
        if save_path is None:
            save_path = self.result_dir
        
        saved_plots = []
        
        # 设置中文字体和样式
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_style("whitegrid")
        
        # 1. 指标分布直方图
        if 'metrics_summary' in metrics_data:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Metrics Distribution Analysis', fontsize=16, fontweight='bold')
            
            metrics_summary = metrics_data['metrics_summary']
            metric_names = list(metrics_summary.keys())[:4]  # 最多显示4个指标
            
            for i, metric_name in enumerate(metric_names):
                ax = axes[i//2, i%2]
                summary = metrics_summary[metric_name]
                
                # 创建分布数据
                values = [summary['min'], summary['q1'] if 'q1' in summary else summary['mean'] - summary['std'],
                         summary['median'], summary['q3'] if 'q3' in summary else summary['mean'] + summary['std'],
                         summary['max']]
                
                ax.hist(values, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_title(f'{metric_name.upper()} Distribution')
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
            
            # 隐藏多余的子图
            for i in range(len(metric_names), 4):
                axes[i//2, i%2].set_visible(False)
            
            plt.tight_layout()
            plot_path = save_path / 'metrics_distribution.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            saved_plots.append(plot_path)
        
        # 2. 类别排名条形图
        if 'class_ranking' in metrics_data:
            for metric_name, rankings in metrics_data['class_ranking'].items():
                if len(rankings) > 1:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    classes = [item[0] for item in rankings[:10]]  # 显示前10个
                    scores = [item[1] for item in rankings[:10]]
                    
                    bars = ax.barh(classes, scores, color='lightcoral')
                    ax.set_title(f'Class Ranking by {metric_name.upper()}', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Score')
                    ax.set_ylabel('Class')
                    
                    # 添加数值标签
                    for bar, score in zip(bars, scores):
                        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                               f'{score:.3f}', va='center', fontsize=10)
                    
                    plt.tight_layout()
                    plot_path = save_path / f'class_ranking_{metric_name}.png'
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    saved_plots.append(plot_path)
        
        return saved_plots
    
    def generate_metrics_report(self, analysis_results: Dict[str, Any], 
                              output_path: Optional[Path] = None) -> str:
        """生成指标分析报告
        
        Args:
            analysis_results: 分析结果
            output_path: 输出路径
            
        Returns:
            报告内容字符串
        """
        if output_path is None:
            output_path = self.result_dir / 'metrics_report.txt'
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("NEUROTRAIN METRICS ANALYSIS REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 基本统计信息
        if 'class_count' in analysis_results:
            report_lines.append(f"Total Classes Analyzed: {analysis_results['class_count']}")
            report_lines.append("")
        
        # 指标摘要
        if 'metrics_summary' in analysis_results:
            report_lines.append("METRICS SUMMARY")
            report_lines.append("-" * 40)
            
            for metric_name, summary in analysis_results['metrics_summary'].items():
                report_lines.append(f"\n{metric_name.upper()}:")
                report_lines.append(f"  Mean: {summary['mean']:.4f}")
                report_lines.append(f"  Std:  {summary['std']:.4f}")
                report_lines.append(f"  Min:  {summary['min']:.4f}")
                report_lines.append(f"  Max:  {summary['max']:.4f}")
                report_lines.append(f"  Median: {summary['median']:.4f}")
        
        # 类别排名
        if 'class_ranking' in analysis_results:
            report_lines.append("\n\nCLASS RANKINGS")
            report_lines.append("-" * 40)
            
            for metric_name, rankings in analysis_results['class_ranking'].items():
                report_lines.append(f"\nTop 5 classes by {metric_name.upper()}:")
                for i, (class_name, score) in enumerate(rankings[:5], 1):
                    report_lines.append(f"  {i}. {class_name}: {score:.4f}")
        
        # 异常值检测
        if 'outliers' in analysis_results and analysis_results['outliers']:
            report_lines.append("\n\nOUTLIER DETECTION")
            report_lines.append("-" * 40)
            
            for metric_name, outliers in analysis_results['outliers'].items():
                if outliers:
                    report_lines.append(f"\n{metric_name.upper()} outliers:")
                    for class_name, value in outliers:
                        report_lines.append(f"  {class_name}: {value:.4f}")
        
        report_lines.append("\n" + "=" * 60)
        
        # 保存报告
        report_content = "\n".join(report_lines)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return report_content
    
    def export_to_json(self, analysis_results: Dict[str, Any], 
                      output_path: Optional[Path] = None) -> Path:
        """导出分析结果为JSON格式
        
        Args:
            analysis_results: 分析结果
            output_path: 输出路径
            
        Returns:
            JSON文件路径
        """
        if output_path is None:
            output_path = self.result_dir / 'metrics_analysis.json'
        
        # 添加元数据
        export_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'analyzer_version': '1.0.0',
                'total_classes': analysis_results.get('class_count', 0)
            },
            'analysis_results': analysis_results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def compare_models(self, model_metrics: Dict[str, Dict[str, float]], 
                      save_comparison: bool = True) -> Dict[str, Any]:
        """比较多个模型的指标
        
        Args:
            model_metrics: 模型指标字典 {model_name: {metric_name: value}}
            save_comparison: 是否保存比较结果
            
        Returns:
            比较分析结果
        """
        comparison_results = {
            'model_count': len(model_metrics),
            'metric_comparison': {},
            'model_ranking': {},
            'best_model_per_metric': {}
        }
        
        if not model_metrics:
            return comparison_results
        
        # 获取所有指标
        all_metrics = set()
        for metrics in model_metrics.values():
            all_metrics.update(metrics.keys())
        
        # 对每个指标进行比较
        for metric_name in all_metrics:
            metric_values = {}
            for model_name, metrics in model_metrics.items():
                if metric_name in metrics:
                    metric_values[model_name] = metrics[metric_name]
            
            if metric_values:
                # 找出最佳模型
                best_model = max(metric_values.items(), key=lambda x: x[1])
                comparison_results['best_model_per_metric'][metric_name] = best_model
                
                # 排序
                sorted_models = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
                comparison_results['model_ranking'][metric_name] = sorted_models
                
                # 统计信息
                values = list(metric_values.values())
                comparison_results['metric_comparison'][metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'range': max(values) - min(values),
                    'best_score': best_model[1],
                    'best_model': best_model[0]
                }
        
        if save_comparison:
            # 保存比较结果
            self.export_to_json(comparison_results, self.result_dir / 'model_comparison.json')
            
            # 生成比较报告
            self._generate_comparison_report(comparison_results)
        
        return comparison_results
    
    def _generate_comparison_report(self, comparison_results: Dict[str, Any]) -> None:
        """生成模型比较报告"""
        report_path = self.result_dir / 'model_comparison_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("MODEL COMPARISON REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"Total Models Compared: {comparison_results['model_count']}\n\n")
            
            # 最佳模型汇总
            f.write("BEST MODEL PER METRIC\n")
            f.write("-" * 40 + "\n")
            for metric_name, (model_name, score) in comparison_results['best_model_per_metric'].items():
                f.write(f"{metric_name.upper()}: {model_name} ({score:.4f})\n")
            
            # 详细排名
            f.write("\n\nDETAILED RANKINGS\n")
            f.write("-" * 40 + "\n")
            for metric_name, rankings in comparison_results['model_ranking'].items():
                f.write(f"\n{metric_name.upper()} Rankings:\n")
                for i, (model_name, score) in enumerate(rankings, 1):
                    f.write(f"  {i}. {model_name}: {score:.4f}\n")

    # ------------------------------------------------------------------
    # New functionality: HTML/PDF reporting integration

    def generate_html_report(
        self,
        summary: Dict[str, Any],
        metrics_overall: Optional[Dict[str, float]] = None,
        metrics_per_class: Optional[Dict[str, Dict[str, float]]] = None,
        charts: Optional[Sequence[Any]] = None,
        monitor_json: Optional[Path] = None,
        artifacts: Optional[Sequence[Dict[str, Any]]] = None,
        notes: Optional[Sequence[str]] = None,
        html_filename: str = "report.html",
        convert_to_pdf: bool = True,
    ) -> Optional[Path]:
        """Generate an enriched HTML (and optional PDF) report.

        This method is a high-level convenience wrapper around the
        :mod:`tools.reporting` package.  It collects the provided data
        and renders the built-in HTML template.  PDF export requires
        either WeasyPrint or pdfkit.

        Parameters
        ----------
        summary:
            Dictionary containing high level experiment summary. Expected
            keys include ``title``, ``task``, ``model_name`` etc.
        metrics_overall:
            Mapping of aggregate metrics.
        metrics_per_class:
            Mapping of per-class metrics.
        charts:
            Sequence of chart entries (file paths or dictionaries as
            accepted by :class:`HTMLReportGenerator`).
        monitor_json:
            Path to monitor JSON file to embed in the report.
        artifacts:
            Additional artefacts to list in the report.
        notes:
            Optional textual notes.
        html_filename:
            Name of the generated HTML file.
        convert_to_pdf:
            If True, attempt to generate PDF via available backend.
        """

        if not REPORTING_AVAILABLE:
            self.logger.warning("HTML reporting not available (missing dependencies)")
            return None

        try:
            summary_obj = ReportSummary(**summary)
        except TypeError as exc:  # pragma: no cover
            raise ValueError(f"Invalid summary fields: {exc}") from exc

        context = build_report_context(
            summary=summary_obj,
            metrics_overall=metrics_overall,
            metrics_per_class=metrics_per_class,
            charts=charts,
            monitor_json=monitor_json,
            artifacts=artifacts,
            notes=notes,
        )

        generator = HTMLReportGenerator()
        html_path = generator.render(
            output_dir=self.result_dir,
            context=context,
            html_filename=html_filename,
            convert_to_pdf=convert_to_pdf,
        )

        self.logger.info("Enriched HTML report generated at %s", html_path)
        return html_path


def analyze_model_metrics(metrics_data: Union[Dict, Path], 
                         task_type: str = 'classification',
                         output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """便捷函数：分析模型指标
    
    Args:
        metrics_data: 指标数据或包含指标的文件路径
        task_type: 任务类型 ('classification', 'segmentation', 'detection', 'regression')
        output_dir: 输出目录
        
    Returns:
        分析结果字典
    """
    analyzer = MetricsAnalyzer(output_dir)
    
    # 加载数据
    if isinstance(metrics_data, (str, Path)):
        with open(metrics_data, 'r', encoding='utf-8') as f:
            if str(metrics_data).endswith('.json'):
                metrics_data = json.load(f)
            else:
                # 假设是CSV格式
                import pandas as pd
                df = pd.read_csv(metrics_data)
                metrics_data = df.to_dict('records')
    
    # 分析指标
    if isinstance(metrics_data, dict) and 'class_metrics' in metrics_data:
        analysis_results = analyzer.analyze_class_metrics(metrics_data['class_metrics'])
    else:
        analysis_results = analyzer.analyze_class_metrics(metrics_data)
    
    # 生成可视化
    plots = analyzer.visualize_metrics(analysis_results)
    
    # 生成报告
    report = analyzer.generate_metrics_report(analysis_results)
    
    # 导出JSON
    json_path = analyzer.export_to_json(analysis_results)
    
    return {
        'analysis_results': analysis_results,
        'report_content': report,
        'generated_plots': plots,
        'json_export': json_path,
        'output_directory': analyzer.result_dir
    }


# 示例用法
if __name__ == "__main__":
    # 创建示例数据
    sample_metrics = {
        'class_1': {'accuracy': 0.95, 'precision': 0.92, 'recall': 0.89, 'f1': 0.90},
        'class_2': {'accuracy': 0.88, 'precision': 0.85, 'recall': 0.91, 'f1': 0.88},
        'class_3': {'accuracy': 0.92, 'precision': 0.89, 'recall': 0.94, 'f1': 0.91},
        'class_4': {'accuracy': 0.87, 'precision': 0.84, 'recall': 0.88, 'f1': 0.86},
        'class_5': {'accuracy': 0.93, 'precision': 0.91, 'recall': 0.92, 'f1': 0.91}
    }
    
    # 使用便捷函数分析
    results = analyze_model_metrics(sample_metrics, task_type='classification')
    print(f"Analysis completed. Results saved to: {results['output_directory']}")