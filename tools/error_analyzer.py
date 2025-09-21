# 错误样本分析工具
# 用于分析模型预测错误的样本，找出错误模式

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import cv2
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# 尝试导入交互式可视化库
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class ErrorAnalyzer:
    """错误样本分析工具"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置matplotlib样式
        plt.style.use('default')
        sns.set_palette("husl")
    
    def analyze_prediction_errors(self, y_true: np.ndarray, 
                                y_pred: np.ndarray,
                                y_prob: np.ndarray = None,
                                class_names: List[str] = None,
                                save_path: Optional[Path] = None) -> Dict:
        """
        分析预测错误
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_prob: 预测概率
            class_names: 类别名称
            save_path: 保存路径
            
        Returns:
            Dict: 错误分析结果
        """
        # 计算错误样本
        error_mask = y_true != y_pred
        error_indices = np.where(error_mask)[0]
        
        analysis = {
            'total_samples': len(y_true),
            'error_samples': len(error_indices),
            'error_rate': len(error_indices) / len(y_true),
            'error_indices': error_indices.tolist(),
            'error_details': {}
        }
        
        # 按类别分析错误
        if class_names:
            for i, class_name in enumerate(class_names):
                class_mask = y_true == i
                class_errors = error_mask & class_mask
                class_error_indices = np.where(class_errors)[0]
                
                analysis['error_details'][class_name] = {
                    'total_samples': np.sum(class_mask),
                    'error_samples': len(class_error_indices),
                    'error_rate': len(class_error_indices) / np.sum(class_mask) if np.sum(class_mask) > 0 else 0,
                    'error_indices': class_error_indices.tolist()
                }
        
        # 分析错误类型
        error_types = {}
        for idx in error_indices:
            true_class = y_true[idx]
            pred_class = y_pred[idx]
            error_type = f"{true_class}_to_{pred_class}"
            
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(idx)
        
        analysis['error_types'] = error_types
        
        # 分析预测置信度
        if y_prob is not None:
            error_confidences = y_prob[error_indices]
            correct_confidences = y_prob[~error_mask]
            
            analysis['confidence_analysis'] = {
                'error_mean_confidence': float(np.mean(error_confidences)),
                'correct_mean_confidence': float(np.mean(correct_confidences)),
                'error_std_confidence': float(np.std(error_confidences)),
                'correct_std_confidence': float(np.std(correct_confidences))
            }
        
        return analysis
    
    def visualize_error_distribution(self, y_true: np.ndarray, 
                                   y_pred: np.ndarray,
                                   class_names: List[str] = None,
                                   save_path: Optional[Path] = None) -> None:
        """
        可视化错误分布
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            class_names: 类别名称
            save_path: 保存路径
        """
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 创建子图
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 原始混淆矩阵
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted Label')
        axes[0].set_ylabel('True Label')
        
        # 归一化混淆矩阵
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Reds',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[1])
        axes[1].set_title('Normalized Confusion Matrix')
        axes[1].set_xlabel('Predicted Label')
        axes[1].set_ylabel('True Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"错误分布图已保存到: {save_path}")
        
        plt.show()
    
    def visualize_error_samples(self, images: np.ndarray,
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              error_indices: List[int],
                              class_names: List[str] = None,
                              max_samples: int = 16,
                              save_path: Optional[Path] = None) -> None:
        """
        可视化错误样本
        
        Args:
            images: 图像数组
            y_true: 真实标签
            y_pred: 预测标签
            error_indices: 错误样本索引
            class_names: 类别名称
            max_samples: 最大显示样本数
            save_path: 保存路径
        """
        # 选择要显示的错误样本
        display_indices = error_indices[:max_samples]
        
        # 计算网格大小
        cols = int(np.ceil(np.sqrt(len(display_indices))))
        rows = int(np.ceil(len(display_indices) / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        else:
            axes = axes.reshape(rows, cols)
        
        for i, idx in enumerate(display_indices):
            row = i // cols
            col = i % cols
            
            if rows == 1:
                ax = axes[col] if cols > 1 else axes
            elif cols == 1:
                ax = axes[row] if rows > 1 else axes
            else:
                ax = axes[row, col]
            
            # 显示图像
            if images.ndim == 4:  # (batch, channels, height, width)
                img = images[idx].transpose(1, 2, 0)
            else:  # (batch, height, width)
                img = images[idx]
            
            # 如果是单通道，转换为RGB
            if img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
            
            ax.imshow(img)
            
            # 设置标题
            true_label = class_names[y_true[idx]] if class_names else f"True: {y_true[idx]}"
            pred_label = class_names[y_pred[idx]] if class_names else f"Pred: {y_pred[idx]}"
            ax.set_title(f"{true_label} → {pred_label}", fontsize=10, color='red')
            ax.axis('off')
        
        # 隐藏多余的子图
        for i in range(len(display_indices), rows * cols):
            row = i // cols
            col = i % cols
            if rows == 1:
                axes[col].axis('off')
            elif cols == 1:
                axes[row].axis('off')
            else:
                axes[row, col].axis('off')
        
        plt.suptitle('Error Samples', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"错误样本可视化已保存到: {save_path}")
        
        plt.show()
    
    def analyze_feature_space(self, features: np.ndarray,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            method: str = 'tsne',
                            save_path: Optional[Path] = None) -> None:
        """
        分析特征空间中的错误分布
        
        Args:
            features: 特征向量
            y_true: 真实标签
            y_pred: 预测标签
            method: 降维方法 ('tsne' 或 'pca')
            save_path: 保存路径
        """
        # 降维
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        elif method == 'pca':
            reducer = PCA(n_components=2)
        else:
            raise ValueError("method must be 'tsne' or 'pca'")
        
        features_2d = reducer.fit_transform(features)
        
        # 创建子图
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 按真实标签着色
        scatter1 = axes[0].scatter(features_2d[:, 0], features_2d[:, 1], 
                                  c=y_true, cmap='tab10', alpha=0.7)
        axes[0].set_title(f'Feature Space (True Labels) - {method.upper()}')
        axes[0].set_xlabel('Component 1')
        axes[0].set_ylabel('Component 2')
        plt.colorbar(scatter1, ax=axes[0])
        
        # 按预测标签着色
        scatter2 = axes[1].scatter(features_2d[:, 0], features_2d[:, 1], 
                                  c=y_pred, cmap='tab10', alpha=0.7)
        axes[1].set_title(f'Feature Space (Predicted Labels) - {method.upper()}')
        axes[1].set_xlabel('Component 1')
        axes[1].set_ylabel('Component 2')
        plt.colorbar(scatter2, ax=axes[1])
        
        # 标记错误样本
        error_mask = y_true != y_pred
        error_points = features_2d[error_mask]
        axes[0].scatter(error_points[:, 0], error_points[:, 1], 
                       c='red', marker='x', s=100, label='Errors')
        axes[1].scatter(error_points[:, 0], error_points[:, 1], 
                       c='red', marker='x', s=100, label='Errors')
        
        axes[0].legend()
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"特征空间分析已保存到: {save_path}")
        
        plt.show()
    
    def analyze_confidence_distribution(self, y_prob: np.ndarray,
                                      y_true: np.ndarray,
                                      y_pred: np.ndarray,
                                      save_path: Optional[Path] = None) -> None:
        """
        分析预测置信度分布
        
        Args:
            y_prob: 预测概率
            y_true: 真实标签
            y_pred: 预测标签
            save_path: 保存路径
        """
        # 计算最大概率（置信度）
        confidence = np.max(y_prob, axis=1)
        
        # 分离正确和错误的预测
        correct_mask = y_true == y_pred
        correct_confidence = confidence[correct_mask]
        error_confidence = confidence[~correct_mask]
        
        # 创建子图
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 置信度分布直方图
        axes[0].hist(correct_confidence, bins=30, alpha=0.7, label='Correct', color='green')
        axes[0].hist(error_confidence, bins=30, alpha=0.7, label='Error', color='red')
        axes[0].set_xlabel('Confidence')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Confidence Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 置信度箱线图
        data_to_plot = [correct_confidence, error_confidence]
        labels = ['Correct', 'Error']
        axes[1].boxplot(data_to_plot, labels=labels)
        axes[1].set_ylabel('Confidence')
        axes[1].set_title('Confidence Box Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"置信度分布分析已保存到: {save_path}")
        
        plt.show()
    
    def create_error_report(self, analysis: Dict,
                          save_path: Optional[Path] = None) -> str:
        """
        创建错误分析报告
        
        Args:
            analysis: 错误分析结果
            save_path: 保存路径
            
        Returns:
            str: 报告内容
        """
        report_lines = []
        report_lines.append("# 模型错误分析报告")
        report_lines.append("")
        
        # 总体统计
        report_lines.append("## 总体统计")
        report_lines.append("")
        report_lines.append(f"- 总样本数: {analysis['total_samples']}")
        report_lines.append(f"- 错误样本数: {analysis['error_samples']}")
        report_lines.append(f"- 错误率: {analysis['error_rate']:.2%}")
        report_lines.append("")
        
        # 按类别分析
        if 'error_details' in analysis:
            report_lines.append("## 按类别错误分析")
            report_lines.append("")
            for class_name, details in analysis['error_details'].items():
                report_lines.append(f"### {class_name}")
                report_lines.append(f"- 总样本数: {details['total_samples']}")
                report_lines.append(f"- 错误样本数: {details['error_samples']}")
                report_lines.append(f"- 错误率: {details['error_rate']:.2%}")
                report_lines.append("")
        
        # 错误类型分析
        if 'error_types' in analysis:
            report_lines.append("## 错误类型分析")
            report_lines.append("")
            for error_type, indices in analysis['error_types'].items():
                report_lines.append(f"- {error_type}: {len(indices)} 个样本")
            report_lines.append("")
        
        # 置信度分析
        if 'confidence_analysis' in analysis:
            conf_analysis = analysis['confidence_analysis']
            report_lines.append("## 置信度分析")
            report_lines.append("")
            report_lines.append(f"- 正确预测平均置信度: {conf_analysis['correct_mean_confidence']:.3f}")
            report_lines.append(f"- 错误预测平均置信度: {conf_analysis['error_mean_confidence']:.3f}")
            report_lines.append(f"- 正确预测置信度标准差: {conf_analysis['correct_std_confidence']:.3f}")
            report_lines.append(f"- 错误预测置信度标准差: {conf_analysis['error_std_confidence']:.3f}")
            report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        if save_path:
            save_path.write_text(report_content, encoding='utf-8')
            print(f"错误分析报告已保存到: {save_path}")
        
        return report_content
    
    def create_interactive_error_dashboard(self, analysis: Dict,
                                         features: np.ndarray = None,
                                         y_true: np.ndarray = None,
                                         y_pred: np.ndarray = None,
                                         save_path: Optional[Path] = None) -> None:
        """
        创建交互式错误分析仪表板
        
        Args:
            analysis: 错误分析结果
            features: 特征向量
            y_true: 真实标签
            y_pred: 预测标签
            save_path: 保存路径
        """
        if not PLOTLY_AVAILABLE:
            print("警告: Plotly库未安装，无法创建交互式仪表板")
            return
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('错误率统计', '错误类型分布', 
                          '置信度分布', '特征空间'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # 错误率统计
        if 'error_details' in analysis:
            classes = list(analysis['error_details'].keys())
            error_rates = [details['error_rate'] for details in analysis['error_details'].values()]
            
            fig.add_trace(
                go.Bar(x=classes, y=error_rates, name='错误率'),
                row=1, col=1
            )
        
        # 错误类型分布
        if 'error_types' in analysis:
            error_types = list(analysis['error_types'].keys())
            error_counts = [len(indices) for indices in analysis['error_types'].values()]
            
            fig.add_trace(
                go.Pie(labels=error_types, values=error_counts, name='错误类型'),
                row=1, col=2
            )
        
        # 置信度分布
        if 'confidence_analysis' in analysis:
            conf_analysis = analysis['confidence_analysis']
            
            # 模拟置信度数据
            correct_conf = np.random.normal(
                conf_analysis['correct_mean_confidence'],
                conf_analysis['correct_std_confidence'],
                1000
            )
            error_conf = np.random.normal(
                conf_analysis['error_mean_confidence'],
                conf_analysis['error_std_confidence'],
                1000
            )
            
            fig.add_trace(
                go.Histogram(x=correct_conf, name='正确预测', opacity=0.7),
                row=2, col=1
            )
            fig.add_trace(
                go.Histogram(x=error_conf, name='错误预测', opacity=0.7),
                row=2, col=1
            )
        
        # 特征空间
        if features is not None and y_true is not None and y_pred is not None:
            # 使用PCA降维
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features)
            
            error_mask = y_true != y_pred
            
            fig.add_trace(
                go.Scatter(x=features_2d[~error_mask, 0], 
                          y=features_2d[~error_mask, 1],
                          mode='markers', name='正确预测',
                          marker=dict(color='green', opacity=0.7)),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=features_2d[error_mask, 0], 
                          y=features_2d[error_mask, 1],
                          mode='markers', name='错误预测',
                          marker=dict(color='red', opacity=0.7)),
                row=2, col=2
            )
        
        # 更新布局
        fig.update_layout(
            title_text="模型错误分析交互式仪表板",
            title_x=0.5,
            showlegend=True,
            height=800
        )
        
        # 保存为HTML文件
        if save_path:
            pyo.plot(fig, filename=str(save_path), auto_open=False)
            print(f"交互式错误分析仪表板已保存到: {save_path}")
        else:
            fig.show()


def analyze_model_errors(model: nn.Module,
                        dataloader: torch.utils.data.DataLoader,
                        device: str = 'cuda',
                        class_names: List[str] = None,
                        output_dir: Path = None) -> Dict:
    """
    分析模型错误
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        class_names: 类别名称
        output_dir: 输出目录
        
    Returns:
        Dict: 错误分析结果
    """
    if output_dir is None:
        output_dir = Path("output/error_analysis")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analyzer = ErrorAnalyzer(output_dir)
    
    # 收集预测结果
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_features = []
    all_images = []
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                images, labels = batch
            else:
                images = batch
                labels = None
            
            images = images.to(device)
            
            # 获取预测
            outputs = model(images)
            if isinstance(outputs, (list, tuple)):
                logits = outputs[0]
            else:
                logits = outputs
            
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy() if labels is not None else [0] * len(predictions))
            all_probabilities.extend(probabilities.cpu().numpy())
            all_images.extend(images.cpu().numpy())
    
    # 转换为numpy数组
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_prob = np.array(all_probabilities)
    images = np.array(all_images)
    
    # 分析错误
    analysis = analyzer.analyze_prediction_errors(
        y_true, y_pred, y_prob, class_names
    )
    
    # 生成可视化
    analyzer.visualize_error_distribution(
        y_true, y_pred, class_names,
        save_path=output_dir / "error_distribution.png"
    )
    
    analyzer.visualize_error_samples(
        images, y_true, y_pred, analysis['error_indices'],
        class_names, save_path=output_dir / "error_samples.png"
    )
    
    analyzer.analyze_confidence_distribution(
        y_prob, y_true, y_pred,
        save_path=output_dir / "confidence_distribution.png"
    )
    
    # 生成报告
    analyzer.create_error_report(
        analysis, save_path=output_dir / "error_report.md"
    )
    
    # 创建交互式仪表板
    analyzer.create_interactive_error_dashboard(
        analysis, save_path=output_dir / "error_dashboard.html"
    )
    
    print(f"错误分析完成，结果保存在: {output_dir}")
    
    return analysis


if __name__ == "__main__":
    # 示例用法
    output_dir = Path("output/error_analysis")
    analyzer = ErrorAnalyzer(output_dir)
    
    # 创建示例数据
    np.random.seed(42)
    y_true = np.random.randint(0, 3, 1000)
    y_pred = np.random.randint(0, 3, 1000)
    y_prob = np.random.dirichlet([1, 1, 1], 1000)
    
    # 分析错误
    analysis = analyzer.analyze_prediction_errors(
        y_true, y_pred, y_prob, 
        class_names=['Class A', 'Class B', 'Class C']
    )
    
    # 生成可视化
    analyzer.visualize_error_distribution(
        y_true, y_pred, ['Class A', 'Class B', 'Class C'],
        save_path=output_dir / "example_error_distribution.png"
    )
    
    # 生成报告
    analyzer.create_error_report(
        analysis, save_path=output_dir / "example_error_report.md"
    )