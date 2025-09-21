# 高级可视化工具
# 提供更多可视化功能和交互式可视化

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
import cv2

# 尝试导入交互式可视化库
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


class AdvancedVisualizer:
    """高级可视化工具类"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置matplotlib样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_attention_heatmap(self, attention_weights: np.ndarray, 
                              tokens: List[str] = None,
                              title: str = "Attention Heatmap",
                              save_path: Optional[Path] = None) -> None:
        """
        绘制注意力权重热力图
        
        Args:
            attention_weights: 注意力权重矩阵 (seq_len, seq_len)
            tokens: 标记列表
            title: 图表标题
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 创建热力图
        im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
        
        # 设置标签
        if tokens:
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right')
            ax.set_yticklabels(tokens)
        else:
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight', rotation=270, labelpad=20)
        
        # 设置标题
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        # 调整布局
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"注意力热力图已保存到: {save_path}")
        
        plt.show()
    
    def plot_feature_maps(self, feature_maps: torch.Tensor, 
                         layer_name: str = "Feature Maps",
                         max_channels: int = 16,
                         save_path: Optional[Path] = None) -> None:
        """
        可视化特征图
        
        Args:
            feature_maps: 特征图张量 (batch, channels, height, width)
            layer_name: 层名称
            max_channels: 最大显示通道数
            save_path: 保存路径
        """
        # 转换为numpy数组
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = feature_maps.detach().cpu().numpy()
        
        batch_size, channels, height, width = feature_maps.shape
        
        # 限制显示的通道数
        channels_to_show = min(channels, max_channels)
        
        # 计算网格大小
        cols = int(np.ceil(np.sqrt(channels_to_show)))
        rows = int(np.ceil(channels_to_show / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(channels_to_show):
            row = i // cols
            col = i % cols
            
            # 取第一个batch的特征图
            feature_map = feature_maps[0, i]
            
            im = axes[row, col].imshow(feature_map, cmap='viridis')
            axes[row, col].set_title(f'Channel {i}', fontsize=10)
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
        
        # 隐藏多余的子图
        for i in range(channels_to_show, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.suptitle(f'{layer_name} - Feature Maps', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"特征图已保存到: {save_path}")
        
        plt.show()
    
    def plot_training_curves(self, train_losses: List[float], 
                           valid_losses: List[float] = None,
                           train_metrics: Dict[str, List[float]] = None,
                           valid_metrics: Dict[str, List[float]] = None,
                           save_path: Optional[Path] = None) -> None:
        """
        绘制训练曲线
        
        Args:
            train_losses: 训练损失列表
            valid_losses: 验证损失列表
            train_metrics: 训练指标字典
            valid_metrics: 验证指标字典
            save_path: 保存路径
        """
        # 计算子图数量
        num_plots = 1  # 损失图
        if train_metrics or valid_metrics:
            num_plots += len(train_metrics or valid_metrics)
        
        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
        if num_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # 绘制损失曲线
        axes[plot_idx].plot(train_losses, label='Train Loss', color='blue', linewidth=2)
        if valid_losses:
            axes[plot_idx].plot(valid_losses, label='Valid Loss', color='red', linewidth=2)
        
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('Loss')
        axes[plot_idx].set_title('Training Loss')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
        
        # 绘制指标曲线
        if train_metrics or valid_metrics:
            all_metrics = set()
            if train_metrics:
                all_metrics.update(train_metrics.keys())
            if valid_metrics:
                all_metrics.update(valid_metrics.keys())
            
            for metric_name in all_metrics:
                if plot_idx >= num_plots:
                    break
                
                if train_metrics and metric_name in train_metrics:
                    axes[plot_idx].plot(train_metrics[metric_name], 
                                      label=f'Train {metric_name}', 
                                      color='blue', linewidth=2)
                
                if valid_metrics and metric_name in valid_metrics:
                    axes[plot_idx].plot(valid_metrics[metric_name], 
                                      label=f'Valid {metric_name}', 
                                      color='red', linewidth=2)
                
                axes[plot_idx].set_xlabel('Epoch')
                axes[plot_idx].set_ylabel(metric_name)
                axes[plot_idx].set_title(f'Training {metric_name}')
                axes[plot_idx].legend()
                axes[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练曲线已保存到: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            class_names: List[str] = None,
                            normalize: bool = True,
                            save_path: Optional[Path] = None) -> None:
        """
        绘制混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            class_names: 类别名称列表
            normalize: 是否归一化
            save_path: 保存路径
        """
        from sklearn.metrics import confusion_matrix
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        # 绘制热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵已保存到: {save_path}")
        
        plt.show()
    
    def create_interactive_dashboard(self, data: Dict, save_path: Optional[Path] = None) -> None:
        """
        创建交互式仪表板（使用Plotly）
        
        Args:
            data: 数据字典
            save_path: 保存路径
        """
        if not PLOTLY_AVAILABLE:
            print("警告: Plotly库未安装，无法创建交互式仪表板。请运行: pip install plotly")
            return
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Loss', 'Validation Metrics', 
                          'Model Performance', 'Data Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 添加训练损失图
        if 'train_losses' in data:
            fig.add_trace(
                go.Scatter(y=data['train_losses'], name='Train Loss',
                          line=dict(color='blue', width=2)),
                row=1, col=1
            )
        
        if 'valid_losses' in data:
            fig.add_trace(
                go.Scatter(y=data['valid_losses'], name='Valid Loss',
                          line=dict(color='red', width=2)),
                row=1, col=1
            )
        
        # 添加验证指标图
        if 'valid_metrics' in data:
            for metric_name, values in data['valid_metrics'].items():
                fig.add_trace(
                    go.Scatter(y=values, name=f'Valid {metric_name}',
                              line=dict(width=2)),
                    row=1, col=2
                )
        
        # 添加模型性能对比图
        if 'model_scores' in data:
            models = list(data['model_scores'].keys())
            scores = list(data['model_scores'].values())
            
            fig.add_trace(
                go.Bar(x=models, y=scores, name='Model Scores',
                      marker_color='lightblue'),
                row=2, col=1
            )
        
        # 添加数据分布图
        if 'data_distribution' in data:
            fig.add_trace(
                go.Histogram(x=data['data_distribution'], name='Data Distribution',
                           marker_color='lightgreen'),
                row=2, col=2
            )
        
        # 更新布局
        fig.update_layout(
            title_text="模型训练交互式仪表板",
            title_x=0.5,
            showlegend=True,
            height=800
        )
        
        # 保存为HTML文件
        if save_path:
            pyo.plot(fig, filename=str(save_path), auto_open=False)
            print(f"交互式仪表板已保存到: {save_path}")
        else:
            fig.show()
    
    def visualize_gradients(self, model: nn.Module, 
                          input_tensor: torch.Tensor,
                          target_class: int = None,
                          save_path: Optional[Path] = None) -> None:
        """
        可视化梯度（Grad-CAM风格）
        
        Args:
            model: 模型
            input_tensor: 输入张量
            target_class: 目标类别
            save_path: 保存路径
        """
        model.eval()
        
        # 注册钩子函数
        gradients = []
        activations = []
        
        def hook_grad(module, grad_input, grad_output):
            gradients.append(grad_output[0])
        
        def hook_activation(module, input, output):
            activations.append(output)
        
        # 注册钩子
        handles = []
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                handles.append(module.register_forward_hook(hook_activation))
                handles.append(module.register_backward_hook(hook_grad))
        
        # 前向传播
        output = model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # 反向传播
        model.zero_grad()
        output[0, target_class].backward()
        
        # 计算Grad-CAM
        if gradients and activations:
            # 取最后一个卷积层的梯度和激活
            grad = gradients[-1]
            activation = activations[-1]
            
            # 计算权重
            weights = grad.mean(dim=(2, 3), keepdim=True)
            
            # 计算加权激活
            cam = (weights * activation).sum(dim=1, keepdim=True)
            cam = torch.relu(cam)
            
            # 归一化
            cam = cam - cam.min()
            cam = cam / cam.max()
            
            # 转换为numpy
            cam = cam.squeeze().detach().cpu().numpy()
            
            # 可视化
            plt.figure(figsize=(10, 5))
            
            # 原始图像
            plt.subplot(1, 2, 1)
            if input_tensor.dim() == 4:
                img = input_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
                if img.shape[2] == 1:
                    img = img.squeeze(2)
                plt.imshow(img, cmap='gray' if img.shape[2] == 1 else None)
            plt.title('Original Image')
            plt.axis('off')
            
            # Grad-CAM
            plt.subplot(1, 2, 2)
            plt.imshow(cam, cmap='jet', alpha=0.8)
            plt.title(f'Grad-CAM (Class {target_class})')
            plt.colorbar()
            plt.axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"梯度可视化已保存到: {save_path}")
            
            plt.show()
        
        # 清理钩子
        for handle in handles:
            handle.remove()
    
    def create_streamlit_app(self, data: Dict) -> None:
        """
        创建Streamlit应用
        
        Args:
            data: 数据字典
        """
        if not STREAMLIT_AVAILABLE:
            print("警告: Streamlit库未安装，无法创建Streamlit应用。请运行: pip install streamlit")
            return
        
        st.set_page_config(page_title="模型分析仪表板", layout="wide")
        
        st.title("模型分析仪表板")
        
        # 侧边栏
        st.sidebar.title("控制面板")
        
        # 选择图表类型
        chart_type = st.sidebar.selectbox(
            "选择图表类型",
            ["训练曲线", "混淆矩阵", "特征图", "注意力热力图"]
        )
        
        # 根据选择显示不同图表
        if chart_type == "训练曲线":
            st.subheader("训练曲线")
            if 'train_losses' in data:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(data['train_losses'], label='Train Loss')
                if 'valid_losses' in data:
                    ax.plot(data['valid_losses'], label='Valid Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
        
        elif chart_type == "混淆矩阵":
            st.subheader("混淆矩阵")
            if 'confusion_matrix' in data:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(data['confusion_matrix'], annot=True, fmt='d', ax=ax)
                st.pyplot(fig)
        
        # 添加更多交互式功能
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 数据统计")
        
        if 'model_info' in data:
            st.sidebar.json(data['model_info'])


def create_visualization_report(results_dir: Path, 
                               output_dir: Optional[Path] = None) -> None:
    """
    创建完整的可视化报告
    
    Args:
        results_dir: 结果目录
        output_dir: 输出目录
    """
    if output_dir is None:
        output_dir = results_dir / "visualizations"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = AdvancedVisualizer(output_dir)
    
    # 这里可以添加从结果目录读取数据的逻辑
    # 并生成各种可视化图表
    
    print(f"可视化报告已生成到: {output_dir}")


if __name__ == "__main__":
    # 示例用法
    visualizer = AdvancedVisualizer(Path("output/visualizations"))
    
    # 示例数据
    train_losses = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2]
    valid_losses = [1.1, 0.9, 0.7, 0.5, 0.4, 0.3]
    
    # 绘制训练曲线
    visualizer.plot_training_curves(
        train_losses=train_losses,
        valid_losses=valid_losses,
        save_path=output_dir / "training_curves.png"
    )