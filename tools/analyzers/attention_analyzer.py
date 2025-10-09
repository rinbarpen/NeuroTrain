"""
Attention Analyzer Module

This module provides comprehensive attention mechanism analysis and visualization
for neural network models, particularly Transformer-based architectures.

Features:
- Multi-head attention visualization
- Attention pattern analysis
- Attention flow visualization
- Interactive attention plots
- Attention statistics and metrics
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any
import json
import logging
from datetime import datetime
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


class AttentionAnalyzer:
    """
    增强的注意力机制分析器
    
    提供全面的注意力机制分析功能，包括：
    1. 多头注意力权重提取和可视化
    2. SE模块等通道注意力分析
    3. 空间注意力机制分析
    4. 注意力模式分析和流可视化
    5. 统计分析和报告生成
    6. 常规注意力机制支持（SE、CBAM、ECA等）
    """
    
    def __init__(self, 
                 output_dir: Union[str, Path] = "output/analysis",
                 logger: Optional[logging.Logger] = None):
        """
        初始化注意力分析器
        
        Args:
            output_dir: 输出目录路径
            logger: 日志记录器，如果为None则创建新的
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化logger
        self.logger = logger or self._setup_logger()
        
        # 设置matplotlib样式
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 存储分析结果
        self.analysis_results = {}
        
        # 支持的注意力机制类型
        self.supported_attention_types = {
            'multi_head': 'Multi-Head Attention',
            'channel': 'Channel Attention (SE, ECA, etc.)',
            'spatial': 'Spatial Attention',
            'mixed': 'Mixed Attention (CBAM, etc.)'
        }
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("AttentionAnalyzer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def extract_attention_weights(self, 
                                model: nn.Module, 
                                input_tensor: torch.Tensor,
                                layer_indices: Optional[List[int]] = None,
                                attention_type: str = 'multi_head') -> Dict[str, torch.Tensor]:
        """
        从模型中提取注意力权重
        
        Args:
            model: 神经网络模型
            input_tensor: 输入张量
            layer_indices: 要提取的层索引列表，None表示提取所有层
            attention_type: 注意力类型 ('multi_head', 'channel', 'spatial', 'mixed')
            
        Returns:
            Dict[str, torch.Tensor]: 层名称到注意力权重的映射
        """
        self.logger.info("开始提取注意力权重...")
        
        attention_weights = {}
        hooks = []
        
        def attention_hook(name, attention_type):
            def hook(module, input, output):
                if attention_type == 'multi_head':
                    # 多头注意力权重提取
                    if hasattr(module, 'attention_weights'):
                        attention_weights[name] = module.attention_weights.detach()
                    elif hasattr(module, 'self_attn') and hasattr(module.self_attn, 'attention_weights'):
                        attention_weights[name] = module.self_attn.attention_weights.detach()
                    elif hasattr(module, 'attn') and hasattr(module.attn, 'attention_weights'):
                        attention_weights[name] = module.attn.attention_weights.detach()
                    elif isinstance(output, tuple) and len(output) > 1:
                        # 假设第二个输出是注意力权重
                        attention_weights[name] = output[1].detach()
                        
                elif attention_type == 'channel':
                    # 通道注意力权重提取（SE模块等）
                    if hasattr(module, 'channel_weights'):
                        attention_weights[name] = module.channel_weights.detach()
                    elif 'se' in name.lower() or 'squeeze' in name.lower():
                        # SE模块输出通常是通道权重
                        if isinstance(output, torch.Tensor):
                            attention_weights[name] = output.detach()
                            
                elif attention_type == 'spatial':
                    # 空间注意力权重提取
                    if hasattr(module, 'spatial_weights'):
                        attention_weights[name] = module.spatial_weights.detach()
                    elif 'spatial' in name.lower():
                        if isinstance(output, torch.Tensor):
                            attention_weights[name] = output.detach()
                            
                elif attention_type == 'mixed':
                    # 混合注意力权重提取（CBAM等）
                    if hasattr(module, 'attention_weights'):
                        attention_weights[name] = module.attention_weights.detach()
                    elif isinstance(output, dict):
                        # CBAM等可能返回字典格式
                        attention_weights[name] = output
                    elif isinstance(output, tuple):
                        attention_weights[name] = output
            return hook
        
        # 根据注意力类型注册相应的hook
        for name, module in model.named_modules():
            if self._is_attention_module(module, attention_type):
                if layer_indices is None or any(str(idx) in name for idx in layer_indices):
                    hook = module.register_forward_hook(attention_hook(name, attention_type))
                    hooks.append(hook)
                    self.logger.debug(f"注册钩子到模块: {name}")
        
        # 前向传播提取注意力权重
        model.eval()
        with torch.no_grad():
            _ = model(input_tensor)
        
        # 清理钩子
        for hook in hooks:
            hook.remove()
        
        self.logger.info(f"成功提取 {len(attention_weights)} 层的注意力权重")
        return attention_weights
    
    def _is_attention_module(self, module: nn.Module, attention_type: str) -> bool:
        """
        判断模块是否为指定类型的注意力模块
        
        Args:
            module: 神经网络模块
            attention_type: 注意力类型
            
        Returns:
            bool: 是否为注意力模块
        """
        module_name = module.__class__.__name__.lower()
        
        if attention_type == 'multi_head':
            return (isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)) or
                   'attention' in module_name or 
                   'multihead' in module_name or
                   hasattr(module, 'num_heads'))
                   
        elif attention_type == 'channel':
            return ('se' in module_name or 
                   'squeeze' in module_name or
                   'excitation' in module_name or
                   'eca' in module_name or
                   'channel' in module_name)
                   
        elif attention_type == 'spatial':
            return ('spatial' in module_name or
                   ('conv' in module_name and hasattr(module, 'kernel_size')))
                   
        elif attention_type == 'mixed':
            return ('cbam' in module_name or
                   'bam' in module_name or
                   'mixed' in module_name)
                   
        return False
    
    def analyze_channel_attention(self, 
                                channel_weights: torch.Tensor,
                                channel_names: Optional[List[str]] = None,
                                save_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        分析通道注意力权重（SE模块等）
        
        Args:
            channel_weights: 通道注意力权重 [batch_size, channels] 或 [channels]
            channel_names: 通道名称列表
            save_path: 保存路径
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        if channel_weights.dim() > 2:
            # 如果是多维张量，取平均
            channel_weights = channel_weights.mean(dim=tuple(range(2, channel_weights.dim())))
        
        if channel_weights.dim() == 2:
            # 如果有batch维度，取平均
            channel_weights = channel_weights.mean(dim=0)
        
        weights_np = channel_weights.cpu().numpy()
        num_channels = len(weights_np)
        
        # 计算统计信息
        analysis_results = {
            'mean_weight': float(np.mean(weights_np)),
            'std_weight': float(np.std(weights_np)),
            'max_weight': float(np.max(weights_np)),
            'min_weight': float(np.min(weights_np)),
            'top_channels': np.argsort(weights_np)[-10:].tolist(),  # 前10个重要通道
            'bottom_channels': np.argsort(weights_np)[:10].tolist(),  # 后10个不重要通道
            'sparsity': float(np.sum(weights_np < 0.1) / num_channels),  # 稀疏性
            'concentration': float(np.sum(weights_np > 0.8) / num_channels)  # 集中度
        }
        
        # 可视化通道注意力权重
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 通道权重分布
        axes[0, 0].bar(range(num_channels), weights_np)
        axes[0, 0].set_title('Channel Attention Weights Distribution')
        axes[0, 0].set_xlabel('Channel Index')
        axes[0, 0].set_ylabel('Attention Weight')
        
        # 2. 权重直方图
        axes[0, 1].hist(weights_np, bins=50, alpha=0.7)
        axes[0, 1].set_title('Weight Distribution Histogram')
        axes[0, 1].set_xlabel('Weight Value')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. 累积分布
        sorted_weights = np.sort(weights_np)[::-1]
        cumsum_weights = np.cumsum(sorted_weights)
        axes[1, 0].plot(cumsum_weights / cumsum_weights[-1])
        axes[1, 0].set_title('Cumulative Weight Distribution')
        axes[1, 0].set_xlabel('Channel Rank')
        axes[1, 0].set_ylabel('Cumulative Weight Ratio')
        
        # 4. Top-K重要通道
        top_k = min(20, num_channels)
        top_indices = np.argsort(weights_np)[-top_k:]
        top_weights = weights_np[top_indices]
        
        axes[1, 1].barh(range(top_k), top_weights)
        if channel_names:
            axes[1, 1].set_yticklabels([channel_names[i] if i < len(channel_names) else f'Ch{i}' 
                                      for i in top_indices])
        else:
            axes[1, 1].set_yticklabels([f'Channel {i}' for i in top_indices])
        axes[1, 1].set_title(f'Top {top_k} Important Channels')
        axes[1, 1].set_xlabel('Attention Weight')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Channel attention analysis saved to {save_path}")
        
        plt.show()
        
        return analysis_results
    
    def analyze_spatial_attention(self, 
                                spatial_weights: torch.Tensor,
                                input_image: Optional[torch.Tensor] = None,
                                save_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        分析空间注意力权重
        
        Args:
            spatial_weights: 空间注意力权重 [batch_size, height, width] 或 [height, width]
            input_image: 原始输入图像（可选）
            save_path: 保存路径
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        if spatial_weights.dim() == 3:
            # 如果有batch维度，取第一个样本
            spatial_weights = spatial_weights[0]
        
        weights_np = spatial_weights.cpu().numpy()
        height, width = weights_np.shape
        
        # 计算统计信息
        analysis_results = {
            'mean_weight': float(np.mean(weights_np)),
            'std_weight': float(np.std(weights_np)),
            'max_weight': float(np.max(weights_np)),
            'min_weight': float(np.min(weights_np)),
            'center_of_mass': [float(x) for x in np.unravel_index(np.argmax(weights_np), weights_np.shape)],
            'attention_area': float(np.sum(weights_np > np.mean(weights_np)) / (height * width)),
            'max_position': [int(x) for x in np.unravel_index(np.argmax(weights_np), weights_np.shape)]
        }
        
        # 可视化空间注意力
        if input_image is not None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 原始图像
            if input_image.dim() == 4:
                input_image = input_image[0]  # 取第一个batch
            if input_image.dim() == 3 and input_image.shape[0] in [1, 3]:
                input_image = input_image.permute(1, 2, 0)
            
            axes[0].imshow(input_image.cpu().numpy())
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # 注意力热图
            im1 = axes[1].imshow(weights_np, cmap='hot', interpolation='bilinear')
            axes[1].set_title('Spatial Attention Heatmap')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1])
            
            # 叠加显示
            axes[2].imshow(input_image.cpu().numpy())
            axes[2].imshow(weights_np, cmap='hot', alpha=0.6, interpolation='bilinear')
            axes[2].set_title('Attention Overlay')
            axes[2].axis('off')
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # 注意力热图
            im1 = axes[0].imshow(weights_np, cmap='hot', interpolation='bilinear')
            axes[0].set_title('Spatial Attention Heatmap')
            axes[0].axis('off')
            plt.colorbar(im1, ax=axes[0])
            
            # 3D表面图
            x = np.arange(width)
            y = np.arange(height)
            X, Y = np.meshgrid(x, y)
            
            ax = fig.add_subplot(122, projection='3d')
            ax.plot_surface(X, Y, weights_np, cmap='hot', alpha=0.8)
            ax.set_title('3D Attention Surface')
            ax.set_xlabel('Width')
            ax.set_ylabel('Height')
            ax.set_zlabel('Attention Weight')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Spatial attention analysis saved to {save_path}")
        
        plt.show()
        
        return analysis_results
    
    def visualize_attention_heads(self, 
                                attention_weights: torch.Tensor,
                                tokens: Optional[List[str]] = None,
                                head_indices: Optional[List[int]] = None,
                                layer_name: str = "Attention",
                                save_path: Optional[Path] = None,
                                figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        可视化多头注意力的各个头
        
        Args:
            attention_weights: 注意力权重 (num_heads, seq_len, seq_len)
            tokens: 标记列表
            head_indices: 要可视化的头索引列表
            layer_name: 层名称
            save_path: 保存路径
            figsize: 图像大小
        """
        if attention_weights.dim() == 3:
            num_heads, seq_len, _ = attention_weights.shape
        else:
            raise ValueError("注意力权重应该是3维张量 (num_heads, seq_len, seq_len)")
        
        # 确定要可视化的头
        if head_indices is None:
            head_indices = list(range(min(8, num_heads)))  # 最多显示8个头
        
        num_heads_to_show = len(head_indices)
        
        # 计算子图布局
        cols = min(4, num_heads_to_show)
        rows = (num_heads_to_show + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        fig.suptitle(f'{layer_name} - Multi-Head Attention', fontsize=16, fontweight='bold')
        
        # 处理单个子图的情况
        if num_heads_to_show == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(-1, 1)
        else:
            axes = axes.reshape(rows, cols)
        
        for i, head_idx in enumerate(head_indices):
            row = i // cols
            col = i % cols
            
            if rows == 1:
                ax = axes[col] if cols > 1 else axes[0]
            elif cols == 1:
                ax = axes[row] if rows > 1 else axes[0]
            else:
                ax = axes[row, col]
            
            # 获取当前头的注意力权重
            head_attention = attention_weights[head_idx].cpu().numpy()
            
            # 绘制热力图
            im = ax.imshow(head_attention, cmap='Blues', aspect='auto')
            
            # 设置标签
            if tokens:
                ax.set_xticks(range(len(tokens)))
                ax.set_yticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
                ax.set_yticklabels(tokens, fontsize=8)
            
            ax.set_title(f'Head {head_idx}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            
            # 添加颜色条
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # 隐藏多余的子图
        for i in range(num_heads_to_show, rows * cols):
            row = i // cols
            col = i % cols
            if rows == 1:
                axes[col].axis('off')
            elif cols == 1:
                axes[row].axis('off')
            else:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"注意力头可视化保存到: {save_path}")
        
        plt.show()
    
    def analyze_attention_patterns(self, 
                                 attention_weights: torch.Tensor,
                                 tokens: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        分析注意力模式
        
        Args:
            attention_weights: 注意力权重 (seq_len, seq_len) 或 (num_heads, seq_len, seq_len)
            tokens: 标记列表
            
        Returns:
            Dict: 分析结果
        """
        self.logger.info("开始分析注意力模式...")
        
        # 处理多头注意力，取平均
        if attention_weights.dim() == 3:
            attention_weights = attention_weights.mean(dim=0)
        
        attention_weights = attention_weights.cpu().numpy()
        
        analysis = {}
        
        # 计算注意力熵 (衡量注意力分散程度)
        attention_probs = attention_weights / (attention_weights.sum(axis=1, keepdims=True) + 1e-8)
        entropy = -np.sum(attention_probs * np.log(attention_probs + 1e-8), axis=1)
        analysis['entropy'] = {
            'values': entropy.tolist(),
            'mean': float(np.mean(entropy)),
            'std': float(np.std(entropy)),
            'description': '注意力熵，值越高表示注意力越分散'
        }
        
        # 计算注意力集中度 (最大注意力权重)
        max_attention = np.max(attention_weights, axis=1)
        analysis['max_attention'] = {
            'values': max_attention.tolist(),
            'mean': float(np.mean(max_attention)),
            'std': float(np.std(max_attention)),
            'description': '最大注意力权重，值越高表示注意力越集中'
        }
        
        # 计算注意力分散度 (标准差)
        attention_std = np.std(attention_weights, axis=1)
        analysis['attention_std'] = {
            'values': attention_std.tolist(),
            'mean': float(np.mean(attention_std)),
            'std': float(np.std(attention_std)),
            'description': '注意力权重标准差，衡量注意力分布的离散程度'
        }
        
        # 找到最受关注的token
        most_attended = np.argmax(attention_weights, axis=1)
        analysis['most_attended'] = {
            'indices': most_attended.tolist(),
            'description': '每个位置最关注的token索引'
        }
        
        # 计算自注意力强度 (对角线元素)
        self_attention = np.diag(attention_weights)
        analysis['self_attention'] = {
            'values': self_attention.tolist(),
            'mean': float(np.mean(self_attention)),
            'std': float(np.std(self_attention)),
            'description': '自注意力强度，表示每个token对自身的关注程度'
        }
        
        # 如果有tokens，添加token相关分析
        if tokens:
            analysis['tokens'] = tokens
            analysis['token_attention_summary'] = {}
            for i, token in enumerate(tokens):
                analysis['token_attention_summary'][token] = {
                    'receives_attention': float(np.sum(attention_weights[:, i])),
                    'gives_attention': float(np.sum(attention_weights[i, :])),
                    'self_attention': float(attention_weights[i, i])
                }
        
        # 添加分析时间戳
        analysis['analysis_timestamp'] = datetime.now().isoformat()
        
        self.logger.info("注意力模式分析完成")
        return analysis
    
    def visualize_attention_flow(self, 
                               attention_weights: torch.Tensor,
                               tokens: Optional[List[str]] = None,
                               threshold: float = 0.1,
                               save_path: Optional[Path] = None,
                               figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        可视化注意力流
        
        Args:
            attention_weights: 注意力权重
            tokens: 标记列表
            threshold: 显示阈值，低于此值的连接不显示
            save_path: 保存路径
            figsize: 图像大小
        """
        if attention_weights.dim() == 3:
            attention_weights = attention_weights.mean(dim=0)
        
        attention_weights = attention_weights.cpu().numpy()
        seq_len = attention_weights.shape[0]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 创建位置坐标
        positions = np.arange(seq_len)
        
        # 绘制连接线
        for i in range(seq_len):
            for j in range(seq_len):
                if attention_weights[i, j] > threshold:
                    # 线条粗细和透明度与注意力权重成正比
                    linewidth = attention_weights[i, j] * 5
                    alpha = min(attention_weights[i, j] * 2, 1.0)
                    
                    ax.plot([i, j], [0, 1], 
                           linewidth=linewidth, 
                           alpha=alpha, 
                           color='blue')
        
        # 设置标签
        if tokens:
            ax.set_xticks(positions)
            ax.set_xticklabels(tokens, rotation=45, ha='right')
        
        ax.set_ylim(-0.1, 1.1)
        ax.set_ylabel('Attention Flow')
        ax.set_xlabel('Token Position')
        ax.set_title('Attention Flow Visualization', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"注意力流可视化保存到: {save_path}")
        
        plt.show()
    
    def generate_attention_report(self, 
                                attention_weights_dict: Dict[int, torch.Tensor],
                                tokens: Optional[List[str]] = None,
                                model_name: str = "Model") -> Dict[str, Any]:
        """
        生成注意力分析报告
        
        Args:
            attention_weights_dict: 各层注意力权重字典
            tokens: 标记列表
            model_name: 模型名称
            
        Returns:
            Dict: 完整的分析报告
        """
        self.logger.info("生成注意力分析报告...")
        
        report = {
            'metadata': {
                'model_name': model_name,
                'analysis_timestamp': datetime.now().isoformat(),
                'num_layers': len(attention_weights_dict),
                'sequence_length': None,
                'num_heads': None
            },
            'layer_analysis': {},
            'summary': {}
        }
        
        # 分析每一层
        all_entropies = []
        all_max_attentions = []
        
        for layer_idx, weights in attention_weights_dict.items():
            self.logger.info(f"分析第 {layer_idx} 层...")
            
            # 记录基本信息
            if report['metadata']['sequence_length'] is None:
                if weights.dim() == 3:
                    report['metadata']['num_heads'] = weights.shape[0]
                    report['metadata']['sequence_length'] = weights.shape[1]
                else:
                    report['metadata']['sequence_length'] = weights.shape[0]
            
            # 分析注意力模式
            layer_analysis = self.analyze_attention_patterns(weights, tokens)
            report['layer_analysis'][f'layer_{layer_idx}'] = layer_analysis
            
            # 收集统计信息
            all_entropies.extend(layer_analysis['entropy']['values'])
            all_max_attentions.extend(layer_analysis['max_attention']['values'])
        
        # 生成总结
        report['summary'] = {
            'overall_entropy': {
                'mean': float(np.mean(all_entropies)),
                'std': float(np.std(all_entropies)),
                'description': '所有层的平均注意力熵'
            },
            'overall_max_attention': {
                'mean': float(np.mean(all_max_attentions)),
                'std': float(np.std(all_max_attentions)),
                'description': '所有层的平均最大注意力权重'
            },
            'attention_diversity': float(np.std([
                layer_data['entropy']['mean'] 
                for layer_data in report['layer_analysis'].values()
            ])),
            'layer_consistency': float(1.0 - np.std([
                layer_data['max_attention']['mean'] 
                for layer_data in report['layer_analysis'].values()
            ]))
        }
        
        # 保存报告
        report_path = self.output_dir / f"{model_name}_attention_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"注意力分析报告保存到: {report_path}")
        return report
    
    def run_full_analysis(self, 
                         model: nn.Module,
                         input_tensor: torch.Tensor,
                         tokens: Optional[List[str]] = None,
                         model_name: str = "Model",
                         layer_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        运行完整的注意力分析
        
        Args:
            model: 神经网络模型
            input_tensor: 输入张量
            tokens: 标记列表
            model_name: 模型名称
            layer_indices: 要分析的层索引列表
            
        Returns:
            Dict: 完整的分析报告
        """
        self.logger.info(f"开始对模型 '{model_name}' 进行完整注意力分析")
        
        try:
            # 1. 提取注意力权重
            attention_weights_dict = self.extract_attention_weights(
                model, input_tensor, layer_indices
            )
            
            if not attention_weights_dict:
                self.logger.warning("未能提取到注意力权重，请检查模型结构")
                return {}
            
            # 2. 生成可视化
            for layer_idx, weights in attention_weights_dict.items():
                # 可视化注意力头
                self.visualize_attention_heads(
                    weights, 
                    tokens=tokens,
                    layer_name=f"Layer {layer_idx}",
                    save_path=self.output_dir / f"layer_{layer_idx}_heads.png"
                )
                
                # 可视化注意力流
                self.visualize_attention_flow(
                    weights,
                    tokens=tokens,
                    save_path=self.output_dir / f"layer_{layer_idx}_flow.png"
                )
            
            # 3. 生成分析报告
            report = self.generate_attention_report(
                attention_weights_dict, tokens, model_name
            )
            
            self.logger.info(f"✅ 注意力分析完成！结果保存在: {self.output_dir}")
            return report
            
        except Exception as e:
            self.logger.error(f"❌ 注意力分析过程中出现错误: {e}")
            raise


# 便捷函数
def analyze_model_attention(model: nn.Module, 
                          input_tensor: torch.Tensor,
                          output_dir: Union[str, Path],
                          tokens: Optional[List[str]] = None,
                          model_name: str = "Model") -> Dict[str, Any]:
    """
    便捷函数：分析模型注意力机制
    
    Args:
        model: 神经网络模型
        input_tensor: 输入张量
        output_dir: 输出目录
        tokens: 标记列表
        model_name: 模型名称
        
    Returns:
        Dict: 分析报告
    """
    analyzer = AttentionAnalyzer(output_dir)
    return analyzer.run_full_analysis(model, input_tensor, tokens, model_name)


if __name__ == "__main__":
    # 示例用法
    print("AttentionAnalyzer 示例")
    
    # 创建示例数据
    seq_len = 10
    num_heads = 8
    attention_weights = torch.randn(num_heads, seq_len, seq_len)
    attention_weights = torch.softmax(attention_weights, dim=-1)
    
    tokens = [f"token_{i}" for i in range(seq_len)]
    
    # 创建分析器
    analyzer = AttentionAnalyzer("output/attention_analysis")
    
    # 可视化注意力头
    analyzer.visualize_attention_heads(
        attention_weights, 
        tokens=tokens,
        save_path=Path("output/attention_analysis/example_heads.png")
    )
    
    # 分析注意力模式
    analysis = analyzer.analyze_attention_patterns(attention_weights, tokens)
    print("注意力模式分析结果:")
    print(f"平均熵: {analysis['entropy']['mean']:.4f}")
    print(f"平均最大注意力: {analysis['max_attention']['mean']:.4f}")