# 注意力机制可视化工具
# 专门用于可视化Transformer模型的注意力权重

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
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


class AttentionVisualizer:
    """注意力机制可视化工具"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置matplotlib样式
        plt.style.use('default')
        sns.set_palette("husl")
    
    def extract_attention_weights(self, model: nn.Module, 
                                input_tensor: torch.Tensor,
                                layer_indices: List[int] = None) -> Dict[int, torch.Tensor]:
        """
        提取模型各层的注意力权重
        
        Args:
            model: 模型
            input_tensor: 输入张量
            layer_indices: 要提取的层索引列表
            
        Returns:
            Dict[int, torch.Tensor]: 层索引到注意力权重的映射
        """
        attention_weights = {}
        hooks = []
        
        def attention_hook(module, input, output, layer_idx):
            # 对于Transformer层，注意力权重通常在forward方法中计算
            # 这里假设模型有attention_weights属性
            if hasattr(module, 'attention_weights'):
                attention_weights[layer_idx] = module.attention_weights.detach()
            elif hasattr(module, 'self_attn') and hasattr(module.self_attn, 'attention_weights'):
                attention_weights[layer_idx] = module.self_attn.attention_weights.detach()
        
        # 注册钩子
        layer_count = 0
        for name, module in model.named_modules():
            if isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
                if layer_indices is None or layer_count in layer_indices:
                    hook = module.register_forward_hook(
                        lambda m, i, o, idx=layer_count: attention_hook(m, i, o, idx)
                    )
                    hooks.append(hook)
                layer_count += 1
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            _ = model(input_tensor)
        
        # 清理钩子
        for hook in hooks:
            hook.remove()
        
        return attention_weights
    
    def visualize_attention_heads(self, attention_weights: torch.Tensor,
                                 tokens: List[str] = None,
                                 head_indices: List[int] = None,
                                 layer_name: str = "Attention",
                                 save_path: Optional[Path] = None) -> None:
        """
        可视化多头注意力的各个头
        
        Args:
            attention_weights: 注意力权重 (num_heads, seq_len, seq_len)
            tokens: 标记列表
            head_indices: 要可视化的头索引列表
            layer_name: 层名称
            save_path: 保存路径
        """
        if attention_weights.dim() == 3:
            num_heads, seq_len, _ = attention_weights.shape
        else:
            # 如果是2D，假设是单头
            attention_weights = attention_weights.unsqueeze(0)
            num_heads, seq_len, _ = attention_weights.shape
        
        if head_indices is None:
            head_indices = list(range(min(num_heads, 8)))  # 最多显示8个头
        
        # 计算子图布局
        num_heads_to_show = len(head_indices)
        cols = min(4, num_heads_to_show)
        rows = (num_heads_to_show + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        else:
            axes = axes.reshape(rows, cols)
        
        for i, head_idx in enumerate(head_indices):
            row = i // cols
            col = i % cols
            
            if rows == 1:
                ax = axes[col] if cols > 1 else axes
            elif cols == 1:
                ax = axes[row] if rows > 1 else axes
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
            
            ax.set_title(f'Head {head_idx}', fontsize=10)
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
        
        plt.suptitle(f'{layer_name} - Multi-Head Attention', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"多头注意力可视化已保存到: {save_path}")
        
        plt.show()
    
    def visualize_attention_flow(self, attention_weights: torch.Tensor,
                               tokens: List[str] = None,
                               threshold: float = 0.1,
                               save_path: Optional[Path] = None) -> None:
        """
        可视化注意力流（注意力强度图）
        
        Args:
            attention_weights: 注意力权重 (seq_len, seq_len)
            tokens: 标记列表
            threshold: 注意力阈值
            save_path: 保存路径
        """
        if attention_weights.dim() == 3:
            # 如果是多头，取平均
            attention_weights = attention_weights.mean(dim=0)
        
        attention_weights = attention_weights.cpu().numpy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 原始注意力热力图
        im1 = ax1.imshow(attention_weights, cmap='Blues', aspect='auto')
        if tokens:
            ax1.set_xticks(range(len(tokens)))
            ax1.set_yticks(range(len(tokens)))
            ax1.set_xticklabels(tokens, rotation=45, ha='right')
            ax1.set_yticklabels(tokens)
        ax1.set_title('Attention Weights')
        ax1.set_xlabel('Key Position')
        ax1.set_ylabel('Query Position')
        plt.colorbar(im1, ax=ax1)
        
        # 注意力流图
        seq_len = attention_weights.shape[0]
        y_pos = np.arange(seq_len)
        
        # 计算每个位置的注意力强度
        attention_strength = attention_weights.sum(axis=1)
        
        # 绘制注意力强度条形图
        bars = ax2.barh(y_pos, attention_strength, color='skyblue', alpha=0.7)
        
        # 高亮超过阈值的注意力
        high_attention = attention_strength > threshold
        for i, (bar, is_high) in enumerate(zip(bars, high_attention)):
            if is_high:
                bar.set_color('red')
                bar.set_alpha(0.8)
        
        if tokens:
            ax2.set_yticks(range(len(tokens)))
            ax2.set_yticklabels(tokens)
        ax2.set_xlabel('Attention Strength')
        ax2.set_title('Attention Flow')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"注意力流可视化已保存到: {save_path}")
        
        plt.show()
    
    def create_attention_animation(self, attention_weights: torch.Tensor,
                                 tokens: List[str] = None,
                                 save_path: Optional[Path] = None) -> None:
        """
        创建注意力动画（需要matplotlib动画支持）
        
        Args:
            attention_weights: 注意力权重 (seq_len, seq_len)
            tokens: 标记列表
            save_path: 保存路径
        """
        try:
            from matplotlib.animation import FuncAnimation
        except ImportError:
            print("警告: matplotlib动画功能不可用")
            return
        
        if attention_weights.dim() == 3:
            attention_weights = attention_weights.mean(dim=0)
        
        attention_weights = attention_weights.cpu().numpy()
        seq_len = attention_weights.shape[0]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def animate(frame):
            ax.clear()
            
            # 获取当前查询位置的注意力
            query_attention = attention_weights[frame]
            
            # 绘制注意力权重
            im = ax.imshow(attention_weights, cmap='Blues', alpha=0.3, aspect='auto')
            
            # 高亮当前查询位置
            ax.axhline(y=frame, color='red', linewidth=2)
            ax.axvline(x=frame, color='red', linewidth=2)
            
            # 绘制当前查询的注意力分布
            y_pos = np.arange(seq_len)
            ax.barh(y_pos, query_attention, alpha=0.7, color='orange')
            
            if tokens:
                ax.set_xticks(range(len(tokens)))
                ax.set_yticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=45, ha='right')
                ax.set_yticklabels(tokens)
            
            ax.set_title(f'Attention for Query Position {frame}')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
        
        # 创建动画
        anim = FuncAnimation(fig, animate, frames=seq_len, interval=500, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=2)
            print(f"注意力动画已保存到: {save_path}")
        else:
            plt.show()
    
    def create_interactive_attention_plot(self, attention_weights: torch.Tensor,
                                        tokens: List[str] = None,
                                        save_path: Optional[Path] = None) -> None:
        """
        创建交互式注意力图（使用Plotly）
        
        Args:
            attention_weights: 注意力权重 (seq_len, seq_len)
            tokens: 标记列表
            save_path: 保存路径
        """
        if not PLOTLY_AVAILABLE:
            print("警告: Plotly库未安装，无法创建交互式图表")
            return
        
        if attention_weights.dim() == 3:
            attention_weights = attention_weights.mean(dim=0)
        
        attention_weights = attention_weights.cpu().numpy()
        
        # 创建热力图
        fig = go.Figure(data=go.Heatmap(
            z=attention_weights,
            x=tokens if tokens else list(range(attention_weights.shape[1])),
            y=tokens if tokens else list(range(attention_weights.shape[0])),
            colorscale='Blues',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Interactive Attention Heatmap',
            xaxis_title='Key Position',
            yaxis_title='Query Position',
            width=800,
            height=600
        )
        
        if save_path:
            pyo.plot(fig, filename=str(save_path), auto_open=False)
            print(f"交互式注意力图已保存到: {save_path}")
        else:
            fig.show()
    
    def analyze_attention_patterns(self, attention_weights: torch.Tensor,
                                 tokens: List[str] = None) -> Dict:
        """
        分析注意力模式
        
        Args:
            attention_weights: 注意力权重 (seq_len, seq_len)
            tokens: 标记列表
            
        Returns:
            Dict: 分析结果
        """
        if attention_weights.dim() == 3:
            attention_weights = attention_weights.mean(dim=0)
        
        attention_weights = attention_weights.cpu().numpy()
        
        analysis = {}
        
        # 计算注意力熵
        attention_probs = attention_weights / attention_weights.sum(axis=1, keepdims=True)
        entropy = -np.sum(attention_probs * np.log(attention_probs + 1e-8), axis=1)
        analysis['entropy'] = entropy.tolist()
        
        # 计算注意力集中度
        max_attention = np.max(attention_weights, axis=1)
        analysis['max_attention'] = max_attention.tolist()
        
        # 计算注意力分散度
        attention_std = np.std(attention_weights, axis=1)
        analysis['attention_std'] = attention_std.tolist()
        
        # 找到最受关注的token
        most_attended = np.argmax(attention_weights, axis=1)
        analysis['most_attended'] = most_attended.tolist()
        
        if tokens:
            analysis['tokens'] = tokens
        
        return analysis


def visualize_model_attention(model: nn.Module, 
                            input_tensor: torch.Tensor,
                            output_dir: Path,
                            tokens: List[str] = None) -> None:
    """
    可视化模型的注意力机制
    
    Args:
        model: 模型
        input_tensor: 输入张量
        output_dir: 输出目录
        tokens: 标记列表
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = AttentionVisualizer(output_dir)
    
    # 提取注意力权重
    attention_weights = visualizer.extract_attention_weights(model, input_tensor)
    
    if not attention_weights:
        print("警告: 未能提取到注意力权重")
        return
    
    # 可视化每一层的注意力
    for layer_idx, weights in attention_weights.items():
        print(f"可视化第 {layer_idx} 层的注意力...")
        
        # 多头注意力可视化
        visualizer.visualize_attention_heads(
            weights, tokens=tokens,
            save_path=output_dir / f"layer_{layer_idx}_attention_heads.png"
        )
        
        # 注意力流可视化
        visualizer.visualize_attention_flow(
            weights, tokens=tokens,
            save_path=output_dir / f"layer_{layer_idx}_attention_flow.png"
        )
        
        # 交互式注意力图
        visualizer.create_interactive_attention_plot(
            weights, tokens=tokens,
            save_path=output_dir / f"layer_{layer_idx}_interactive.html"
        )
        
        # 分析注意力模式
        analysis = visualizer.analyze_attention_patterns(weights, tokens)
        
        # 保存分析结果
        import json
        with open(output_dir / f"layer_{layer_idx}_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
    
    print(f"注意力可视化完成，结果保存在: {output_dir}")


if __name__ == "__main__":
    # 示例用法
    output_dir = Path("output/attention_visualization")
    visualizer = AttentionVisualizer(output_dir)
    
    # 创建示例注意力权重
    seq_len = 10
    num_heads = 8
    attention_weights = torch.randn(num_heads, seq_len, seq_len)
    attention_weights = torch.softmax(attention_weights, dim=-1)
    
    tokens = [f"token_{i}" for i in range(seq_len)]
    
    # 可视化多头注意力
    visualizer.visualize_attention_heads(
        attention_weights, tokens=tokens,
        save_path=output_dir / "example_attention_heads.png"
    )
    
    # 可视化注意力流
    visualizer.visualize_attention_flow(
        attention_weights, tokens=tokens,
        save_path=output_dir / "example_attention_flow.png"
    )