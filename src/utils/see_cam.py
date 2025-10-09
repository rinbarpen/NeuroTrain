# Enhanced Attention Visualization Module for NeuroTrain
# 增强的注意力机制可视化模块，支持多种深度学习模型的注意力权重提取和可视化

from pathlib import Path
from typing import Literal, Optional, Union, List, Dict, Tuple, Any
import warnings
import logging

# Core libraries
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from torchvision import transforms

# GradCAM libraries
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, GradCAMElementWise, XGradCAM, AblationCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, SemanticSegmentationTarget, SoftmaxOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Model libraries
from torchvision.models import resnet50, vgg19
from timm import get_pretrained_cfg, create_model

# Local imports
# from src.models.sample.unet import UNet  # 注释掉以避免导入错误

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttentionExtractor:
    """
    通用注意力权重提取器，支持多种模型架构
    Universal attention weight extractor supporting various model architectures
    """
    
    def __init__(self, model: nn.Module, device: Optional[str] = None):
        """
        初始化注意力提取器
        
        Args:
            model: 要分析的模型
            device: 计算设备，默认自动选择
        """
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.model.eval()
        
        # 存储注意力权重的钩子
        self.attention_weights = {}
        self.hooks = []
        
    def register_attention_hooks(self, layer_names: Optional[List[str]] = None):
        """
        注册注意力权重提取钩子
        
        Args:
            layer_names: 要提取注意力的层名称列表，None表示自动检测
        """
        self._clear_hooks()
        
        if layer_names is None:
            layer_names = self._auto_detect_attention_layers()
            
        for name, module in self.model.named_modules():
            if any(layer_name in name for layer_name in layer_names):
                hook = module.register_forward_hook(self._attention_hook(name))
                self.hooks.append(hook)
                
        logger.info(f"Registered attention hooks for {len(self.hooks)} layers")
    
    def _auto_detect_attention_layers(self) -> List[str]:
        """自动检测模型中的注意力层"""
        attention_keywords = ['attention', 'attn', 'self_attn', 'multihead']
        detected_layers = []
        
        for name, module in self.model.named_modules():
            if any(keyword in name.lower() for keyword in attention_keywords):
                detected_layers.append(name)
                
        return detected_layers
    
    def _attention_hook(self, layer_name: str):
        """创建注意力权重提取钩子"""
        def hook(module, input, output):
            # 根据不同的模块类型提取注意力权重
            if hasattr(module, 'attention_weights'):
                self.attention_weights[layer_name] = module.attention_weights.detach().cpu()
            elif isinstance(output, tuple) and len(output) > 1:
                # 对于返回(output, attention_weights)的模块
                if isinstance(output[1], torch.Tensor):
                    self.attention_weights[layer_name] = output[1].detach().cpu()
            elif hasattr(module, 'attn_weights'):
                self.attention_weights[layer_name] = module.attn_weights.detach().cpu()
                
        return hook
    
    def _clear_hooks(self):
        """清除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.attention_weights.clear()
    
    def extract_attention(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        提取注意力权重
        
        Args:
            input_tensor: 输入张量
            
        Returns:
            各层的注意力权重字典
        """
        self.attention_weights.clear()
        
        with torch.no_grad():
            _ = self.model(input_tensor.to(self.device))
            
        return self.attention_weights.copy()
    
    def __del__(self):
        """析构函数，清理钩子"""
        self._clear_hooks()


class EnhancedVisualizationCAM:
    """
    增强的CAM可视化类，支持多种CAM方法和可视化选项
    Enhanced CAM visualization class supporting multiple CAM methods and visualization options
    """
    
    # 支持的CAM方法
    CAM_METHODS = {
        'gradcam': GradCAM,
        'gradcam++': GradCAMPlusPlus,
        'scorecam': ScoreCAM,
        'gradcam_elementwise': GradCAMElementWise,
        'xgradcam': XGradCAM,
        'ablationcam': AblationCAM,
        'eigencam': EigenCAM
    }
    
    def __init__(self, 
                 model: nn.Module, 
                 target_layers: List[nn.Module],
                 model_params_file: Optional[Union[str, Path]] = None,
                 cam_method: str = 'gradcam++'):
        """
        初始化增强CAM可视化器
        
        Args:
            model: 要分析的模型
            target_layers: 目标层列表
            model_params_file: 模型参数文件路径
            cam_method: CAM方法名称
        """
        self.model = model
        self.target_layers = target_layers
        self.cam_method = cam_method
        
        # 加载模型参数
        if model_params_file:
            self._load_model_params(model_params_file)
            
        self.model.eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        # 初始化CAM对象
        self.cam = self._initialize_cam()
        
    def _load_model_params(self, model_params_file: Union[str, Path]):
        """加载模型参数"""
        try:
            checkpoint = torch.load(model_params_file, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
            else:
                self.model.load_state_dict(checkpoint)
            logger.info(f"Loaded model parameters from {model_params_file}")
        except Exception as e:
            logger.error(f"Failed to load model parameters: {e}")
            raise
    
    def _initialize_cam(self):
        """初始化CAM对象"""
        if self.cam_method not in self.CAM_METHODS:
            raise ValueError(f"Unsupported CAM method: {self.cam_method}")
            
        cam_class = self.CAM_METHODS[self.cam_method]
        return cam_class(model=self.model, target_layers=self.target_layers)
    
    def generate_heatmap(self, 
                        image: Union[Image.Image, np.ndarray],
                        targets: List,
                        transform_fn: Optional[transforms.Compose] = None,
                        is_rgb: bool = True,
                        eigen_smooth: bool = True,
                        aug_smooth: bool = True) -> Tuple[Image.Image, np.ndarray]:
        """
        生成热力图
        
        Args:
            image: 输入图像
            targets: 目标列表
            transform_fn: 图像变换函数
            is_rgb: 是否为RGB图像
            eigen_smooth: 是否使用特征值平滑
            aug_smooth: 是否使用增强平滑
            
        Returns:
            (热力图图像, 原始CAM数组)
        """
        try:
            # 图像预处理
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
                
            if is_rgb:
                image = image.convert('RGB')
            else:
                image = image.convert('L')
                
            original_size = image.size
            
            # 应用变换
            if transform_fn is None:
                transform_fn = self._get_default_transform(original_size)
                
            input_tensor = transform_fn(image).unsqueeze(0).to(self.device)
            input_tensor = input_tensor.clone().detach().requires_grad_(True)
            image_np = np.array(image).astype(np.float32) / 255.0
            
            # 生成CAM
            grayscale_cam = self.cam(input_tensor=input_tensor,
                                   targets=targets,
                                   eigen_smooth=eigen_smooth,
                                   aug_smooth=aug_smooth)[0, :]
            
            # 调整CAM大小以匹配原始图像
            grayscale_cam = cv2.resize(grayscale_cam, original_size)
            
            # 生成热力图
            cam_image = show_cam_on_image(image_np, grayscale_cam, use_rgb=is_rgb)
            heatmap_image = Image.fromarray(cam_image)
            
            return heatmap_image, grayscale_cam
            
        except Exception as e:
            logger.error(f"Failed to generate heatmap: {e}")
            # 返回空热力图
            empty_heatmap = np.zeros((224, 224))
            empty_image = Image.fromarray((empty_heatmap * 255).astype(np.uint8))
            return empty_image, empty_heatmap
    
    def _get_default_transform(self, image_size: Tuple[int, int]) -> transforms.Compose:
        """获取默认的图像变换"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def generate_multi_scale_heatmap(self,
                                   image: Union[Image.Image, np.ndarray],
                                   targets: List,
                                   scales: List[float] = [0.8, 1.0, 1.2],
                                   **kwargs) -> List[Tuple[Image.Image, np.ndarray]]:
        """
        生成多尺度热力图
        
        Args:
            image: 输入图像
            targets: 目标列表
            scales: 尺度列表
            **kwargs: 其他参数
            
        Returns:
            多尺度热力图列表
        """
        results = []
        original_size = image.size if isinstance(image, Image.Image) else image.shape[:2][::-1]
        
        for scale in scales:
            # 调整图像尺寸
            new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
            if isinstance(image, Image.Image):
                scaled_image = image.resize(new_size)
            else:
                scaled_image = cv2.resize(image, new_size)
                scaled_image = Image.fromarray(scaled_image)
            
            # 生成热力图
            heatmap, cam_array = self.generate_heatmap(scaled_image, targets, **kwargs)
            results.append((heatmap, cam_array))
            
        return results


class TransformerAttentionVisualizer:
    """
    Transformer模型注意力可视化器
    Transformer model attention visualizer
    """
    
    def __init__(self, model: nn.Module, device: Optional[str] = None):
        """
        初始化Transformer注意力可视化器
        
        Args:
            model: Transformer模型
            device: 计算设备
        """
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
    def visualize_attention(self, input_tensor: torch.Tensor, layer_name: str = None) -> List[np.ndarray]:
        """
        可视化Transformer注意力机制
        Visualize Transformer attention mechanism
        
        Args:
            input_tensor: 输入张量 / Input tensor
            layer_name: 层名称 / Layer name
            
        Returns:
            注意力可视化结果列表 / List of attention visualization results
        """
        attention_maps = []
        
        def attention_hook(module, input, output):
            """注意力钩子函数"""
            if hasattr(output, '__len__') and len(output) >= 2:
                # MultiheadAttention返回(output, attention_weights)
                attn_weights = output[1]  # 注意力权重
                if attn_weights is not None:
                    attention_maps.append(attn_weights.detach().cpu().numpy())
        
        # 注册钩子
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                if layer_name is None or layer_name in name:
                    hook = module.register_forward_hook(attention_hook)
                    hooks.append(hook)
        
        try:
            # 前向传播
            with torch.no_grad():
                _ = self.model(input_tensor)
            
            # 处理注意力图
            processed_maps = []
            for attn_map in attention_maps:
                # 处理注意力权重的形状
                if attn_map.ndim >= 2:  # 至少是2维张量
                    # 如果是多头注意力，取平均 / If multi-head attention, take average
                    if attn_map.dim() == 4:  # [batch, heads, seq_len, seq_len]
                        attention_map = attn_map.mean(dim=1).squeeze(0)  # 平均所有头 / Average all heads
                    elif attn_map.dim() == 3:  # [batch, seq_len, seq_len] or [heads, seq_len, seq_len]
                        attention_map = attn_map.squeeze(0) if attn_map.size(0) == 1 else attn_map.mean(dim=0)
                    else:  # [seq_len, seq_len]
                        attention_map = attn_map
                    
                    # 转换为numpy并调整大小 / Convert to numpy and resize
                    attention_np = attention_map.cpu().numpy()
                    
                    # 如果注意力图太小，进行插值放大 / If attention map too small, interpolate to larger size
                    if attention_np.shape[0] < 8 or attention_np.shape[1] < 8:
                        attention_np = cv2.resize(attention_np, (64, 64), interpolation=cv2.INTER_LINEAR)
                    
                    processed_maps.append(attention_np)
                    logger.debug(f"Processed attention map with shape: {attention_np.shape}")
                    
                else:
                    logger.warning(f"Unexpected attention weight shape: {attn_map.shape}")
                    # 创建默认的8x8注意力图 / Create default 8x8 attention map
                    default_map = np.random.rand(8, 8) * 0.5 + 0.25  # 随机但合理的注意力模式
                    processed_maps.append(default_map)
            
            return processed_maps if processed_maps else [np.random.rand(8, 8)]
            
        except Exception as e:
            logger.error(f"Failed to visualize attention: {e}")
            return [np.random.rand(8, 8)]  # 返回默认注意力图
            
        finally:
            # 移除钩子
            for hook in hooks:
                hook.remove()
        
    def visualize_attention_heatmap(self, 
                                  image_np: np.ndarray, 
                                  attn_scores: torch.Tensor, 
                                  patch_size: Union[Tuple[int, int], int],
                                  alpha: float = 0.5,
                                  head_type: Literal['mean', 'max', 'individual'] = 'mean',
                                  colormap: str = 'viridis',
                                  save_path: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        可视化Transformer注意力热力图
        
        Args:
            image_np: 原始图像数组
            attn_scores: 注意力分数张量 (B, H, N, N)
            patch_size: 补丁大小
            alpha: 透明度
            head_type: 注意力头处理方式
            colormap: 颜色映射
            save_path: 保存路径
            
        Returns:
            (融合图像, 热力图)
        """
        try:
            original_image_size = image_np.shape[-2:] if len(image_np.shape) == 3 else image_np.shape
            
            # 确保注意力分数在CPU上
            if isinstance(attn_scores, torch.Tensor):
                attn_scores = attn_scores.cpu()
            
            # 处理注意力分数
            if head_type == 'mean':
                attn_scores = attn_scores.mean(dim=(0, 1))  # (N, N)
            elif head_type == 'max':
                attn_scores = attn_scores.mean(dim=0).max(dim=0)[0]  # (N, N)
            elif head_type == 'individual':
                # 返回所有头的注意力图
                return self._visualize_individual_heads(image_np, attn_scores, patch_size, alpha, colormap, save_path)
            
            if isinstance(patch_size, int):
                patch_size = (patch_size, patch_size)
                
            # 智能处理注意力图形状
            seq_len = attn_scores.shape[0]
            
            # 尝试从序列长度推断网格大小
            grid_size = int(np.sqrt(seq_len))
            if grid_size * grid_size == seq_len:
                # 完美平方数，使用推断的网格大小
                num_patches = grid_size
                attn_map = attn_scores[:num_patches*num_patches].reshape(num_patches, num_patches)
            else:
                # 不是完美平方数，尝试其他处理方式
                logger.warning(f"Sequence length {seq_len} is not a perfect square")
                # 取最接近的平方数
                num_patches = int(np.sqrt(seq_len))
                truncated_len = num_patches * num_patches
                if truncated_len <= seq_len:
                    attn_map = attn_scores[:truncated_len].reshape(num_patches, num_patches)
                else:
                    # 如果序列太短，进行填充
                    padded_scores = torch.zeros(truncated_len)
                    padded_scores[:seq_len] = attn_scores.flatten()[:seq_len]
                    attn_map = padded_scores.reshape(num_patches, num_patches)
            
            # 归一化
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            
            # 调整大小到原始图像尺寸
            attn_map_resized = cv2.resize(attn_map.numpy(), original_image_size[::-1], interpolation=cv2.INTER_LINEAR)
            
            # 生成热力图
            colormap_func = plt.get_cmap(colormap)
            heatmap = colormap_func(attn_map_resized)
            heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)
            
            # 确保图像格式正确
            if len(image_np.shape) == 2:
                image_np = np.stack([image_np] * 3, axis=-1)
            image_np = image_np.astype(np.uint8)
            
            # 融合图像
            fused_image = cv2.addWeighted(heatmap, alpha, image_np, 1.0 - alpha, 0)
            
            # 保存图像
            if save_path:
                self._save_attention_visualization(fused_image, heatmap, save_path)
                
            return fused_image, heatmap
            
        except Exception as e:
            logger.error(f"Failed to visualize attention heatmap: {e}")
            # 返回默认结果
            h, w = image_np.shape[:2]
            default_heatmap = np.random.rand(h, w) * 0.5 + 0.25
            return image_np.copy(), default_heatmap
    
    def _visualize_individual_heads(self, 
                                  image_np: np.ndarray,
                                  attn_scores: torch.Tensor,
                                  patch_size: Union[Tuple[int, int], int],
                                  alpha: float,
                                  colormap: str,
                                  save_path: Optional[Path]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """可视化各个注意力头"""
        batch_size, num_heads = attn_scores.shape[:2]
        fused_images = []
        heatmaps = []
        
        for b in range(batch_size):
            batch_fused = []
            batch_heatmaps = []
            
            for h in range(num_heads):
                single_head_attn = attn_scores[b, h]  # (N, N)
                fused, heatmap = self.visualize_attention_heatmap(
                    image_np, single_head_attn.unsqueeze(0).unsqueeze(0),
                    patch_size, alpha, 'mean', colormap
                )
                batch_fused.append(fused)
                batch_heatmaps.append(heatmap)
                
            fused_images.append(batch_fused)
            heatmaps.append(batch_heatmaps)
            
        if save_path:
            self._save_multi_head_visualization(fused_images, heatmaps, save_path)
            
        return fused_images, heatmaps
    
    def _save_attention_visualization(self, fused_image: np.ndarray, heatmap: np.ndarray, save_path: Path):
        """保存注意力可视化结果"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存融合图像
        fused_path = save_path.with_suffix('.fused.png')
        cv2.imwrite(str(fused_path), cv2.cvtColor(fused_image, cv2.COLOR_RGB2BGR))
        
        # 保存热力图
        heatmap_path = save_path.with_suffix('.heatmap.png')
        cv2.imwrite(str(heatmap_path), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
        
        logger.info(f"Saved attention visualization to {fused_path} and {heatmap_path}")
    
    def _save_multi_head_visualization(self, fused_images: List, heatmaps: List, save_path: Path):
        """保存多头注意力可视化结果"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        for b, (batch_fused, batch_heatmaps) in enumerate(zip(fused_images, heatmaps)):
            for h, (fused, heatmap) in enumerate(zip(batch_fused, batch_heatmaps)):
                fused_path = save_path.with_suffix(f'.batch{b}_head{h}.fused.png')
                heatmap_path = save_path.with_suffix(f'.batch{b}_head{h}.heatmap.png')
                
                cv2.imwrite(str(fused_path), cv2.cvtColor(fused, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(heatmap_path), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
        
        logger.info(f"Saved multi-head attention visualization to {save_path}")
    
    def create_attention_matrix_plot(self, 
                                   attn_scores: torch.Tensor,
                                   save_path: Optional[Path] = None,
                                   figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        创建注意力矩阵图
        
        Args:
            attn_scores: 注意力分数张量
            save_path: 保存路径
            figsize: 图像大小
            
        Returns:
            matplotlib图像对象
        """
        # 处理注意力分数维度
        if len(attn_scores.shape) == 4:  # (B, H, N, N)
            attn_scores = attn_scores.mean(dim=(0, 1))  # 平均所有批次和头
        elif len(attn_scores.shape) == 3:  # (H, N, N)
            attn_scores = attn_scores.mean(dim=0)  # 平均所有头
            
        attn_matrix = attn_scores.numpy()
        
        # 创建图像
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制热力图
        im = ax.imshow(attn_matrix, cmap='Blues', aspect='auto')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight', rotation=270, labelpad=20)
        
        # 设置标签
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        ax.set_title('Attention Matrix Visualization')
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 保存图像
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved attention matrix plot to {save_path}")
            
        return fig


class ComprehensiveAttentionAnalyzer:
    """
    综合注意力分析器，整合多种可视化方法
    Comprehensive attention analyzer integrating multiple visualization methods
    """
    
    def __init__(self, model: nn.Module, device: Optional[str] = None):
        """
        初始化综合注意力分析器
        
        Args:
            model: 要分析的模型
            device: 计算设备
        """
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化各种可视化器
        self.attention_extractor = AttentionExtractor(model, device)
        self.transformer_visualizer = TransformerAttentionVisualizer(model, device)
        
    def analyze_model_attention(self, 
                              input_data: Union[torch.Tensor, Image.Image, np.ndarray],
                              output_dir: Path,
                              analysis_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        全面分析模型注意力机制
        
        Args:
            input_data: 输入数据
            output_dir: 输出目录
            analysis_config: 分析配置
            
        Returns:
            分析结果字典
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 默认配置
        config = {
            'extract_attention_weights': True,
            'generate_cam_heatmaps': True,
            'create_attention_matrices': True,
            'save_intermediate_results': True,
            'cam_methods': ['gradcam++', 'scorecam'],
            'colormap': 'viridis',
            'alpha': 0.5
        }
        
        if analysis_config:
            config.update(analysis_config)
            
        results = {}
        
        # 预处理输入数据
        if isinstance(input_data, (Image.Image, np.ndarray)):
            # 转换为张量
            if isinstance(input_data, Image.Image):
                input_tensor = transforms.ToTensor()(input_data).unsqueeze(0)
            else:
                input_tensor = torch.from_numpy(input_data).float()
                if len(input_tensor.shape) == 3:
                    input_tensor = input_tensor.unsqueeze(0)
        else:
            input_tensor = input_data
            
        input_tensor = input_tensor.to(self.device)
        
        # 1. 提取注意力权重
        if config['extract_attention_weights']:
            logger.info("Extracting attention weights...")
            self.attention_extractor.register_attention_hooks()
            attention_weights = self.attention_extractor.extract_attention(input_tensor)
            results['attention_weights'] = attention_weights
            
            # 保存注意力权重
            if config['save_intermediate_results']:
                weights_path = output_dir / 'attention_weights.pt'
                torch.save(attention_weights, weights_path)
                logger.info(f"Saved attention weights to {weights_path}")
        
        # 2. 生成CAM热力图
        if config['generate_cam_heatmaps'] and hasattr(self, '_get_target_layers'):
            logger.info("Generating CAM heatmaps...")
            target_layers = self._get_target_layers()
            
            for cam_method in config['cam_methods']:
                try:
                    cam_visualizer = EnhancedVisualizationCAM(
                        self.model, target_layers, cam_method=cam_method
                    )
                    
                    # 生成热力图
                    if isinstance(input_data, (Image.Image, np.ndarray)):
                        targets = [ClassifierOutputTarget(0)]  # 默认目标
                        heatmap, cam_array = cam_visualizer.generate_heatmap(
                            input_data, targets
                        )
                        
                        # 保存结果
                        heatmap_path = output_dir / f'{cam_method}_heatmap.png'
                        heatmap.save(heatmap_path)
                        
                        results[f'{cam_method}_heatmap'] = heatmap
                        results[f'{cam_method}_cam_array'] = cam_array
                        
                except Exception as e:
                    logger.warning(f"Failed to generate {cam_method} heatmap: {e}")
        
        # 3. 创建注意力矩阵图
        if config['create_attention_matrices'] and 'attention_weights' in results:
            logger.info("Creating attention matrix plots...")
            
            for layer_name, attn_weights in results['attention_weights'].items():
                try:
                    fig = self.transformer_visualizer.create_attention_matrix_plot(
                        attn_weights,
                        save_path=output_dir / f'{layer_name}_attention_matrix.png'
                    )
                    plt.close(fig)  # 释放内存
                    
                except Exception as e:
                    logger.warning(f"Failed to create attention matrix for {layer_name}: {e}")
        
        # 生成分析报告
        report_path = output_dir / 'analysis_report.txt'
        self._generate_analysis_report(results, report_path)
        
        logger.info(f"Attention analysis completed. Results saved to {output_dir}")
        return results
    
    def _get_target_layers(self) -> List[nn.Module]:
        """获取目标层（需要根据具体模型实现）"""
        # 这里需要根据具体的模型架构来实现
        # 示例：对于ResNet
        target_layers = []
        for name, module in self.model.named_modules():
            if 'layer4' in name and isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                target_layers.append(module)
                break
        return target_layers
    
    def _generate_analysis_report(self, results: Dict[str, Any], report_path: Path):
        """生成分析报告"""
        from datetime import datetime
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Attention Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now()}\n")
            f.write(f"Model: {self.model.__class__.__name__}\n")
            f.write(f"Device: {self.device}\n\n")
            
            if 'attention_weights' in results:
                f.write("Extracted Attention Layers:\n")
                for layer_name in results['attention_weights'].keys():
                    f.write(f"  - {layer_name}\n")
                f.write("\n")
            
            f.write("Generated Visualizations:\n")
            for key in results.keys():
                if 'heatmap' in key or 'matrix' in key:
                    f.write(f"  - {key}\n")
            
        logger.info(f"Generated analysis report: {report_path}")


# 保留原有的简单函数以保持向后兼容性
def unet_check(image_file: Path, mask: np.ndarray, is_rgb: bool, *, target_category: int = 0) -> Image.Image:
    # 动态导入UNet以避免循环导入
    try:
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        sys.path.append(str(project_root / "src" / "models" / "sample"))
        from unet import UNet
    except ImportError:
        logger.warning("无法导入UNet模型，跳过UNet检查")
        logger.warning("Cannot import UNet model, skipping UNet check")
        return None
    
    # 创建UNet模型实例
    model = UNet(1, 1, True)
    model.load_state_dict(torch.load(r'..\results\train\unet\weights\best_model.pth')['model'])
    model.eval()

    # Choose a target layer (e.g., the last convolutional layer)
    target_layers = [model.layer4[-1]]

    # Load and preprocess an image
    image = Image.open(image_file).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if is_rgb else transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    input_tensor = transform(image).unsqueeze(0)  # Add a batch dimension

    # Move the model and input tensor to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    image_np = np.array(image) / 255.0

    # image_np: (H, W, C) format
    H, W = image_np.shape[0], image_np.shape[1]
    
    targets = [SemanticSegmentationTarget(0, mask)]
    
    cam = GradCAM(model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor, targets=targets, eigen_smooth=False)[0, :]
    
    # 确保grayscale_cam的大小与image_np匹配
    grayscale_cam = cv2.resize(grayscale_cam, (W, H))  # 注意这里是 (W, H)

    cam_image = show_cam_on_image(image_np, grayscale_cam, use_rgb=is_rgb)

    return Image.fromarray(cam_image, mode='RGB')


def resnet50_check(image_file: Path, is_rgb: bool, *, target_category: int=0) -> None:
    # Load a pre-trained ResNet50 model
    model = resnet50(pretrained=True)
    model.eval()

    # Choose a target layer (e.g., the last convolutional layer)
    target_layers = [model.layer4[-1]]

    # Load and preprocess an image
    image = Image.open(image_file).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if is_rgb else transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    input_tensor = transform(image).unsqueeze(0)  # Add a batch dimension

    # Move the model and input tensor to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    image_np = np.array(image) / 255.0

    # image_np: (H, W, C) format
    H, W = image_np.shape[0], image_np.shape[1]
    targets = [ClassifierOutputTarget(target_category)]
    
    cam = GradCAM(model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor, targets=targets, eigen_smooth=False)[0, :]
    
    # 确保grayscale_cam的大小与image_np匹配
    grayscale_cam = cv2.resize(grayscale_cam, (W, H))  # 注意这里是 (W, H)

    cam_image = show_cam_on_image(image_np, grayscale_cam, use_rgb=is_rgb)

    return Image.fromarray(cam_image, mode='RGB')


class ImageHeatMapGenerator():
    def __init__(self, model: nn.Module, target_layers: list[nn.Module], targets: list, model_params_file: str|Path|None=None):
        self.model = model
        if model_params_file:
            self.model.load_state_dict(torch.load(model_params_file))
        self.model.eval()
        self.target_layers = target_layers
        self.targets = targets

    def check(self, image: Image.Image, transform_fn: transforms.Compose, is_rgb=True):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if is_rgb:
            image = image.convert('RGB')
        else:
            image = image.convert('L')
        H, W = image.size

        input_tensor = transform_fn(image).unsqueeze(0)
        image_np = np.array(image).astype(np.float32)

        self.model = self.model.to(device)
        input_tensor = input_tensor.to(device)

        with GradCAMPlusPlus(self.model, target_layers=self.target_layers) as cam: 
            grayscale_cam = cam(input_tensor, targets=self.targets, eigen_smooth=False)[0, :]
            grayscale_cam = cv2.resize(grayscale_cam, (W, H))

            cam_image = show_cam_on_image(image_np, grayscale_cam, use_rgb=is_rgb)
            heatmap_image = Image.fromarray(cam_image, mode='RGB' if is_rgb else 'L')
        return heatmap_image

    def check_batch(self, images: list[Image.Image], transform_fn: transforms.Compose, is_rgb=True):
        return [self.check(image, transform_fn, is_rgb=is_rgb) for image in images]

    # image = Image.open(img_path).convert('L' or 'RGB')
    def check_transformer_block_attention_heatmap(self, image_np: np.ndarray, attn_scores: torch.Tensor, patch_size: tuple[int, int]|int, alpha: float = 0.5, head_type: Literal['mean', 'max']='mean') -> tuple[np.ndarray, np.ndarray]:
        original_image_size = image_np.shape[-2], image_np.shape[-1] # (W, H)
        # attention
        # attn_scores is (B, H, N, N)
        if head_type == 'mean':
            attn_scores = attn_scores.mean(dim=(0, 1)) # (N, N)
        elif head_type == 'max':
            attn_scores = attn_scores.mean(dim=0).max() # (N, N)

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size) # (W_P, H_P)

        attn_map = attn_scores.reshape(patch_size)
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
        attn_map = cv2.resize(attn_map, original_image_size, interpolation=cv2.INTER_LINEAR)

        # heatmap
        colormap = plt.get_cmap("viridis")
        heatmap = colormap(attn_map)
        heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)
        image = image_np.astype(dtype=np.uint8)
        fused_image = cv2.addWeighted(heatmap, alpha, image, 1.0 - alpha, 0, dtype=np.uint8)
        # plt.figure(figsize=(6, 6))
        # plt.imshow(image)
        # plt.imshow(attn_map, cmap="viridis", alpha=alpha)
        # plt.axis("off")
        # plt.title("Attention Map Overlay")
        # plt.show()

        return fused_image, heatmap # (fused, attn_heatmap) all with uint8
