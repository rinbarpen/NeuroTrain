#!/usr/bin/env python3
"""
简化的热力图可视化工具
支持多种热力图类型：CAM热力图、注意力机制热力图等
一个脚本解决所有可视化需求
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Literal, Optional, Union, Tuple
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# 依赖库的导入（处理可选依赖）
try:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRAD_CAM_AVAILABLE = True
except ImportError:
    GRAD_CAM_AVAILABLE = False
    print("警告: pytorch_grad_cam 未安装，CAM功能将不可用")

try:
    from torchvision.models import resnet50
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    print("警告: torchvision.models 不可用")

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("警告: timm 不可用，部分预训练模型将无法加载")

# 在依赖库导入部分添加painter支持
try:
    from src.utils.painter import Plot, Font, CmapPresets
    PAINTER_AVAILABLE = True
except ImportError:
    PAINTER_AVAILABLE = False
    print("警告: painter.py 不可用，将使用matplotlib的默认保存功能")


class HeatmapGenerator:
    """统一的热力图生成器"""

    def __init__(self, 
                 device: Optional[str] = None,
                 colormap: str = "viridis",
                 alpha: float = 0.5):
        """
        初始化热力图生成器
        
        Args:
            device: 计算设备 ('cpu', 'cuda')
            colormap: 热力图颜色映射 ('viridis', 'jet', 'hot', 'coolwarm', etc.)
            alpha: 热力图叠加透明度 [0, 1]
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.colormap = colormap
        self.alpha = alpha
        
    def _normalize_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        """标准化热力图到[0,1]范围"""
        if heatmap.max() > heatmap.min():
            return (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        return heatmap
    
    def _apply_colormap_and_overlay(self, 
                                   image_np: np.ndarray, 
                                   heatmap: np.ndarray,
                                   resize_to_image: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用颜色映射并叠加到原图
        
        Returns:
            overlay_image: 叠加后的图像
            colored_heatmap: 带颜色的热力图
        """
        if resize_to_image:
            H, W = image_np.shape[:2]
            heatmap = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)
        
        # 标准化热力图
        heatmap_norm = self._normalize_heatmap(heatmap)
        
        # 应用颜色映射
        colormap_fn = plt.get_cmap(self.colormap)
        colored_heatmap = colormap_fn(heatmap_norm)
        colored_heatmap = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
        
        # 确保原图为uint8格式
        if image_np.dtype != np.uint8:
            image_display = (image_np * 255).astype(np.uint8) if image_np.max() <= 1.0 else image_np.astype(np.uint8)
        else:
            image_display = image_np
        
        # 叠加
        overlay_image = cv2.addWeighted(colored_heatmap, self.alpha, image_display, 1.0 - self.alpha, 0)
        
        return overlay_image, colored_heatmap

    def generate_cam_heatmap(self, 
                           model: torch.nn.Module,
                           image_path: Union[str, Path],
                           target_layers: List[torch.nn.Module],
                           target_category: int = 0,
                           is_rgb: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成GradCAM++热力图
        
        Returns:
            overlay_image: 叠加后的图像
            heatmap: 原始热力图
        """
        if not GRAD_CAM_AVAILABLE:
            raise ImportError("pytorch_grad_cam 未安装，无法生成CAM热力图")
        
        # 加载和预处理图像
        image = Image.open(image_path)
        image = image.convert('RGB') if is_rgb else image.convert('L')
        
        # 构建变换
        transform = self._build_classification_transform(is_rgb)
        input_tensor = transform(image).unsqueeze(0)
        
        # 设备迁移
        model = model.to(self.device)
        input_tensor = input_tensor.to(self.device)
        model.eval()
        
        # 生成CAM
        targets = [ClassifierOutputTarget(target_category)]
        cam = GradCAMPlusPlus(model, target_layers=target_layers)
        
        grayscale_cam = cam(input_tensor, targets=targets, eigen_smooth=False)[0, :]
        
        # 原图处理
        image_np = np.array(image)
        if image_np.ndim == 2:  # 灰度图转3通道
            image_np = np.stack([image_np] * 3, axis=-1)
        image_np = image_np.astype(np.float32) / 255.0
        
        # 应用颜色映射和叠加
        overlay_image, colored_heatmap = self._apply_colormap_and_overlay(image_np, grayscale_cam)
        
        return overlay_image, grayscale_cam

    def generate_attention_heatmap(self, 
                                 image_np: np.ndarray,
                                 attention_weights: torch.Tensor,
                                 patch_size: Union[int, Tuple[int, int]] = 16,
                                 head_fusion: Literal['mean', 'max', 'sum'] = 'mean',
                                 discard_ratio: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成Transformer注意力热力图
        
        Args:
            image_np: 原图像数组 [H, W, C]
            attention_weights: 注意力权重 [batch, heads, seq_len, seq_len] 或 [heads, seq_len, seq_len]
            patch_size: patch大小
            head_fusion: 多头融合方式
            discard_ratio: 丢弃最低注意力的比例
        
        Returns:
            overlay_image: 叠加后的图像
            attention_map: 注意力热力图
        """
        # 处理attention_weights维度
        if attention_weights.dim() == 4:  # [B, H, N, N]
            attention_weights = attention_weights[0]  # 取第一个batch
        
        # 多头融合
        if head_fusion == 'mean':
            attn_map = attention_weights.mean(dim=0)  # [N, N]
        elif head_fusion == 'max':
            attn_map = attention_weights.max(dim=0)[0]
        elif head_fusion == 'sum':
            attn_map = attention_weights.sum(dim=0)
        elif head_fusion == 'min':
            attn_map = attention_weights.min(dim=0)[0]
        else:
            raise ValueError(f"不支持的注意力头融合方式: {head_fusion}")
        
        # 提取CLS token到patch的注意力（假设第0个是CLS token）
        if attn_map.size(0) > 1:
            patch_attention = attn_map[0, 1:]  # [N-1] 忽略CLS自注意力
        else:
            patch_attention = attn_map.flatten()
        
        # 丢弃低注意力
        if discard_ratio > 0:
            num_discard = int(len(patch_attention) * discard_ratio)
            _, indices = torch.topk(patch_attention, num_discard, largest=False)
            patch_attention[indices] = 0
        
        # 重塑为空间网格
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        
        grid_size = int(np.sqrt(len(patch_attention)))
        if grid_size * grid_size != len(patch_attention):
            # 如果不是完美平方数，尝试推断
            print(f"警告: patch数量 {len(patch_attention)} 不是完美平方数，使用最接近的网格")
            grid_size = int(np.sqrt(len(patch_attention)))
            patch_attention = patch_attention[:grid_size*grid_size]
        
        attention_2d = patch_attention.view(grid_size, grid_size).cpu().numpy()
        
        # 应用颜色映射和叠加
        overlay_image, colored_heatmap = self._apply_colormap_and_overlay(image_np, attention_2d)
        
        return overlay_image, attention_2d

    def generate_vit_attention_rollout(self,
                                     model: torch.nn.Module,
                                     image_path: Union[str, Path],
                                     head_fusion: Literal['mean', 'max', 'min'] = 'mean',
                                     discard_ratio: float = 0.9,
                                     is_rgb: bool = True,
                                     attention_layer_name: str = 'attn_drop') -> Tuple[np.ndarray, np.ndarray]:
        """
        生成Vision Transformer注意力回溯热力图
        
        Args:
            model: ViT模型
            image_path: 输入图像路径
            head_fusion: 多头注意力融合方式
            discard_ratio: 丢弃最低注意力的比例
            is_rgb: 是否为RGB图像
            attention_layer_name: 注意力层名称模式
        
        Returns:
            overlay_image: 叠加后的图像
            attention_map: 注意力热力图
        """
        # 注意力钩子收集器
        attentions = []
        
        def get_attention(module, input, output):
            attentions.append(output.cpu())
        
        # 注册钩子
        hooks = []
        for name, module in model.named_modules():
            if attention_layer_name in name:
                hook = module.register_forward_hook(get_attention)
                hooks.append(hook)
        
        try:
            # 加载和预处理图像
            image = Image.open(image_path)
            image = image.convert('RGB') if is_rgb else image.convert('L')
            
            # 构建变换（ViT通常用不同的标准化）
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            input_tensor = transform(image).unsqueeze(0)
            
            # 设备迁移
            model = model.to(self.device)
            input_tensor = input_tensor.to(self.device)
            model.eval()
            
            # 前向传播收集注意力
            attentions.clear()
            with torch.no_grad():
                output = model(input_tensor)
            
            # 执行注意力回溯
            attention_map = self._rollout_attention(attentions, discard_ratio, head_fusion)
            
            # 原图处理
            image_np = np.array(image)
            if image_np.ndim == 2:
                image_np = np.stack([image_np] * 3, axis=-1)
            
            # 应用颜色映射和叠加
            overlay_image, colored_heatmap = self._apply_colormap_and_overlay(image_np, attention_map)
            
            return overlay_image, attention_map
            
        finally:
            # 清理钩子
            for hook in hooks:
                hook.remove()

    def _rollout_attention(self, attentions: List[torch.Tensor], 
                          discard_ratio: float, 
                          head_fusion: str) -> np.ndarray:
        """
        执行注意力回溯算法
        基于 https://github.com/jacobgil/vit-explain 的实现
        """
        result = torch.eye(attentions[0].size(-1))
        with torch.no_grad():
            for attention in attentions:
                if head_fusion == "mean":
                    attention_heads_fused = attention.mean(axis=1)
                elif head_fusion == "max":
                    attention_heads_fused = attention.max(axis=1)[0]
                elif head_fusion == "min":
                    attention_heads_fused = attention.min(axis=1)[0]
                else:
                    raise ValueError(f"不支持的注意力头融合类型: {head_fusion}")

                # 丢弃最低注意力，但保留class token
                flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
                _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
                indices = indices[indices != 0]  # 不丢弃class token
                flat[0, indices] = 0

                I = torch.eye(attention_heads_fused.size(-1))
                a = (attention_heads_fused + 1.0*I)/2
                a = a / a.sum(dim=-1)

                result = torch.matmul(a, result)
        
        # 提取class token到图像patch的注意力
        mask = result[0, 0, 1:]  # 忽略class token自注意力
        # 对于224x224图像，从196个patch到14x14网格
        width = int(mask.size(-1)**0.5)
        mask = mask.reshape(width, width).numpy()
        mask = mask / np.max(mask) if np.max(mask) > 0 else mask
        return mask

    def generate_custom_heatmap(self, 
                              image_np: np.ndarray,
                              heatmap: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成自定义热力图（用户提供热力图数据）
        
        Args:
            image_np: 原图像 [H, W, C]
            heatmap: 热力图数据 [H', W']
        
        Returns:
            overlay_image: 叠加后的图像
            colored_heatmap: 彩色热力图
        """
        overlay_image, colored_heatmap = self._apply_colormap_and_overlay(image_np, heatmap)
        return overlay_image, colored_heatmap

    def _build_classification_transform(self, is_rgb: bool) -> transforms.Compose:
        """构建分类模型的图像变换"""
        if is_rgb:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def save_result(self, 
                   overlay_image: np.ndarray,
                   heatmap: np.ndarray,
                   output_path: Union[str, Path],
                   save_separate: bool = False,
                   show: bool = False) -> None:
        """
        保存结果图像
        
        Args:
            overlay_image: 叠加图像
            heatmap: 原始热力图
            output_path: 输出路径
            save_separate: 是否分别保存原始热力图
            show: 是否显示
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存叠加图像
        plt.figure(figsize=(10, 5))
        
        if save_separate:
            plt.subplot(1, 2, 1)
            plt.imshow(overlay_image)
            plt.title("Heatmap Overlay")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(heatmap, cmap=self.colormap)
            plt.title("Raw Heatmap")
            plt.axis('off')
        else:
            plt.imshow(overlay_image)
            plt.title("Heatmap Visualization")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        
        if show:
            plt.show()
        else:
            plt.close()
        
        print(f"热力图已保存到: {output_path}")

    def save_result_with_painter(self, 
                                overlay_image: np.ndarray,
                                heatmap: np.ndarray,
                                output_path: Union[str, Path],
                                save_separate: bool = False,
                                show: bool = False,
                                layer_info: Optional[str] = None,
                                module_info: Optional[str] = None) -> None:
        """
        使用painter.py保存热力图结果，支持添加层级和模块信息
        
        Args:
            overlay_image: 叠加图像
            heatmap: 原始热力图
            output_path: 输出路径
            save_separate: 是否分别保存原始热力图
            show: 是否显示
            layer_info: 层级信息（如"layer4[-1]"）
            module_info: 模块信息（如"ResNet50"）
        """
        if not PAINTER_AVAILABLE:
            # 回退到原始方法
            self.save_result(overlay_image, heatmap, output_path, save_separate, show)
            return
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建Plot实例
        if save_separate:
            plot = Plot(nrows=1, ncols=2, figsize=(12, 5))
            plot.set_theme_preset('scientific')
            
            # 热力图叠加子图
            subplot1 = plot.subplot(0)
            subplot1.image(overlay_image).title("Heatmap Overlay", Font.title_font())
            subplot1._ax.axis('off')
            
            # 原始热力图子图
            subplot2 = plot.subplot(1)
            cmap = CmapPresets.get_cmap('feature', 'attention')
            subplot2.image(heatmap, cmap=cmap).title("Raw Heatmap", Font.title_font())
            subplot2._ax.axis('off')
            
            # 添加图像信息注释
            if layer_info or module_info:
                info_text = []
                if module_info:
                    info_text.append(f"Model: {module_info}")
                if layer_info:
                    info_text.append(f"Layer: {layer_info}")
                
                # 在图像底部添加信息
                subplot1._ax.text(0.5, -0.1, " | ".join(info_text), 
                                 transform=subplot1._ax.transAxes,
                                 ha='center', va='top',
                                 **Font.small_font().build(),
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        else:
            plot = Plot(nrows=1, ncols=1, figsize=(8, 6))
            plot.set_theme_preset('scientific')
            
            subplot = plot.subplot(0)
            subplot.image(overlay_image).title("Heatmap Visualization", Font.title_font())
            subplot._ax.axis('off')
            
            # 添加图像信息注释
            if layer_info or module_info:
                info_text = []
                if module_info:
                    info_text.append(f"Model: {module_info}")
                if layer_info:
                    info_text.append(f"Layer: {layer_info}")
                
                # 在图像底部添加信息
                subplot._ax.text(0.5, -0.1, " | ".join(info_text), 
                                transform=subplot._ax.transAxes,
                                ha='center', va='top',
                                **Font.small_font().build(),
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 保存为PNG格式
        plot.save(str(output_path), format='png', bbox_inches='tight', dpi=300)
        
        if show:
            plot.show()
        else:
            plot.close()
        
        print(f"热力图已保存到: {output_path}")

    def extract_layer_module_info(self, 
                                 model: Optional[torch.nn.Module] = None,
                                 model_name: Optional[str] = None,
                                 target_layers: Optional[List[torch.nn.Module]] = None) -> Tuple[str, str]:
        """
        从模型中提取层级和模块信息
        
        Args:
            model: PyTorch模型
            model_name: 模型名称
            target_layers: 目标层列表
        
        Returns:
            layer_info: 层级信息字符串
            module_info: 模块信息字符串
        """
        module_info = model_name or "Unknown Model"
        layer_info = "Unknown Layer"
        
        if model is not None and target_layers is not None:
            # 尝试找到目标层在模型中的名称
            layer_names = []
            for target_layer in target_layers:
                for name, module in model.named_modules():
                    if module is target_layer:
                        layer_names.append(name)
                        break
                else:
                    # 如果找不到具体名称，尝试通过类型推断
                    layer_type = type(target_layer).__name__
                    layer_names.append(f"{layer_type}")
            
            if layer_names:
                layer_info = " + ".join(layer_names)
        
        return layer_info, module_info


def load_pretrained_model(model_name: str = "resnet50") -> torch.nn.Module:
    """加载预训练模型"""
    if model_name == "resnet50" and TORCHVISION_AVAILABLE:
        try:
            model = resnet50(weights="IMAGENET1K_V1")
        except:
            model = resnet50(pretrained=True)
        return model
    elif model_name.startswith("deit") or model_name.startswith("vit"):
        if TIMM_AVAILABLE:
            return timm.create_model(model_name, pretrained=True)
        else:
            # 尝试从torch hub加载
            if "deit" in model_name:
                return torch.hub.load('facebookresearch/deit:main', model_name, pretrained=True)
            else:
                raise ImportError(f"无法加载ViT模型 {model_name}，请安装timm或检查模型名称")
    elif TIMM_AVAILABLE:
        return timm.create_model(model_name, pretrained=True)
    else:
        raise ImportError(f"无法加载模型 {model_name}，请检查依赖库安装")


def get_target_layers(model: torch.nn.Module, model_name: str = "resnet50") -> List[torch.nn.Module]:
    """获取模型的目标层"""
    if "resnet" in model_name.lower():
        return [model.layer4[-1]]
    elif "vit" in model_name.lower() or "deit" in model_name.lower():
        return [model.blocks[-1].norm1]
    else:
        # 默认策略：找最后的卷积层
        conv_layers = []
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.Conv3d)):
                conv_layers.append(module)
        return [conv_layers[-1]] if conv_layers else []


def main():
    parser = argparse.ArgumentParser(description="统一热力图可视化工具")
    parser.add_argument("-i", "--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("-o", "--output", type=str, required=True, help="输出图像路径")
    parser.add_argument("--type", choices=["cam", "attention", "vit-rollout", "custom"], default="cam", 
                       help="热力图类型")
    
    # CAM相关参数
    parser.add_argument("--model", type=str, default="resnet50", help="模型名称")
    parser.add_argument("--category", type=int, default=0, help="目标类别")
    
    # 注意力相关参数
    parser.add_argument("--attention-file", type=str, help="注意力权重文件路径(.pt/.npy)")
    parser.add_argument("--patch-size", type=int, default=16, help="patch大小")
    parser.add_argument("--head-fusion", choices=["mean", "max", "sum", "min"], default="mean",
                       help="多头注意力融合方式")
    parser.add_argument("--discard-ratio", type=float, default=0.9, help="丢弃最低注意力的比例")
    parser.add_argument("--attention-layer", type=str, default="attn_drop", help="注意力层名称模式")
    
    # 通用参数
    parser.add_argument("--colormap", type=str, default="viridis", help="颜色映射")
    parser.add_argument("--alpha", type=float, default=0.5, help="叠加透明度")
    parser.add_argument("--rgb", action="store_true", help="RGB模式")
    parser.add_argument("--show", action="store_true", help="显示结果")
    parser.add_argument("--save-separate", action="store_true", help="分别保存原始热力图")
    parser.add_argument("--heatmap-file", type=str, default=None, help="自定义热力图文件(.npy/.pt)，用于--type=custom")
    # painter与元信息
    parser.add_argument("--no-painter", action="store_true", help="不使用painter保存（回退matplotlib）")
    parser.add_argument("--model-name", type=str, default=None, help="自定义模型名称（用于attention/custom类型的标注）")
    parser.add_argument("--layer-info", type=str, default=None, help="自定义层级信息（用于attention/custom类型的标注）")
    
    args = parser.parse_args()
    
    # 创建热力图生成器
    generator = HeatmapGenerator(colormap=args.colormap, alpha=args.alpha)
    
    # 准备元信息容器
    layer_info: Optional[str] = None
    module_info: Optional[str] = None
    
    if args.type == "cam":
        # CAM热力图
        model = load_pretrained_model(args.model)
        target_layers = get_target_layers(model, args.model)
        
        overlay_image, heatmap = generator.generate_cam_heatmap(
            model=model,
            image_path=args.image,
            target_layers=target_layers,
            target_category=args.category,
            is_rgb=args.rgb
        )
        # 元信息
        li, mi = generator.extract_layer_module_info(model=model, model_name=args.model, target_layers=target_layers)
        layer_info, module_info = li, mi
        
    elif args.type == "attention":
        # 注意力热力图（从文件加载）
        if not args.attention_file:
            raise ValueError("注意力热力图需要提供 --attention-file 参数")
        
        # 加载注意力权重
        if args.attention_file.endswith('.pt'):
            attention_weights = torch.load(args.attention_file, map_location='cpu')
        elif args.attention_file.endswith('.npy'):
            attention_weights = torch.from_numpy(np.load(args.attention_file))
        else:
            raise ValueError("不支持的注意力文件格式，请使用.pt或.npy")
        
        # 加载图像
        image = Image.open(args.image)
        if args.rgb:
            image = image.convert('RGB')
        image_np = np.array(image)
        if image_np.ndim == 2:
            image_np = np.stack([image_np] * 3, axis=-1)
        
        overlay_image, heatmap = generator.generate_attention_heatmap(
            image_np=image_np,
            attention_weights=attention_weights,
            patch_size=args.patch_size,
            head_fusion=args.head_fusion,
            discard_ratio=args.discard_ratio
        )
        # 元信息（来自用户提供）
        module_info = args.model_name or "CustomModel"
        layer_info = args.layer_info or "Transformer-Attention"
        
    elif args.type == "vit-rollout":
        # ViT注意力回溯热力图
        model = load_pretrained_model(args.model)
        
        overlay_image, heatmap = generator.generate_vit_attention_rollout(
            model=model,
            image_path=args.image,
            head_fusion=args.head_fusion,
            discard_ratio=args.discard_ratio,
            is_rgb=args.rgb,
            attention_layer_name=args.attention_layer
        )
        # 元信息
        module_info = args.model
        layer_info = f"rollout@{args.attention_layer}"
        
    elif args.type == "custom":
        # 自定义热力图：从文件加载二维热力图，或生成示例
        image = Image.open(args.image)
        image_np = np.array(image)
        if image_np.ndim == 2:
            image_np = np.stack([image_np] * 3, axis=-1)
        
        if args.heatmap_file is not None:
            if args.heatmap_file.endswith('.pt'):
                heatmap_arr = torch.load(args.heatmap_file, map_location='cpu')
                if isinstance(heatmap_arr, torch.Tensor):
                    heatmap_arr = heatmap_arr.squeeze().cpu().numpy()
            elif args.heatmap_file.endswith('.npy'):
                heatmap_arr = np.load(args.heatmap_file)
            else:
                raise ValueError("不支持的自定义热力图文件格式，请使用.pt或.npy")
            
            # 若是三维或更高维，尝试自动降到二维
            if heatmap_arr.ndim > 2:
                heatmap_arr = np.mean(heatmap_arr, axis=tuple(range(heatmap_arr.ndim - 2)))
        else:
            # 生成示例热力图（低分辨率）
            H, W = image_np.shape[:2]
            heatmap_arr = np.random.rand(max(1, H//4), max(1, W//4))
        
        overlay_image, heatmap = generator.generate_custom_heatmap(
            image_np=image_np,
            heatmap=heatmap_arr
        )
        # 元信息（来自用户提供）
        module_info = args.model_name or "CustomModel"
        layer_info = args.layer_info or "Custom-Heatmap"
    
    # 保存结果（优先使用painter并写入元信息）
    use_painter = (not args.no_painter) and PAINTER_AVAILABLE
    if use_painter:
        generator.save_result_with_painter(
            overlay_image=overlay_image,
            heatmap=heatmap,
            output_path=args.output,
            save_separate=args.save_separate,
            show=args.show,
            layer_info=layer_info,
            module_info=module_info,
        )
    else:
        generator.save_result(
            overlay_image=overlay_image,
            heatmap=heatmap,
            output_path=args.output,
            save_separate=args.save_separate,
            show=args.show
        )


if __name__ == "__main__":
    main()