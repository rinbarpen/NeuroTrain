#!/usr/bin/env python3
"""
超级简单的热力图工具
仅需一行函数调用即可生成各种热力图
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Union, Optional, List, Tuple
from PIL import Image
import matplotlib.pyplot as plt


def quick_heatmap(image_path: Union[str, Path], 
                 output_path: Union[str, Path],
                 heatmap_type: str = "cam",
                 model: Optional[torch.nn.Module] = None,
                 model_name: str = "resnet50",
                 target_category: int = 0,
                 attention_data: Optional[Union[torch.Tensor, np.ndarray]] = None,
                 colormap: str = "viridis",
                 alpha: float = 0.5,
                 show: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    一键生成热力图 - 最简单的调用方式
    
    Args:
        image_path: 输入图像路径
        output_path: 输出图像路径
        heatmap_type: 热力图类型 ("cam", "attention", "custom")
        model: 模型实例 (对于cam类型，可选，会自动加载预训练模型)
        model_name: 模型名称 (当model为None时使用)
        target_category: 目标类别 (对于cam类型)
        attention_data: 注意力数据 (对于attention和custom类型)
        colormap: 颜色映射
        alpha: 叠加透明度
        show: 是否显示结果
        
    Returns:
        overlay_image: 叠加后的图像
        heatmap: 原始热力图
        
    Examples:
        # CAM热力图
        overlay, heatmap = quick_heatmap("image.jpg", "cam_output.png", "cam")
        
        # 自定义热力图
        attention_map = np.random.rand(14, 14)
        overlay, heatmap = quick_heatmap("image.jpg", "attn_output.png", "custom", 
                                       attention_data=attention_map)
        
        # ViT注意力
        vit_model = timm.create_model('vit_base_patch16_224', pretrained=True)
        overlay, heatmap = quick_heatmap("image.jpg", "vit_output.png", "attention", 
                                       model=vit_model)
    """
    from heatmap_all import HeatmapGenerator, load_pretrained_model, get_target_layers
    
    # 创建生成器
    generator = HeatmapGenerator(colormap=colormap, alpha=alpha)
    
    if heatmap_type == "cam":
        # CAM热力图
        if model is None:
            model = load_pretrained_model(model_name)
        target_layers = get_target_layers(model, model_name)
        
        overlay_image, heatmap = generator.generate_cam_heatmap(
            model=model,
            image_path=image_path,
            target_layers=target_layers,
            target_category=target_category,
            is_rgb=True
        )
        
    elif heatmap_type == "attention":
        # ViT注意力回溯
        if model is None:
            model = load_pretrained_model("deit_tiny_patch16_224")
        
        overlay_image, heatmap = generator.generate_vit_attention_rollout(
            model=model,
            image_path=image_path,
            head_fusion="mean",
            discard_ratio=0.9,
            is_rgb=True
        )
        
    elif heatmap_type == "custom":
        # 自定义热力图
        if attention_data is None:
            raise ValueError("custom类型需要提供attention_data")
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # 转换attention_data
        if isinstance(attention_data, torch.Tensor):
            attention_data = attention_data.squeeze().cpu().numpy()
        
        overlay_image, heatmap = generator.generate_custom_heatmap(
            image_np=image_np,
            heatmap=attention_data
        )
        
    else:
        raise ValueError(f"不支持的热力图类型: {heatmap_type}")
    
    # 保存结果
    generator.save_result(
        overlay_image=overlay_image,
        heatmap=heatmap,
        output_path=output_path,
        show=show
    )
    
    return overlay_image, heatmap


def attention_heatmap(image_path: Union[str, Path],
                     attention_weights: Union[torch.Tensor, np.ndarray],
                     output_path: Union[str, Path],
                     head_fusion: str = "mean",
                     discard_ratio: float = 0.0,
                     patch_size: int = 16,
                     colormap: str = "viridis",
                     alpha: float = 0.5,
                     show: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    注意力热力图专用函数 - 对于已经提取的注意力权重
    
    Args:
        image_path: 输入图像路径
        attention_weights: 注意力权重 [batch, heads, seq_len, seq_len] 或 [heads, seq_len, seq_len]
        output_path: 输出图像路径
        head_fusion: 多头融合方式 ("mean", "max", "sum")
        discard_ratio: 丢弃最低注意力的比例
        patch_size: patch大小
        colormap: 颜色映射
        alpha: 叠加透明度
        show: 是否显示
        
    Returns:
        overlay_image: 叠加后的图像
        attention_map: 注意力热力图
    """
    from heatmap_all import HeatmapGenerator
    
    generator = HeatmapGenerator(colormap=colormap, alpha=alpha)
    
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    
    # 转换注意力权重
    if isinstance(attention_weights, np.ndarray):
        attention_weights = torch.from_numpy(attention_weights)
    
    overlay_image, attention_map = generator.generate_attention_heatmap(
        image_np=image_np,
        attention_weights=attention_weights,
        patch_size=patch_size,
        head_fusion=head_fusion,
        discard_ratio=discard_ratio
    )
    
    # 保存结果
    generator.save_result(
        overlay_image=overlay_image,
        heatmap=attention_map,
        output_path=output_path,
        show=show
    )
    
    return overlay_image, attention_map


def cam_heatmap(image_path: Union[str, Path],
               output_path: Union[str, Path],
               model: Optional[torch.nn.Module] = None,
               model_name: str = "resnet50",
               target_category: int = 0,
               colormap: str = "viridis",
               alpha: float = 0.5,
               show: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    CAM热力图专用函数
    
    Args:
        image_path: 输入图像路径
        output_path: 输出图像路径
        model: 模型实例 (可选，会自动加载预训练模型)
        model_name: 模型名称 (当model为None时使用)
        target_category: 目标类别
        colormap: 颜色映射
        alpha: 叠加透明度
        show: 是否显示
        
    Returns:
        overlay_image: 叠加后的图像
        heatmap: CAM热力图
    """
    return quick_heatmap(
        image_path=image_path,
        output_path=output_path,
        heatmap_type="cam",
        model=model,
        model_name=model_name,
        target_category=target_category,
        colormap=colormap,
        alpha=alpha,
        show=show
    )


def vit_heatmap(image_path: Union[str, Path],
               output_path: Union[str, Path],
               model: Optional[torch.nn.Module] = None,
               model_name: str = "deit_tiny_patch16_224",
               head_fusion: str = "mean",
               discard_ratio: float = 0.9,
               colormap: str = "viridis",
               alpha: float = 0.5,
               show: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    ViT注意力热力图专用函数
    
    Args:
        image_path: 输入图像路径
        output_path: 输出图像路径
        model: ViT模型实例 (可选，会自动加载预训练模型)
        model_name: 模型名称 (当model为None时使用)
        head_fusion: 多头注意力融合方式
        discard_ratio: 丢弃最低注意力的比例
        colormap: 颜色映射
        alpha: 叠加透明度
        show: 是否显示
        
    Returns:
        overlay_image: 叠加后的图像
        attention_map: 注意力热力图
    """
    from heatmap_all import HeatmapGenerator, load_pretrained_model
    
    generator = HeatmapGenerator(colormap=colormap, alpha=alpha)
    
    if model is None:
        model = load_pretrained_model(model_name)
    
    overlay_image, attention_map = generator.generate_vit_attention_rollout(
        model=model,
        image_path=image_path,
        head_fusion=head_fusion,
        discard_ratio=discard_ratio,
        is_rgb=True
    )
    
    # 保存结果
    generator.save_result(
        overlay_image=overlay_image,
        heatmap=attention_map,
        output_path=output_path,
        show=show
    )
    
    return overlay_image, attention_map


# 示例使用函数
def demo():
    """演示各种热力图的使用方法"""
    print("热力图工具演示")
    print("请确保您有测试图像文件")
    
    # 测试图像路径
    test_image = "test_image.jpg"
    
    try:
        # 1. CAM热力图 (ResNet50)
        print("\n1. 生成ResNet50 CAM热力图...")
        overlay, heatmap = cam_heatmap(test_image, "resnet_cam.png", show=True)
        print("✓ ResNet50 CAM热力图已生成")
        
        # 2. ViT注意力热力图
        print("\n2. 生成ViT注意力热力图...")
        overlay, heatmap = vit_heatmap(test_image, "vit_attention.png", show=True)
        print("✓ ViT注意力热力图已生成")
        
        # 3. 自定义热力图
        print("\n3. 生成自定义热力图...")
        custom_attention = np.random.rand(14, 14)  # 示例注意力图
        overlay, heatmap = quick_heatmap(test_image, "custom_heatmap.png", 
                                       "custom", attention_data=custom_attention, show=True)
        print("✓ 自定义热力图已生成")
        
        print("\n所有热力图生成完成！")
        
    except FileNotFoundError:
        print(f"请在当前目录提供测试图像文件: {test_image}")
    except Exception as e:
        print(f"生成热力图时出错: {e}")


if __name__ == "__main__":
    demo()