"""
Dataset基础使用示例

本示例展示如何使用NeuroTrain的数据集模块加载和使用各种数据集。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import (
    get_dataset,
    get_train_dataset,
    get_test_dataset,
    get_train_valid_test_dataloader,
    random_sample
)
from src.config import get_config
import matplotlib.pyplot as plt
import numpy as np
import torch


def example_1_single_dataset():
    """示例1: 加载单个数据集"""
    print("=" * 80)
    print("示例1: 加载单个DRIVE数据集")
    print("=" * 80)
    
    config = {
        'dataset': {
            'name': 'drive',
            'root_dir': 'data/drive',
            'is_rgb': True,
            'train_split': 0.8,
            'image_size': [512, 512]
        }
    }
    
    # 获取训练和测试数据集
    train_dataset = get_train_dataset(config)
    test_dataset = get_test_dataset(config)
    
    print(f"训练样本数: {len(train_dataset)}")
    print(f"测试样本数: {len(test_dataset)}")
    
    # 获取一个样本
    if len(train_dataset) > 0:
        image, mask = train_dataset[0]
        print(f"图像形状: {image.shape}")
        print(f"掩码形状: {mask.shape}")
        
        # 可视化
        visualize_sample(image, mask, "DRIVE Dataset Sample")


def example_2_hybrid_dataset():
    """示例2: 使用混合数据集"""
    print("\n" + "=" * 80)
    print("示例2: 混合数据集（DRIVE + ChaseDB1）")
    print("=" * 80)
    
    config = {
        'dataset': {
            'name': 'enhanced_hybrid',
            'datasets': ['drive', 'medical/chasedb1'],
            'sampling_strategy': 'weighted',
            'ratios': [0.6, 0.4],
            'weights': [1.0, 1.2],
            'drive': {
                'root_dir': 'data/drive',
                'is_rgb': True
            },
            'medical/chasedb1': {
                'root_dir': 'data/chasedb1',
                'is_rgb': True
            }
        }
    }
    
    dataset = get_dataset(config)
    print(f"混合数据集总样本数: {len(dataset)}")
    
    # 采样不同来源的样本
    for i in range(min(3, len(dataset))):
        image, mask = dataset[i]
        source = dataset.get_source(i) if hasattr(dataset, 'get_source') else 'unknown'
        print(f"样本 {i}: 来源={source}, 图像形状={image.shape}")


def example_3_cifar10():
    """示例3: 加载CIFAR-10数据集"""
    print("\n" + "=" * 80)
    print("示例3: CIFAR-10图像分类数据集")
    print("=" * 80)
    
    config = {
        'dataset': {
            'name': 'cifar10',
            'root_dir': 'data/cifar10',
            'train': True,
            'download': True
        }
    }
    
    dataset = get_dataset(config)
    print(f"数据集大小: {len(dataset)}")
    print(f"类别: {dataset.classes if hasattr(dataset, 'classes') else 'N/A'}")
    
    # 获取并可视化一些样本
    if len(dataset) > 0:
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for i, ax in enumerate(axes.flat):
            if i < len(dataset):
                image, label = dataset[i]
                # 转换为numpy并调整维度
                if isinstance(image, torch.Tensor):
                    image = image.permute(1, 2, 0).numpy()
                ax.imshow(image)
                ax.set_title(f"Class: {label}")
                ax.axis('off')
        plt.suptitle("CIFAR-10 Samples")
        plt.tight_layout()
        plt.savefig('examples/output/cifar10_samples.png')
        print("已保存可视化结果到 examples/output/cifar10_samples.png")


def example_4_data_augmentation():
    """示例4: 数据增强"""
    print("\n" + "=" * 80)
    print("示例4: 数据增强示例")
    print("=" * 80)
    
    config = {
        'dataset': {
            'name': 'drive',
            'root_dir': 'data/drive',
            'is_rgb': True,
            'augmentation': {
                'random_flip': True,
                'random_rotation': True,
                'rotation_range': 15,
                'brightness_range': [0.8, 1.2],
                'elastic_deformation': True
            }
        }
    }
    
    dataset = get_dataset(config)
    
    if len(dataset) > 0:
        # 获取同一样本多次，查看不同的增强效果
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for i in range(4):
            image, mask = dataset[0]  # 重复获取第一个样本
            
            # 显示图像
            if isinstance(image, torch.Tensor):
                image = image.permute(1, 2, 0).numpy()
            axes[0, i].imshow(image)
            axes[0, i].set_title(f"Augmented Image {i+1}")
            axes[0, i].axis('off')
            
            # 显示掩码
            if isinstance(mask, torch.Tensor):
                mask = mask.squeeze().numpy()
            axes[1, i].imshow(mask, cmap='gray')
            axes[1, i].set_title(f"Augmented Mask {i+1}")
            axes[1, i].axis('off')
        
        plt.suptitle("Data Augmentation Examples")
        plt.tight_layout()
        plt.savefig('examples/output/augmentation_examples.png')
        print("已保存增强示例到 examples/output/augmentation_examples.png")


def example_5_dataloader():
    """示例5: 使用DataLoader"""
    print("\n" + "=" * 80)
    print("示例5: DataLoader使用")
    print("=" * 80)
    
    config = {
        'dataset': {
            'name': 'cifar10',
            'root_dir': 'data/cifar10',
            'train': True,
            'download': True
        },
        'training': {
            'batch_size': 32,
            'num_workers': 2
        }
    }
    
    # 获取DataLoader
    train_loader, valid_loader, test_loader = get_train_valid_test_dataloader(config)
    
    print(f"训练批次数: {len(train_loader)}")
    if valid_loader:
        print(f"验证批次数: {len(valid_loader)}")
    if test_loader:
        print(f"测试批次数: {len(test_loader)}")
    
    # 获取一个批次
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"\n批次 {batch_idx + 1}:")
        print(f"  图像形状: {images.shape}")
        print(f"  标签形状: {labels.shape}")
        if batch_idx == 0:
            print(f"  图像数据类型: {images.dtype}")
            print(f"  图像值范围: [{images.min():.3f}, {images.max():.3f}]")
        
        if batch_idx >= 2:  # 只打印前3个批次
            break


def example_6_random_sampling():
    """示例6: 随机采样"""
    print("\n" + "=" * 80)
    print("示例6: 随机采样子集")
    print("=" * 80)
    
    config = {
        'dataset': {
            'name': 'cifar10',
            'root_dir': 'data/cifar10',
            'train': True,
            'download': True
        }
    }
    
    dataset = get_dataset(config)
    print(f"原始数据集大小: {len(dataset)}")
    
    # 随机采样100个样本
    subset = random_sample(dataset, n=100)
    print(f"采样后子集大小: {len(subset)}")
    
    # 统计子集中的类别分布
    if hasattr(subset.dataset, 'targets'):
        subset_labels = [subset.dataset.targets[i] for i in subset.indices]
        unique, counts = np.unique(subset_labels, return_counts=True)
        print("\n子集类别分布:")
        for cls, count in zip(unique, counts):
            print(f"  类别 {cls}: {count} 样本")


def visualize_sample(image, mask, title):
    """可视化单个样本"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 显示图像
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].axis('off')
    
    # 显示掩码
    if isinstance(mask, torch.Tensor):
        mask = mask.squeeze().numpy()
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Mask")
    axes[1].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # 创建输出目录
    output_dir = Path('examples/output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / 'sample_visualization.png')
    print(f"已保存可视化结果")


def main():
    """运行所有示例"""
    print("NeuroTrain Dataset模块使用示例")
    print("=" * 80)
    
    # 创建输出目录
    output_dir = Path('examples/output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 运行各个示例
        example_1_single_dataset()
        example_2_hybrid_dataset()
        example_3_cifar10()
        example_4_data_augmentation()
        example_5_dataloader()
        example_6_random_sampling()
        
        print("\n" + "=" * 80)
        print("所有示例运行完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

