#!/usr/bin/env python3
"""
自动缓存功能演示

展示数据集的自动缓存功能：
- 第一次加载时自动创建缓存
- 之后的加载自动从缓存读取
- 完全透明，无需手动调用
"""

import sys
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.dataset.mnist_dataset import MNISTDataset
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_auto_cache():
    """演示自动缓存功能"""
    logger.info("=" * 80)
    logger.info("自动缓存功能演示")
    logger.info("=" * 80)
    
    logger.info("\n第1次加载 - 将自动创建缓存")
    logger.info("-" * 80)
    start_time = time.time()
    
    # 第一次加载：缓存不存在，会加载数据并自动保存到缓存
    dataset1 = MNISTDataset(
        root_dir=Path("data/mnist"),
        split='train',
        # enable_cache=True 是默认值，可以不写
    )
    
    time1 = time.time() - start_time
    logger.info(f"数据集大小: {len(dataset1)}")
    logger.info(f"加载时间: {time1:.4f} 秒")
    
    # 验证数据可以正常访问
    sample = dataset1[0]
    logger.info(f"样本形状: image={sample['image'].shape}, mask={sample['mask'].shape}")
    
    logger.info("\n第2次加载 - 将自动从缓存读取")
    logger.info("-" * 80)
    start_time = time.time()
    
    # 第二次加载：缓存存在，会自动从缓存加载
    dataset2 = MNISTDataset(
        root_dir=Path("data/mnist"),
        split='train',
    )
    
    time2 = time.time() - start_time
    logger.info(f"数据集大小: {len(dataset2)}")
    logger.info(f"加载时间: {time2:.4f} 秒")
    
    # 验证数据一致性
    sample2 = dataset2[0]
    logger.info(f"样本形状: image={sample2['image'].shape}, mask={sample2['mask'].shape}")
    
    logger.info("\n性能对比")
    logger.info("-" * 80)
    logger.info(f"第1次加载（创建缓存）: {time1:.4f} 秒")
    logger.info(f"第2次加载（从缓存）:   {time2:.4f} 秒")
    if time2 > 0:
        logger.info(f"加速比: {time1/time2:.2f}x")
    
    logger.info("\n✓ 完全自动，无需手动调用任何缓存方法！")


def demo_disable_cache():
    """演示禁用缓存"""
    logger.info("\n" + "=" * 80)
    logger.info("禁用缓存演示")
    logger.info("=" * 80)
    
    logger.info("\n加载数据集（禁用缓存）...")
    start_time = time.time()
    
    # 禁用缓存
    dataset = MNISTDataset(
        root_dir=Path("data/mnist"),
        split='train',
        enable_cache=False  # 显式禁用缓存
    )
    
    time_taken = time.time() - start_time
    logger.info(f"数据集大小: {len(dataset)}")
    logger.info(f"加载时间: {time_taken:.4f} 秒")
    logger.info("✓ 不使用缓存，每次都重新加载")


def demo_force_rebuild():
    """演示强制重建缓存"""
    logger.info("\n" + "=" * 80)
    logger.info("强制重建缓存演示")
    logger.info("=" * 80)
    
    logger.info("\n强制重建缓存...")
    start_time = time.time()
    
    # 强制重建缓存（忽略现有缓存）
    dataset = MNISTDataset(
        root_dir=Path("data/mnist"),
        split='train',
        force_rebuild_cache=True  # 强制重建
    )
    
    time_taken = time.time() - start_time
    logger.info(f"数据集大小: {len(dataset)}")
    logger.info(f"加载时间: {time_taken:.4f} 秒")
    logger.info("✓ 已重建缓存")


def demo_multiple_splits():
    """演示多个数据集划分的自动缓存"""
    logger.info("\n" + "=" * 80)
    logger.info("多划分自动缓存演示")
    logger.info("=" * 80)
    
    for split in ['train', 'valid', 'test']:
        logger.info(f"\n加载 {split} 数据集...")
        start_time = time.time()
        
        dataset = MNISTDataset(
            root_dir=Path("data/mnist"),
            split=split,
        )
        
        time_taken = time.time() - start_time
        logger.info(f"  大小: {len(dataset)}")
        logger.info(f"  时间: {time_taken:.4f} 秒")
    
    logger.info("\n✓ 所有划分都已自动缓存")


def demo_version_management():
    """演示版本管理"""
    logger.info("\n" + "=" * 80)
    logger.info("版本管理演示")
    logger.info("=" * 80)
    
    logger.info("\n加载 v1 版本...")
    dataset_v1 = MNISTDataset(
        root_dir=Path("data/mnist"),
        split='train',
        cache_version='v1'
    )
    logger.info(f"  v1 大小: {len(dataset_v1)}")
    
    logger.info("\n加载 v2 版本...")
    dataset_v2 = MNISTDataset(
        root_dir=Path("data/mnist"),
        split='train',
        cache_version='v2'
    )
    logger.info(f"  v2 大小: {len(dataset_v2)}")
    
    logger.info("\n✓ 不同版本独立缓存，互不干扰")


def show_usage():
    """展示使用方法"""
    logger.info("\n" + "=" * 80)
    logger.info("使用说明")
    logger.info("=" * 80)
    
    usage = """
数据集缓存功能使用非常简单：

1. 【默认启用】缓存功能默认启用，直接使用即可：
   
   dataset = MNISTDataset(
       root_dir=Path("data/mnist"),
       split='train'
   )
   
   第一次运行：自动创建缓存
   之后运行：自动从缓存加载

2. 【禁用缓存】如果不需要缓存：
   
   dataset = MNISTDataset(
       root_dir=Path("data/mnist"),
       split='train',
       enable_cache=False
   )

3. 【强制重建】数据更新后需要重建缓存：
   
   dataset = MNISTDataset(
       root_dir=Path("data/mnist"),
       split='train',
       force_rebuild_cache=True
   )

4. 【版本管理】为不同的实验使用不同版本：
   
   dataset = MNISTDataset(
       root_dir=Path("data/mnist"),
       split='train',
       cache_version='exp1'
   )

5. 【查看缓存】使用命令行工具：
   
   python tools/dataset_cache_tool.py list
   python tools/dataset_cache_tool.py info MNIST
   python tools/dataset_cache_tool.py clear MNIST

完全自动，无需任何手动操作！
    """
    logger.info(usage)


def main():
    """主函数"""
    logger.info("\n" + "=" * 80)
    logger.info("数据集自动缓存功能演示")
    logger.info("=" * 80)
    
    try:
        # 运行演示
        demo_auto_cache()
        # demo_disable_cache()
        # demo_force_rebuild()
        # demo_multiple_splits()
        # demo_version_management()
        # show_usage()
        
        logger.info("\n" + "=" * 80)
        logger.info("演示完成！")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"演示过程中出错: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

