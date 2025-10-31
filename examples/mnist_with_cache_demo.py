#!/usr/bin/env python3
"""
MNIST数据集缓存功能演示

这个示例展示了如何在MNIST数据集中使用缓存功能，
以及缓存对加载速度的影响。
"""

import sys
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.dataset.mnist_dataset import MNISTDataset
from src.dataset.cache_manager import DatasetCacheManager
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_without_cache():
    """不使用缓存加载数据集"""
    logger.info("=" * 80)
    logger.info("场景1: 不使用缓存加载数据集")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    dataset = MNISTDataset(
        root_dir=Path("data/mnist"),
        split='train',
        enable_cache=False  # 禁用缓存
    )
    
    load_time = time.time() - start_time
    
    logger.info(f"数据集大小: {len(dataset)}")
    logger.info(f"加载时间: {load_time:.4f} 秒")
    logger.info("")
    
    return load_time


def load_with_cache_first_time():
    """第一次使用缓存（需要构建缓存）"""
    logger.info("=" * 80)
    logger.info("场景2: 第一次使用缓存（构建缓存）")
    logger.info("=" * 80)
    
    # 先清除现有缓存
    cache_manager = DatasetCacheManager(
        dataset_name='MNIST',
        version='v1',
        enable_cache=True
    )
    cache_manager.clear()
    
    start_time = time.time()
    
    dataset = MNISTDataset(
        root_dir=Path("data/mnist"),
        split='train',
        enable_cache=True,
        cache_version='v1'
    )
    
    # 保存到缓存
    dataset.save_to_cache(format='pkl')
    
    build_time = time.time() - start_time
    
    logger.info(f"数据集大小: {len(dataset)}")
    logger.info(f"构建缓存时间: {build_time:.4f} 秒")
    
    # 显示缓存信息
    info = cache_manager.get_cache_info()
    logger.info(f"缓存大小: {info.get('total_size_mb', 0):.2f} MB")
    logger.info("")
    
    return build_time


def load_with_cache_from_disk():
    """从缓存加载数据集"""
    logger.info("=" * 80)
    logger.info("场景3: 从缓存加载数据集")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    dataset = MNISTDataset(
        root_dir=Path("data/mnist"),
        split='train',
        enable_cache=True,
        cache_version='v1'
    )
    
    # 从缓存加载
    if dataset.load_from_cache(format='pkl'):
        logger.info("✓ 成功从缓存加载")
    else:
        logger.warning("✗ 缓存加载失败")
    
    load_time = time.time() - start_time
    
    logger.info(f"数据集大小: {len(dataset)}")
    logger.info(f"加载时间: {load_time:.4f} 秒")
    logger.info("")
    
    return load_time


def compare_performance():
    """比较性能"""
    logger.info("=" * 80)
    logger.info("性能比较")
    logger.info("=" * 80)
    
    # 测试不使用缓存
    time_no_cache = load_without_cache()
    
    # 测试第一次构建缓存
    time_build_cache = load_with_cache_first_time()
    
    # 测试从缓存加载
    time_from_cache = load_with_cache_from_disk()
    
    # 显示比较结果
    logger.info("=" * 80)
    logger.info("性能总结")
    logger.info("=" * 80)
    logger.info(f"不使用缓存:     {time_no_cache:.4f} 秒")
    logger.info(f"构建缓存:       {time_build_cache:.4f} 秒 ({time_build_cache/time_no_cache:.2f}x)")
    logger.info(f"从缓存加载:     {time_from_cache:.4f} 秒 ({time_from_cache/time_no_cache:.2f}x)")
    logger.info(f"加速比:        {time_no_cache/time_from_cache:.2f}x")
    logger.info("")


def demonstrate_cache_management():
    """演示缓存管理功能"""
    logger.info("=" * 80)
    logger.info("缓存管理功能演示")
    logger.info("=" * 80)
    
    cache_manager = DatasetCacheManager(
        dataset_name='MNIST',
        version='v1',
        enable_cache=True
    )
    
    # 创建多个划分的缓存
    for split in ['train', 'valid', 'test']:
        logger.info(f"\n创建 {split} 数据集缓存...")
        dataset = MNISTDataset(
            root_dir=Path("data/mnist"),
            split=split,
            enable_cache=True,
            cache_version='v1'
        )
        dataset.save_to_cache(format='pkl')
    
    # 显示所有缓存信息
    logger.info("\n" + "=" * 80)
    logger.info("所有缓存文件:")
    logger.info("=" * 80)
    
    info = cache_manager.get_cache_info()
    logger.info(f"数据集: {info['dataset_name']}")
    logger.info(f"版本: {info['version']}")
    logger.info(f"缓存目录: {info['cache_dir']}")
    logger.info(f"文件数: {info.get('total_files', 0)}")
    logger.info(f"总大小: {info.get('total_size_mb', 0):.2f} MB")
    
    if info.get('files'):
        logger.info("\n文件详情:")
        for file_info in info['files']:
            meta = file_info.get('metadata', {})
            logger.info(f"  - {file_info['filename']}")
            logger.info(f"    划分: {meta.get('split', 'unknown')}")
            logger.info(f"    大小: {file_info['size'] / 1024 / 1024:.2f} MB")
            logger.info(f"    样本数: {meta.get('num_samples', 'unknown')}")


def demonstrate_version_management():
    """演示版本管理"""
    logger.info("=" * 80)
    logger.info("版本管理演示")
    logger.info("=" * 80)
    
    # 创建v1版本缓存
    logger.info("创建 v1 版本缓存...")
    dataset_v1 = MNISTDataset(
        root_dir=Path("data/mnist"),
        split='train',
        enable_cache=True,
        cache_version='v1'
    )
    dataset_v1.save_to_cache(format='pkl')
    
    # 创建v2版本缓存
    logger.info("创建 v2 版本缓存...")
    dataset_v2 = MNISTDataset(
        root_dir=Path("data/mnist"),
        split='train',
        enable_cache=True,
        cache_version='v2'
    )
    dataset_v2.save_to_cache(format='pkl')
    
    # 显示不同版本的信息
    for version in ['v1', 'v2']:
        logger.info(f"\n版本 {version} 信息:")
        cache_manager = DatasetCacheManager(
            dataset_name='MNIST',
            version=version,
            enable_cache=True
        )
        info = cache_manager.get_cache_info()
        logger.info(f"  文件数: {info.get('total_files', 0)}")
        logger.info(f"  大小: {info.get('total_size_mb', 0):.2f} MB")


def main():
    """主函数"""
    logger.info("\n" + "=" * 80)
    logger.info("MNIST数据集缓存功能演示")
    logger.info("=" * 80 + "\n")
    
    try:
        # 选择要运行的演示
        # compare_performance()
        demonstrate_cache_management()
        # demonstrate_version_management()
        
        logger.info("\n演示完成！")
        
    except Exception as e:
        logger.error(f"演示过程中出错: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

