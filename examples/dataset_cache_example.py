"""
数据集缓存功能使用示例

展示如何使用数据集缓存功能来加速数据加载。
"""

import sys
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


def example1_basic_cache():
    """示例1: 基本的缓存使用"""
    logger.info("=" * 80)
    logger.info("示例1: 基本的缓存使用")
    logger.info("=" * 80)
    
    # 创建数据集时启用缓存
    dataset = MNISTDataset(
        root_dir=Path("data/mnist"),
        split='train',
        enable_cache=True,  # 启用缓存
        cache_version='v1'   # 缓存版本
    )
    
    logger.info(f"数据集大小: {len(dataset)}")
    
    # 保存到缓存
    logger.info("保存数据集到缓存...")
    dataset.save_to_cache(format='pkl')
    
    logger.info("示例1完成\n")


def example2_load_from_cache():
    """示例2: 从缓存加载数据集"""
    logger.info("=" * 80)
    logger.info("示例2: 从缓存加载数据集")
    logger.info("=" * 80)
    
    # 创建数据集并尝试从缓存加载
    dataset = MNISTDataset(
        root_dir=Path("data/mnist"),
        split='train',
        enable_cache=True
    )
    
    # 尝试从缓存加载
    if dataset.load_from_cache(format='pkl'):
        logger.info("成功从缓存加载！")
    else:
        logger.info("缓存不存在，正常加载数据集")
        # 这里会正常加载数据集
        # ...
        # 然后保存到缓存
        dataset.save_to_cache(format='pkl')
    
    logger.info(f"数据集大小: {len(dataset)}")
    logger.info("示例2完成\n")


def example3_cache_manager():
    """示例3: 使用缓存管理器"""
    logger.info("=" * 80)
    logger.info("示例3: 使用缓存管理器")
    logger.info("=" * 80)
    
    # 创建缓存管理器
    cache_manager = DatasetCacheManager(
        dataset_name='mnist',
        version='v1',
        enable_cache=True
    )
    
    # 获取缓存信息
    cache_info = cache_manager.get_cache_info()
    logger.info(f"数据集: {cache_info['dataset_name']}")
    logger.info(f"缓存目录: {cache_info['cache_dir']}")
    logger.info(f"缓存文件数: {cache_info.get('total_files', 0)}")
    logger.info(f"总大小: {cache_info.get('total_size_mb', 0):.2f} MB")
    
    if cache_info.get('files'):
        logger.info("\n缓存文件列表:")
        for file_info in cache_info['files']:
            logger.info(f"  - {file_info['filename']}")
            logger.info(f"    大小: {file_info['size'] / 1024 / 1024:.2f} MB")
            if 'metadata' in file_info:
                meta = file_info['metadata']
                logger.info(f"    划分: {meta.get('split', 'unknown')}")
                logger.info(f"    样本数: {meta.get('num_samples', 'unknown')}")
    
    logger.info("示例3完成\n")


def example4_clear_cache():
    """示例4: 清除缓存"""
    logger.info("=" * 80)
    logger.info("示例4: 清除缓存")
    logger.info("=" * 80)
    
    # 创建缓存管理器
    cache_manager = DatasetCacheManager(
        dataset_name='mnist',
        version='v1',
        enable_cache=True
    )
    
    # 清除特定划分的缓存
    logger.info("清除训练集缓存...")
    count = cache_manager.clear(split='train')
    logger.info(f"清除了 {count} 个文件")
    
    # 清除所有缓存
    # logger.info("清除所有缓存...")
    # count = cache_manager.clear()
    # logger.info(f"清除了所有缓存")
    
    logger.info("示例4完成\n")


def example5_custom_cache_format():
    """示例5: 使用不同的缓存格式"""
    logger.info("=" * 80)
    logger.info("示例5: 使用不同的缓存格式")
    logger.info("=" * 80)
    
    dataset = MNISTDataset(
        root_dir=Path("data/mnist"),
        split='train',
        enable_cache=True
    )
    
    # 使用pickle格式（默认，适合Python对象）
    logger.info("使用pickle格式保存...")
    dataset.save_to_cache(format='pkl')
    
    # 使用torch格式（适合tensor数据）
    logger.info("使用torch格式保存...")
    dataset.save_to_cache(format='pt')
    
    # 从不同格式加载
    logger.info("从pickle格式加载...")
    if dataset.load_from_cache(format='pkl'):
        logger.info("成功!")
    
    logger.info("示例5完成\n")


def example6_force_rebuild():
    """示例6: 强制重建缓存"""
    logger.info("=" * 80)
    logger.info("示例6: 强制重建缓存")
    logger.info("=" * 80)
    
    # 即使缓存存在，也重新构建
    dataset = MNISTDataset(
        root_dir=Path("data/mnist"),
        split='train',
        enable_cache=True,
        force_rebuild_cache=True  # 强制重建
    )
    
    logger.info("强制重建缓存...")
    # 正常加载数据集...
    # dataset.load_data()  # 实际的数据加载逻辑
    
    # 保存新的缓存
    dataset.save_to_cache(format='pkl')
    
    logger.info("示例6完成\n")


def main():
    """运行所有示例"""
    logger.info("数据集缓存功能示例\n")
    
    try:
        # 运行示例（根据需要选择）
        # example1_basic_cache()
        # example2_load_from_cache()
        example3_cache_manager()
        # example4_clear_cache()
        # example5_custom_cache_format()
        # example6_force_rebuild()
        
    except Exception as e:
        logger.error(f"示例运行出错: {e}", exc_info=True)


if __name__ == "__main__":
    main()

