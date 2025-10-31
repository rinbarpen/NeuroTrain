#!/usr/bin/env python3
"""
数据集缓存管理工具

用于管理数据集缓存的命令行工具，支持查看、清除、验证等操作。

使用示例:
    # 查看所有缓存信息
    python tools/dataset_cache_tool.py list
    
    # 查看特定数据集的缓存
    python tools/dataset_cache_tool.py info mnist
    
    # 清除特定数据集的缓存
    python tools/dataset_cache_tool.py clear mnist --split train
    
    # 清除所有缓存
    python tools/dataset_cache_tool.py clear-all
    
    # 验证缓存完整性
    python tools/dataset_cache_tool.py verify mnist
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.dataset.cache_manager import DatasetCacheManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def list_all_caches(cache_root: Path):
    """列出所有缓存的数据集"""
    logger.info("=" * 80)
    logger.info("所有缓存数据集")
    logger.info("=" * 80)
    
    datasets_dir = cache_root / "datasets"
    if not datasets_dir.exists():
        logger.warning(f"缓存目录不存在: {datasets_dir}")
        return
    
    total_size = 0
    dataset_count = 0
    
    for dataset_dir in datasets_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        dataset_name = dataset_dir.name
        dataset_count += 1
        
        logger.info(f"\n数据集: {dataset_name}")
        
        # 遍历版本目录
        for version_dir in dataset_dir.iterdir():
            if not version_dir.is_dir():
                continue
            
            version = version_dir.name
            cache_manager = DatasetCacheManager(
                dataset_name=dataset_name,
                cache_root=cache_root,
                version=version,
                enable_cache=True
            )
            
            info = cache_manager.get_cache_info()
            
            logger.info(f"  版本: {version}")
            logger.info(f"  缓存目录: {info['cache_dir']}")
            logger.info(f"  文件数: {info.get('total_files', 0)}")
            logger.info(f"  大小: {info.get('total_size_mb', 0):.2f} MB")
            
            total_size += info.get('total_size', 0)
    
    logger.info(f"\n总计:")
    logger.info(f"  数据集数: {dataset_count}")
    logger.info(f"  总大小: {total_size / 1024 / 1024:.2f} MB")


def show_dataset_info(
    dataset_name: str,
    cache_root: Path,
    version: str = "v1"
):
    """显示特定数据集的缓存信息"""
    logger.info("=" * 80)
    logger.info(f"数据集缓存信息: {dataset_name}")
    logger.info("=" * 80)
    
    cache_manager = DatasetCacheManager(
        dataset_name=dataset_name,
        cache_root=cache_root,
        version=version,
        enable_cache=True
    )
    
    info = cache_manager.get_cache_info()
    
    logger.info(f"数据集: {info['dataset_name']}")
    logger.info(f"版本: {info['version']}")
    logger.info(f"缓存目录: {info['cache_dir']}")
    logger.info(f"状态: {'启用' if info['enabled'] else '禁用'}")
    logger.info(f"文件数: {info.get('total_files', 0)}")
    logger.info(f"总大小: {info.get('total_size_mb', 0):.2f} MB")
    
    if info.get('files'):
        logger.info("\n缓存文件详情:")
        for file_info in info['files']:
            logger.info(f"\n  文件: {file_info['filename']}")
            logger.info(f"  大小: {file_info['size'] / 1024 / 1024:.2f} MB")
            logger.info(f"  修改时间: {file_info['modified']}")
            
            if 'metadata' in file_info:
                meta = file_info['metadata']
                logger.info(f"  划分: {meta.get('split', 'unknown')}")
                logger.info(f"  样本数: {meta.get('num_samples', 'unknown')}")
                logger.info(f"  数据集类: {meta.get('dataset_class', 'unknown')}")
                logger.info(f"  创建时间: {meta.get('created_at', 'unknown')}")


def clear_cache(
    dataset_name: str,
    cache_root: Path,
    version: str = "v1",
    split: Optional[str] = None
):
    """清除特定数据集的缓存"""
    logger.info("=" * 80)
    logger.info(f"清除缓存: {dataset_name}")
    logger.info("=" * 80)
    
    cache_manager = DatasetCacheManager(
        dataset_name=dataset_name,
        cache_root=cache_root,
        version=version,
        enable_cache=True
    )
    
    if split:
        logger.info(f"清除划分: {split}")
        count = cache_manager.clear(split=split)
    else:
        logger.info("清除所有缓存")
        count = cache_manager.clear()
    
    logger.info(f"已清除 {count} 个文件")


def clear_all_caches(cache_root: Path):
    """清除所有缓存"""
    logger.info("=" * 80)
    logger.info("清除所有缓存")
    logger.info("=" * 80)
    
    datasets_dir = cache_root / "datasets"
    if not datasets_dir.exists():
        logger.warning(f"缓存目录不存在: {datasets_dir}")
        return
    
    # 询问确认
    response = input(f"确定要清除所有缓存吗？这将删除 {datasets_dir} 下的所有文件 (y/N): ")
    if response.lower() != 'y':
        logger.info("取消操作")
        return
    
    import shutil
    shutil.rmtree(datasets_dir)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("已清除所有缓存")


def verify_cache(
    dataset_name: str,
    cache_root: Path,
    version: str = "v1"
):
    """验证缓存完整性"""
    logger.info("=" * 80)
    logger.info(f"验证缓存: {dataset_name}")
    logger.info("=" * 80)
    
    cache_manager = DatasetCacheManager(
        dataset_name=dataset_name,
        cache_root=cache_root,
        version=version,
        enable_cache=True
    )
    
    info = cache_manager.get_cache_info()
    
    if not info.get('files'):
        logger.warning("没有缓存文件")
        return
    
    valid_count = 0
    invalid_count = 0
    
    for file_info in info['files']:
        cache_path = cache_root / dataset_name / version / file_info['filename']
        
        if cache_manager._check_validity(cache_path):
            logger.info(f"✓ {file_info['filename']} - 有效")
            valid_count += 1
        else:
            logger.warning(f"✗ {file_info['filename']} - 无效")
            invalid_count += 1
    
    logger.info(f"\n验证结果:")
    logger.info(f"  有效: {valid_count}")
    logger.info(f"  无效: {invalid_count}")
    logger.info(f"  总计: {valid_count + invalid_count}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="数据集缓存管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s list
  %(prog)s info mnist
  %(prog)s clear mnist --split train
  %(prog)s clear-all
  %(prog)s verify mnist
        """
    )
    
    parser.add_argument(
        '--cache-root',
        type=Path,
        default=Path.cwd() / 'cache',
        help='缓存根目录 (默认: ./cache)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # list 命令
    subparsers.add_parser('list', help='列出所有缓存')
    
    # info 命令
    info_parser = subparsers.add_parser('info', help='显示数据集缓存信息')
    info_parser.add_argument('dataset', help='数据集名称')
    info_parser.add_argument('--version', default='v1', help='缓存版本')
    
    # clear 命令
    clear_parser = subparsers.add_parser('clear', help='清除数据集缓存')
    clear_parser.add_argument('dataset', help='数据集名称')
    clear_parser.add_argument('--version', default='v1', help='缓存版本')
    clear_parser.add_argument('--split', help='数据集划分 (train/valid/test)')
    
    # clear-all 命令
    subparsers.add_parser('clear-all', help='清除所有缓存')
    
    # verify 命令
    verify_parser = subparsers.add_parser('verify', help='验证缓存完整性')
    verify_parser.add_argument('dataset', help='数据集名称')
    verify_parser.add_argument('--version', default='v1', help='缓存版本')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'list':
            list_all_caches(args.cache_root)
        elif args.command == 'info':
            show_dataset_info(args.dataset, args.cache_root, args.version)
        elif args.command == 'clear':
            clear_cache(args.dataset, args.cache_root, args.version, args.split)
        elif args.command == 'clear-all':
            clear_all_caches(args.cache_root)
        elif args.command == 'verify':
            verify_cache(args.dataset, args.cache_root, args.version)
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"执行命令时出错: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

