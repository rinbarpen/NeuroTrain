#!/usr/bin/env python3
"""
数据集预读取功能演示

展示如何使用预读取功能来加速数据加载。
"""

import sys
import time
from pathlib import Path
import torch

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


def demo_without_prefetch():
    """演示不使用预读取"""
    logger.info("=" * 80)
    logger.info("场景1: 不使用预读取")
    logger.info("=" * 80)
    
    dataset = MNISTDataset(
        root_dir=Path("data/mnist"),
        split='train'
    )
    
    # 创建DataLoader（不启用预读取）
    dataloader = dataset.dataloader(
        batch_size=32,
        shuffle=False,
        num_workers=0,
        enable_prefetch=False  # 禁用预读取
    )
    
    logger.info(f"数据集大小: {len(dataset)}")
    logger.info(f"Batch数: {len(dataloader)}")
    
    # 计时
    start_time = time.time()
    total_samples = 0
    
    for batch_idx, batch in enumerate(dataloader):
        total_samples += batch['image'].size(0)
        
        # 模拟训练处理时间
        time.sleep(0.001)
        
        if batch_idx >= 99:  # 只测试前100个batch
            break
    
    elapsed = time.time() - start_time
    logger.info(f"处理 {batch_idx + 1} 个batch，共 {total_samples} 个样本")
    logger.info(f"总时间: {elapsed:.4f} 秒")
    logger.info(f"平均每batch: {elapsed / (batch_idx + 1) * 1000:.2f} ms")
    logger.info("")
    
    return elapsed


def demo_with_prefetch():
    """演示使用预读取"""
    logger.info("=" * 80)
    logger.info("场景2: 使用预读取")
    logger.info("=" * 80)
    
    dataset = MNISTDataset(
        root_dir=Path("data/mnist"),
        split='train'
    )
    
    # 创建DataLoader（启用预读取）
    dataloader = dataset.dataloader(
        batch_size=32,
        shuffle=False,
        num_workers=0,
        enable_prefetch=True,  # 启用预读取
        prefetch_buffer_size=4  # 预读取4个样本
    )
    
    logger.info(f"数据集大小: {len(dataset)}")
    logger.info(f"Batch数: {len(dataloader)}")
    
    # 计时
    start_time = time.time()
    total_samples = 0
    
    for batch_idx, batch in enumerate(dataloader):
        total_samples += batch['image'].size(0)
        
        # 模拟训练处理时间
        time.sleep(0.001)
        
        if batch_idx >= 99:  # 只测试前100个batch
            break
    
    elapsed = time.time() - start_time
    logger.info(f"处理 {batch_idx + 1} 个batch，共 {total_samples} 个样本")
    logger.info(f"总时间: {elapsed:.4f} 秒")
    logger.info(f"平均每batch: {elapsed / (batch_idx + 1) * 1000:.2f} ms")
    logger.info("")
    
    return elapsed


def demo_prefetch_with_shuffle():
    """演示带shuffle的预读取"""
    logger.info("=" * 80)
    logger.info("场景3: 使用预读取 + Shuffle")
    logger.info("=" * 80)
    
    dataset = MNISTDataset(
        root_dir=Path("data/mnist"),
        split='train'
    )
    
    # 创建DataLoader（启用预读取 + shuffle）
    dataloader = dataset.dataloader(
        batch_size=32,
        shuffle=True,  # 启用shuffle
        num_workers=0,
        enable_prefetch=True,
        prefetch_buffer_size=4
    )
    
    logger.info(f"数据集大小: {len(dataset)}")
    logger.info(f"Batch数: {len(dataloader)}")
    
    # 计时
    start_time = time.time()
    total_samples = 0
    
    for batch_idx, batch in enumerate(dataloader):
        total_samples += batch['image'].size(0)
        
        # 模拟训练处理时间
        time.sleep(0.001)
        
        if batch_idx >= 99:
            break
    
    elapsed = time.time() - start_time
    logger.info(f"处理 {batch_idx + 1} 个batch，共 {total_samples} 个样本")
    logger.info(f"总时间: {elapsed:.4f} 秒")
    logger.info(f"平均每batch: {elapsed / (batch_idx + 1) * 1000:.2f} ms")
    logger.info("")
    
    return elapsed


def demo_different_buffer_sizes():
    """演示不同的缓冲区大小"""
    logger.info("=" * 80)
    logger.info("场景4: 不同的缓冲区大小对比")
    logger.info("=" * 80)
    
    buffer_sizes = [1, 2, 4, 8]
    results = {}
    
    for buffer_size in buffer_sizes:
        logger.info(f"\n测试缓冲区大小: {buffer_size}")
        logger.info("-" * 40)
        
        dataset = MNISTDataset(
            root_dir=Path("data/mnist"),
            split='train'
        )
        
        dataloader = dataset.dataloader(
            batch_size=32,
            shuffle=False,
            num_workers=0,
            enable_prefetch=True,
            prefetch_buffer_size=buffer_size
        )
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            time.sleep(0.001)
            if batch_idx >= 99:
                break
        
        elapsed = time.time() - start_time
        results[buffer_size] = elapsed
        
        logger.info(f"时间: {elapsed:.4f} 秒")
    
    logger.info("\n" + "=" * 80)
    logger.info("缓冲区大小对比总结")
    logger.info("=" * 80)
    for size, time_taken in results.items():
        logger.info(f"缓冲区大小 {size}: {time_taken:.4f} 秒")


def demo_manual_prefetch():
    """演示手动使用预读取包装器"""
    logger.info("=" * 80)
    logger.info("场景5: 手动使用预读取包装器")
    logger.info("=" * 80)
    
    from src.dataset.prefetch_wrapper import PrefetchDataset, SequentialPrefetchDataset
    
    dataset = MNISTDataset(
        root_dir=Path("data/mnist"),
        split='train'
    )
    
    # 方式1: 使用通用预读取包装器
    logger.info("\n使用PrefetchDataset包装器...")
    prefetch_dataset = PrefetchDataset(
        dataset,
        buffer_size=4,
        enable_prefetch=True
    )
    
    start_time = time.time()
    for i in range(100):
        sample = prefetch_dataset[i]
        time.sleep(0.0001)  # 模拟处理
    elapsed1 = time.time() - start_time
    logger.info(f"PrefetchDataset: {elapsed1:.4f} 秒")
    
    # 停止预读取
    prefetch_dataset.stop_prefetch()
    
    # 方式2: 使用顺序预读取包装器
    logger.info("\n使用SequentialPrefetchDataset包装器...")
    seq_prefetch_dataset = SequentialPrefetchDataset(
        dataset,
        buffer_size=4,
        enable_prefetch=True
    )
    
    start_time = time.time()
    for i in range(100):
        sample = seq_prefetch_dataset[i]
        time.sleep(0.0001)
    elapsed2 = time.time() - start_time
    logger.info(f"SequentialPrefetchDataset: {elapsed2:.4f} 秒")
    
    seq_prefetch_dataset.stop_prefetch()


def show_usage():
    """显示使用方法"""
    logger.info("\n" + "=" * 80)
    logger.info("预读取功能使用说明")
    logger.info("=" * 80)
    
    usage = """
预读取功能可以在模型处理当前batch时，提前加载下一个batch的数据。

1. 【在DataLoader中启用】（推荐）

   dataloader = dataset.dataloader(
       batch_size=32,
       shuffle=False,
       enable_prefetch=True,        # 启用预读取
       prefetch_buffer_size=4       # 预读取4个样本
   )

2. 【手动包装数据集】

   from src.dataset.prefetch_wrapper import PrefetchDataset
   
   prefetch_dataset = PrefetchDataset(
       dataset,
       buffer_size=4,
       enable_prefetch=True
   )
   
   dataloader = DataLoader(prefetch_dataset, batch_size=32)

3. 【配置建议】

   - buffer_size: 2-8 之间合适，太大占用内存，太小效果不明显
   - 顺序访问（不shuffle）: 预读取效果最好
   - 随机访问（shuffle）: 预读取仍有效，但效果略差
   - CPU密集型预处理: 预读取效果显著
   - IO密集型加载: 预读取效果更明显

4. 【性能提示】

   - 预读取会占用额外内存
   - 预读取线程与主线程并发运行
   - 适合数据加载是瓶颈的场景
   - 与num_workers可以配合使用
    """
    logger.info(usage)


def main():
    """主函数"""
    logger.info("\n" + "=" * 80)
    logger.info("数据集预读取功能演示")
    logger.info("=" * 80)
    
    try:
        # 运行演示
        time1 = demo_without_prefetch()
        time2 = demo_with_prefetch()
        
        logger.info("=" * 80)
        logger.info("性能对比")
        logger.info("=" * 80)
        logger.info(f"不使用预读取: {time1:.4f} 秒")
        logger.info(f"使用预读取:   {time2:.4f} 秒")
        if time1 > time2:
            logger.info(f"加速比: {time1/time2:.2f}x")
        logger.info("")
        
        # 其他演示
        # demo_prefetch_with_shuffle()
        # demo_different_buffer_sizes()
        # demo_manual_prefetch()
        # show_usage()
        
        logger.info("=" * 80)
        logger.info("演示完成！")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"演示过程中出错: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

