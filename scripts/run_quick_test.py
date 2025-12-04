#!/usr/bin/env python3
"""
快速测试脚本 - 测试核心功能
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import set_config
from src.dataset import get_dataset, get_all_dataloader

def main():
    print("=" * 80)
    print("快速测试数据集功能")
    print("=" * 80)
    
    # 测试配置
    config = {
        "task": "test",
        "run_id": "quick_test",
        "seed": 42,
        "device": "cpu",
        "dataset": {
            "name": "cifar10",
            "root_dir": "data/cifar10",
            "config": {"download": True, "valid_ratio": 0.1}
        },
        "train": {"batch_size": 32},
        "test": {"batch_size": 32},
        "valid": {"batch_size": 32},
        "dataloader": {"num_workers": 0, "shuffle": True}
    }
    
    try:
        # 1. 测试配置设置
        set_config(config)
        print("✓ 配置设置成功")
        
        # 2. 测试数据集加载
        print("\n测试数据集加载...")
        train_dataset = get_dataset("train")
        test_dataset = get_dataset("test")
        valid_dataset = get_dataset("valid")
        
        print(f"✓ 训练集: {type(train_dataset).__name__}, 长度: {len(train_dataset) if train_dataset else 0}")
        print(f"✓ 测试集: {type(test_dataset).__name__}, 长度: {len(test_dataset) if test_dataset else 0}")
        print(f"✓ 验证集: {type(valid_dataset).__name__}, 长度: {len(valid_dataset) if valid_dataset else 0}")
        
        # 3. 测试DataLoader创建
        print("\n测试DataLoader创建...")
        train_loader, valid_loader, test_loader = get_all_dataloader(use_valid=True)
        print(f"✓ 训练DataLoader: {type(train_loader).__name__ if train_loader else None}")
        print(f"✓ 验证DataLoader: {type(valid_loader).__name__ if valid_loader else None}")
        print(f"✓ 测试DataLoader: {type(test_loader).__name__ if test_loader else None}")
        
        # 4. 测试获取批次
        if train_loader:
            batch = next(iter(train_loader))
            print(f"✓ 成功获取训练批次: {type(batch)}")
        
        # 5. 测试采样配置
        print("\n测试采样配置...")
        config_with_sampling = config.copy()
        config_with_sampling["dataset"] = config["dataset"].copy()
        config_with_sampling["dataset"]["sample_ratio"] = {"train": 0.1, "test": 0.2}
        set_config(config_with_sampling)
        
        train_dataset_sampled = get_dataset("train")
        test_dataset_sampled = get_dataset("test")
        print(f"✓ 采样后训练集长度: {len(train_dataset_sampled) if train_dataset_sampled else 0}")
        print(f"✓ 采样后测试集长度: {len(test_dataset_sampled) if test_dataset_sampled else 0}")
        
        print("\n" + "=" * 80)
        print("所有测试通过！")
        print("=" * 80)
        return 0
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

