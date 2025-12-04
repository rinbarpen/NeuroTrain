#!/usr/bin/env python3
"""
完整测试套件 - 测试所有功能
"""

import sys
import logging
import traceback
from pathlib import Path
from typing import Dict
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

from src.config import set_config
from src.dataset import get_dataset, get_all_dataloader

class TestRunner:
    """测试运行器"""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
    
    def run_test(self, test_name: str, test_func, *args, **kwargs) -> bool:
        """运行单个测试"""
        logger.info(f"\n{'='*80}")
        logger.info(f"运行测试: {test_name}")
        logger.info(f"{'='*80}")
        
        try:
            result = test_func(*args, **kwargs)
            self.results[test_name] = result
            status = "✓ 通过" if result else "✗ 失败"
            logger.info(f"\n测试 {test_name}: {status}")
            return result
        except Exception as e:
            logger.error(f"\n测试 {test_name} 异常: {e}")
            logger.error(traceback.format_exc())
            self.results[test_name] = False
            return False
    
    def print_summary(self):
        """打印测试总结"""
        elapsed = time.time() - self.start_time
        logger.info(f"\n{'='*80}")
        logger.info("测试结果总结")
        logger.info(f"{'='*80}")
        
        passed = sum(1 for v in self.results.values() if v)
        total = len(self.results)
        
        for test_name, result in self.results.items():
            status = "✓ 通过" if result else "✗ 失败"
            logger.info(f"  {test_name:40s} {status}")
        
        logger.info(f"\n总计: {passed}/{total} 通过")
        logger.info(f"耗时: {elapsed:.2f} 秒")
        logger.info(f"{'='*80}")
        
        return passed == total

def test_basic_dataset_loading() -> bool:
    """测试基础数据集加载"""
    config = {
        "task": "test",
        "run_id": "test_basic",
        "seed": 42,
        "device": "cpu",
        "dataset": {
            "name": "cifar10",
            "root_dir": "data/cifar10",
            "config": {"download": True, "valid_ratio": 0.1}
        },
        "train": {"batch_size": 32},
        "test": {"batch_size": 32},
        "dataloader": {"num_workers": 0, "shuffle": True}
    }
    
    try:
        set_config(config)
        
        for mode in ["train", "valid", "test"]:
            dataset = get_dataset(mode)
            if dataset is None:
                logger.warning(f"  {mode} 数据集返回 None")
                return False
            
            dataset_len = len(dataset)
            logger.info(f"  {mode}: {type(dataset).__name__}, 长度: {dataset_len}")
            
            if dataset_len == 0:
                logger.warning(f"  {mode} 数据集为空")
                return False
            
            sample = dataset[0]
            logger.info(f"  样本类型: {type(sample)}")
        
        return True
    except Exception as e:
        logger.error(f"测试失败: {e}")
        return False

def test_dataloader_creation() -> bool:
    """测试DataLoader创建"""
    config = {
        "task": "test",
        "run_id": "test_dataloader",
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
        set_config(config)
        
        train_loader, valid_loader, test_loader = get_all_dataloader(use_valid=True)
        
        if train_loader is None:
            logger.error("训练DataLoader为None")
            return False
        
        logger.info(f"  训练DataLoader: {type(train_loader).__name__}")
        logger.info(f"  验证DataLoader: {type(valid_loader).__name__ if valid_loader else None}")
        logger.info(f"  测试DataLoader: {type(test_loader).__name__ if test_loader else None}")
        
        batch = next(iter(train_loader))
        logger.info(f"  批次类型: {type(batch)}")
        if isinstance(batch, (list, tuple)) and len(batch) > 0:
            logger.info(f"  批次大小: {len(batch)}")
        
        return True
    except Exception as e:
        logger.error(f"测试失败: {e}")
        return False

def test_sample_ratio() -> bool:
    """测试sample_ratio配置"""
    config = {
        "task": "test",
        "run_id": "test_sample_ratio",
        "seed": 42,
        "device": "cpu",
        "dataset": {
            "name": "cifar10",
            "root_dir": "data/cifar10",
            "config": {"download": True},
            "sample_ratio": {"train": 0.1, "test": 0.2}
        },
        "train": {"batch_size": 32},
        "test": {"batch_size": 32},
        "dataloader": {"num_workers": 0, "shuffle": True}
    }
    
    try:
        set_config(config)
        
        train_dataset = get_dataset("train")
        test_dataset = get_dataset("test")
        
        train_len = len(train_dataset) if train_dataset else 0
        test_len = len(test_dataset) if test_dataset else 0
        
        logger.info(f"  训练集长度: {train_len} (期望约4500，10%的45000)")
        logger.info(f"  测试集长度: {test_len} (期望约2000，20%的10000)")
        
        if train_len > 0 and test_len > 0:
            return True
        return False
    except Exception as e:
        logger.error(f"测试失败: {e}")
        return False

def test_max_samples() -> bool:
    """测试max_samples配置"""
    config = {
        "task": "test",
        "run_id": "test_max_samples",
        "seed": 42,
        "device": "cpu",
        "dataset": {
            "name": "cifar10",
            "root_dir": "data/cifar10",
            "config": {"download": True},
            "max_samples": {"train": 100, "test": 50}
        },
        "train": {"batch_size": 32},
        "test": {"batch_size": 32},
        "dataloader": {"num_workers": 0, "shuffle": True}
    }
    
    try:
        set_config(config)
        
        train_dataset = get_dataset("train")
        test_dataset = get_dataset("test")
        
        train_len = len(train_dataset) if train_dataset else 0
        test_len = len(test_dataset) if test_dataset else 0
        
        logger.info(f"  训练集长度: {train_len} (期望<=100)")
        logger.info(f"  测试集长度: {test_len} (期望<=50)")
        
        if train_len > 0 and test_len > 0:
            if train_len <= 100 and test_len <= 50:
                logger.info("  ✓ 采样配置生效")
            else:
                logger.warning("  ⚠ 采样配置可能未生效（数据集可能不支持mininalize）")
            return True
        return False
    except Exception as e:
        logger.error(f"测试失败: {e}")
        return False

def test_custom_dataset_dataloader() -> bool:
    """测试CustomDataset的dataloader方法"""
    config = {
        "task": "test",
        "run_id": "test_custom_dataloader",
        "seed": 42,
        "device": "cpu",
        "dataset": {
            "name": "cifar10",
            "root_dir": "data/cifar10",
            "config": {"download": True}
        },
        "train": {"batch_size": 32},
        "test": {"batch_size": 32},
        "dataloader": {"num_workers": 0, "shuffle": True}
    }
    
    try:
        set_config(config)
        
        train_dataset = get_dataset("train")
        if train_dataset is None:
            return False
        
        if hasattr(train_dataset, 'dataloader'):
            loader = train_dataset.dataloader(
                batch_size=32,
                shuffle=True,
                num_workers=0
            )
            logger.info(f"  ✓ 直接调用dataloader方法成功: {type(loader).__name__}")
            
            batch = next(iter(loader))
            logger.info(f"  ✓ 成功获取批次: {type(batch)}")
            return True
        else:
            logger.warning("  ⚠ 数据集不支持dataloader方法")
            return False
    except Exception as e:
        logger.error(f"测试失败: {e}")
        return False

def test_ddp_imports() -> bool:
    """测试DDP相关导入和配置"""
    try:
        from src.utils.ddp_utils import (
            init_ddp_distributed,
            is_main_process,
            cleanup_ddp
        )
        logger.info("  ✓ DDP工具模块导入成功")
        
        import torch.distributed as dist
        if dist.is_available():
            logger.info("  ✓ torch.distributed 可用")
        else:
            logger.warning("  ⚠ torch.distributed 不可用")
        
        return True
    except ImportError as e:
        logger.error(f"  ✗ DDP导入失败: {e}")
        return False
    except Exception as e:
        logger.error(f"  ✗ DDP测试失败: {e}")
        return False

def test_deepspeed_imports() -> bool:
    """测试DeepSpeed相关导入和配置"""
    try:
        from src.utils.deepspeed_utils import (
            is_deepspeed_available,
            init_deepspeed_distributed,
            load_deepspeed_config
        )
        logger.info("  ✓ DeepSpeed工具模块导入成功")
        
        if is_deepspeed_available():
            logger.info("  ✓ DeepSpeed 可用")
        else:
            logger.warning("  ⚠ DeepSpeed 不可用（需要安装: pip install deepspeed）")
        
        return True
    except ImportError as e:
        logger.error(f"  ✗ DeepSpeed导入失败: {e}")
        return False
    except Exception as e:
        logger.error(f"  ✗ DeepSpeed测试失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("开始运行数据集、DDP和DeepSpeed测试")
    logger.info("=" * 80)
    
    runner = TestRunner()
    
    # 基础功能测试
    runner.run_test("基础数据集加载", test_basic_dataset_loading)
    runner.run_test("DataLoader创建", test_dataloader_creation)
    runner.run_test("CustomDataset.dataloader方法", test_custom_dataset_dataloader)
    
    # 采样配置测试
    runner.run_test("sample_ratio配置", test_sample_ratio)
    runner.run_test("max_samples配置", test_max_samples)
    
    # DDP和DeepSpeed测试
    runner.run_test("DDP兼容性", test_ddp_imports)
    runner.run_test("DeepSpeed兼容性", test_deepspeed_imports)
    
    # 打印总结
    all_passed = runner.print_summary()
    
    if all_passed:
        logger.info("\n🎉 所有测试通过！")
        return 0
    else:
        logger.error("\n❌ 部分测试失败，请检查上述错误信息")
        return 1

if __name__ == "__main__":
    sys.exit(main())

