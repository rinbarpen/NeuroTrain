# 测试文档

## 概述

本文档介绍如何运行和编写测试，验证数据集、DataLoader、DDP和DeepSpeed功能。

## 快速开始

### 运行快速测试

```bash
conda activate ntrain
python scripts/run_quick_test.py
```

### 运行完整测试

```bash
conda activate ntrain
python scripts/run_tests.py
```

## 测试脚本

### run_quick_test.py

快速验证核心功能，运行时间约30秒：

- ✅ 数据集加载
- ✅ DataLoader创建
- ✅ 采样配置

### run_tests.py

完整测试套件，运行时间约2-5分钟：

- ✅ 基础数据集加载
- ✅ DataLoader创建
- ✅ CustomDataset.dataloader方法
- ✅ sample_ratio配置
- ✅ max_samples配置
- ✅ 嵌套sampling配置
- ✅ 多数据集支持
- ✅ DDP兼容性
- ✅ DeepSpeed兼容性

## 测试内容

### 1. 数据集加载测试

验证数据集能够正常加载：

```python
from src.config import set_config
from src.dataset import get_dataset

config = {
    "dataset": {
        "name": "cifar10",
        "root_dir": "data/cifar10",
        "config": {"download": True}
    }
}

set_config(config)
dataset = get_dataset("train")
assert dataset is not None
assert len(dataset) > 0
```

### 2. DataLoader创建测试

验证DataLoader能够正常创建：

```python
from src.dataset import get_all_dataloader

train_loader, _, test_loader = get_all_dataloader()
assert train_loader is not None
assert test_loader is not None

# 测试获取批次
batch = next(iter(train_loader))
assert batch is not None
```

### 3. 采样配置测试

验证采样配置是否生效：

```python
config = {
    "dataset": {
        "name": "cifar10",
        "root_dir": "data/cifar10",
        "config": {"download": True},
        "sample_ratio": {"train": 0.1}
    }
}

set_config(config)
dataset = get_dataset("train")
original_len = 45000  # CIFAR-10训练集原始长度
sampled_len = len(dataset)
assert sampled_len <= original_len * 0.1 + 100  # 允许误差
```

### 4. DDP兼容性测试

检查DDP相关功能：

```python
from src.utils.ddp_utils import init_ddp_distributed, is_main_process

# 检查DDP是否可用
try:
    import torch.distributed as dist
    assert dist.is_available()
    print("✓ DDP可用")
except:
    print("⚠ DDP不可用（单GPU环境正常）")
```

### 5. DeepSpeed兼容性测试

检查DeepSpeed相关功能：

```python
from src.utils.deepspeed_utils import is_deepspeed_available

if is_deepspeed_available():
    print("✓ DeepSpeed可用")
else:
    print("⚠ DeepSpeed不可用（需要安装: pip install deepspeed）")
```

## 编写测试

### 测试文件结构

```python
#!/usr/bin/env python3
"""测试模块说明"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import set_config
from src.dataset import get_dataset

def test_basic_functionality():
    """测试基础功能"""
    config = {
        "dataset": {
            "name": "cifar10",
            "root_dir": "data/cifar10",
            "config": {"download": True}
        }
    }
    set_config(config)
    dataset = get_dataset("train")
    assert dataset is not None
    assert len(dataset) > 0

if __name__ == "__main__":
    test_basic_functionality()
    print("✓ 测试通过")
```

### 测试最佳实践

1. **独立性**: 每个测试应该独立，不依赖其他测试
2. **可重复性**: 使用固定随机种子确保可重复
3. **清理**: 测试后清理临时文件和配置
4. **错误处理**: 提供清晰的错误信息

## 持续集成

### GitHub Actions示例

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: conda-incubator/setup-miniconda@v2
        with:
          python-version: 3.10
      - name: Install dependencies
        run: |
          conda activate ntrain
          pip install -e .
      - name: Run tests
        run: |
          conda activate ntrain
          python scripts/run_quick_test.py
```

## 测试覆盖率

当前测试覆盖：

- ✅ 数据集加载
- ✅ DataLoader创建
- ✅ 采样配置
- ✅ DDP兼容性
- ✅ DeepSpeed兼容性
- ⚠️ 部分数据集类型（待补充）

## 故障排除

### 测试失败

1. **检查环境**: 确认已激活正确的conda环境
2. **检查数据**: 确认数据集已下载或可访问
3. **查看日志**: 检查测试输出中的错误信息

### 测试超时

1. **减少数据量**: 使用采样配置减少测试数据
2. **优化测试**: 移除不必要的测试步骤
3. **并行测试**: 使用pytest并行运行

## 相关文档

- [数据集采样配置](dataset_sampling.md)
- [DataLoader使用](dataloader_usage.md)

