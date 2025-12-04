# DataLoader使用文档

## 概述

NeuroTrain提供了统一的DataLoader创建接口，支持自动采样配置、DDP和DeepSpeed兼容。

## 核心功能

- ✅ 使用 `CustomDataset.dataloader()` 方法创建DataLoader
- ✅ 自动处理采样配置
- ✅ 支持DDP和DeepSpeed
- ✅ 灵活的配置选项

## 基本使用

### 方式1: 使用 get_all_dataloader()

推荐方式，自动从配置创建所有DataLoader：

```python
from src.config import set_config
from src.dataset import get_all_dataloader

config = {
    "dataset": {
        "name": "cifar10",
        "root_dir": "data/cifar10",
        "config": {"download": True}
    },
    "train": {"batch_size": 32},
    "test": {"batch_size": 32},
    "dataloader": {
        "num_workers": 4,
        "shuffle": True,
        "pin_memory": True
    }
}

set_config(config)

# 创建所有DataLoader
train_loader, valid_loader, test_loader = get_all_dataloader(use_valid=True)
```

### 方式2: 使用 dataset.dataloader()

直接调用数据集的 `dataloader()` 方法：

```python
from src.dataset import get_dataset

dataset = get_dataset("train")

# 创建DataLoader
loader = dataset.dataloader(
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=False
)
```

### 方式3: 使用 get_dataloader()

使用工具函数创建DataLoader：

```python
from src.dataset import get_dataset, get_dataloader

dataset = get_dataset("train")

loader = get_dataloader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=False
)
```

## 配置选项

### 全局配置

在配置文件中设置DataLoader选项：

```yaml
dataloader:
  num_workers: 4
  shuffle: true
  pin_memory: true
  drop_last: false
```

### 按split配置

可以为不同的split设置不同的选项：

```yaml
dataloader:
  shuffle:
    train: true
    test: false
    valid: false
  num_workers:
    train: 4
    test: 2
    valid: 2
  pin_memory:
    train: true
    test: true
    valid: true
  drop_last:
    train: true
    test: false
    valid: false
```

## 参数说明

### batch_size

批次大小，必须为正整数：

```python
loader = dataset.dataloader(batch_size=32)
```

### shuffle

是否打乱数据，默认为 `True`（训练集）或 `False`（测试集/验证集）：

```python
loader = dataset.dataloader(batch_size=32, shuffle=True)
```

### num_workers

数据加载的进程数，默认为 `0`（单进程）：

```python
loader = dataset.dataloader(batch_size=32, num_workers=4)
```

**注意**: 
- `num_workers=0`: 单进程加载（主进程）
- `num_workers>0`: 多进程加载（推荐4-8）

### pin_memory

是否将数据固定到内存，默认为 `True`（GPU训练时推荐）：

```python
loader = dataset.dataloader(batch_size=32, pin_memory=True)
```

### drop_last

是否丢弃最后一个不完整的批次，默认为 `False`：

```python
loader = dataset.dataloader(batch_size=32, drop_last=True)
```

### collate_fn

自定义批次合并函数：

```python
def my_collate_fn(batch):
    # 自定义合并逻辑
    return batch

loader = dataset.dataloader(batch_size=32, collate_fn=my_collate_fn)
```

## DDP支持

DataLoader自动支持DDP训练：

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化DDP
dist.init_process_group(backend='nccl')

# 创建DataLoader（会自动使用DistributedSampler）
train_loader, _, test_loader = get_all_dataloader()

# 训练
model = DDP(model)
for epoch in range(epochs):
    for batch in train_loader:
        # 训练代码
        pass
```

## DeepSpeed支持

DataLoader与DeepSpeed兼容：

```python
import deepspeed

# 创建DataLoader
train_loader, _, test_loader = get_all_dataloader()

# 初始化DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=ds_config
)

# 训练
for epoch in range(epochs):
    for batch in train_loader:
        # 训练代码
        loss = model_engine(batch)
        model_engine.backward(loss)
        model_engine.step()
```

## 使用示例

### 示例1: 基础训练

```python
from src.config import set_config
from src.dataset import get_all_dataloader

config = {
    "dataset": {
        "name": "cifar10",
        "root_dir": "data/cifar10",
        "config": {"download": True}
    },
    "train": {"batch_size": 32},
    "test": {"batch_size": 32},
    "dataloader": {
        "num_workers": 4,
        "shuffle": True
    }
}

set_config(config)
train_loader, _, test_loader = get_all_dataloader()

# 训练循环
for epoch in range(epochs):
    for batch in train_loader:
        # 训练代码
        pass
```

### 示例2: 带采样配置

```python
config = {
    "dataset": {
        "name": "cifar10",
        "root_dir": "data/cifar10",
        "config": {"download": True},
        "sample_ratio": {"train": 0.1, "test": 0.2}
    },
    "train": {"batch_size": 32},
    "test": {"batch_size": 32},
    "dataloader": {"num_workers": 4}
}

set_config(config)
train_loader, _, test_loader = get_all_dataloader()
```

### 示例3: 自定义配置

```python
from src.dataset import get_dataset

dataset = get_dataset("train")

loader = dataset.dataloader(
    batch_size=64,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    drop_last=True
)
```

## 性能优化建议

1. **num_workers**: 
   - CPU密集型任务：`num_workers = CPU核心数`
   - I/O密集型任务：`num_workers = 2-4`

2. **pin_memory**: 
   - GPU训练时设置为 `True`
   - CPU训练时设置为 `False`

3. **prefetch**: 
   - 大数据集时启用预读取
   - 小数据集时可能不需要

4. **batch_size**: 
   - 根据GPU内存调整
   - 使用梯度累积模拟更大的batch_size

## 故障排除

### 问题1: 数据加载慢

**可能原因**:
- `num_workers` 设置过小
- 数据预处理复杂
- I/O瓶颈

**解决方案**:
- 增加 `num_workers`
- 优化数据预处理
- 使用SSD存储数据

### 问题2: 内存不足

**可能原因**:
- `num_workers` 设置过大
- `pin_memory=True` 占用内存
- `batch_size` 过大

**解决方案**:
- 减少 `num_workers`
- 设置 `pin_memory=False`
- 减小 `batch_size`

### 问题3: DDP训练数据不同步

**可能原因**:
- 未使用 `DistributedSampler`
- 随机种子不一致

**解决方案**:
- 确保使用 `get_all_dataloader()` 自动处理
- 设置统一的随机种子

## 相关文档

- [数据集采样配置](dataset_sampling.md)
- [DDP训练指南](ddp_training.md)
- [DeepSpeed训练指南](deepspeed_training.md)

