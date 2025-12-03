# 数据集采样配置文档

## 概述

NeuroTrain支持灵活的数据集采样配置，可以通过 `sample_ratio`（按比例）或 `max_samples`（按数量）来控制训练数据量。这对于快速实验、调试和资源受限的场景非常有用。

## 功能特性

- ✅ 支持 `sample_ratio` 按比例采样（0.0-1.0）
- ✅ 支持 `max_samples` 按数量采样（正整数）
- ✅ 支持嵌套 `sampling` 配置
- ✅ 支持不同split（train/test/valid）的独立配置
- ✅ 自动处理采样和DataLoader创建

## 配置方式

### 方式1: 直接配置

在数据集配置中直接指定采样参数：

```yaml
dataset:
  name: cifar10
  root_dir: data/cifar10
  config:
    download: true
    valid_ratio: 0.1
  # 按比例采样
  sample_ratio:
    train: 0.1  # 训练集使用10%
    test: 0.2   # 测试集使用20%
  # 或按数量采样
  max_samples:
    train: 100  # 训练集最多100个样本
    test: 50    # 测试集最多50个样本
```

### 方式2: 嵌套配置

使用 `sampling` 子配置：

```yaml
dataset:
  name: cifar10
  root_dir: data/cifar10
  config:
    download: true
  sampling:
    sample_ratio:
      train: 0.05
    max_samples:
      test: 20
    sample_shuffle:
      train: true
      test: false
```

### 方式3: TOML配置

```toml
[dataset]
name = "cifar10"
root_dir = "data/cifar10"

[dataset.config]
download = true
valid_ratio = 0.1

[dataset.sample_ratio]
train = 0.1
test = 0.2

[dataset.max_samples]
train = 100
test = 50
```

## 配置参数说明

### sample_ratio

按比例采样数据集，取值范围 `0.0-1.0`：

- `0.0 < sample_ratio < 1.0`: 使用指定比例的数据
- `sample_ratio = 1.0`: 使用全部数据（等同于不采样）
- `sample_ratio = 0.0`: 无效配置，会被忽略

**示例**:
```yaml
sample_ratio:
  train: 0.1  # 使用训练集的10%
  test: 0.2   # 使用测试集的20%
```

### max_samples

按数量采样数据集，必须是正整数：

- `max_samples > 0`: 使用最多指定数量的样本
- 如果数据集总数小于 `max_samples`，则使用全部数据

**示例**:
```yaml
max_samples:
  train: 100  # 训练集最多100个样本
  test: 50    # 测试集最多50个样本
```

### sample_shuffle

控制采样时是否随机打乱，默认为 `true`（训练集）或 `false`（测试集/验证集）：

```yaml
sample_shuffle:
  train: true   # 训练集随机采样
  test: false   # 测试集顺序采样
```

## 优先级规则

1. **sample_ratio 和 max_samples**: 如果同时指定，`sample_ratio` 优先
2. **配置来源**: 按以下顺序查找配置：
   - `dataset.sample_ratio` / `dataset.max_samples`
   - `dataset.sampling.sample_ratio` / `dataset.sampling.max_samples`
   - `dataset.config.sample_ratio` / `dataset.config.max_samples`
   - `dataset.config.sampling.sample_ratio` / `dataset.config.sampling.max_samples`

## 使用示例

### 示例1: 快速实验（使用10%数据）

```yaml
dataset:
  name: cifar10
  root_dir: data/cifar10
  config:
    download: true
  sample_ratio:
    train: 0.1
    test: 0.1
```

### 示例2: 调试模式（使用固定数量）

```yaml
dataset:
  name: cifar10
  root_dir: data/cifar10
  config:
    download: true
  max_samples:
    train: 50
    test: 10
```

### 示例3: 混合配置

```yaml
dataset:
  name: cifar10
  root_dir: data/cifar10
  config:
    download: true
  sampling:
    sample_ratio:
      train: 0.2
    max_samples:
      test: 100
```

## 代码使用

### 通过配置自动应用

采样配置会在 `get_dataset()` 时自动应用：

```python
from src.config import set_config
from src.dataset import get_dataset, get_all_dataloader

config = {
    "dataset": {
        "name": "cifar10",
        "root_dir": "data/cifar10",
        "config": {"download": True},
        "sample_ratio": {"train": 0.1, "test": 0.2}
    },
    "train": {"batch_size": 32},
    "test": {"batch_size": 32}
}

set_config(config)

# 采样会自动应用
train_dataset = get_dataset("train")
test_dataset = get_dataset("test")

# 创建DataLoader
train_loader, _, test_loader = get_all_dataloader()
```

### 手动应用采样

也可以手动调用 `mininalize()` 方法：

```python
from src.dataset import get_dataset

dataset = get_dataset("train")

# 使用10%的数据，随机采样
dataset.mininalize(dataset_size=0.1, random_sample=True)

# 或使用固定数量
dataset.mininalize(dataset_size=100, random_sample=True)
```

## 注意事项

1. **数据集兼容性**: 只有继承自 `CustomDataset` 并实现 `mininalize()` 方法的数据集才支持采样。不支持的数据集会显示警告但不影响使用。

2. **采样时机**: 采样在数据集创建后、DataLoader创建前进行。

3. **随机性**: 使用 `sample_shuffle=True` 时，每次运行可能得到不同的样本子集。如需可复现，请设置随机种子。

4. **性能**: 采样不会影响原始数据集，只是创建了一个采样器。对性能影响很小。

5. **验证集**: 验证集的采样配置与训练集和测试集独立。

## 故障排除

### 采样配置未生效

**症状**: 数据集长度没有变化

**可能原因**:
1. 数据集不支持 `mininalize()` 方法
2. 配置格式错误
3. 配置被其他配置覆盖

**解决方案**:
1. 检查日志中的警告信息
2. 验证配置格式是否正确
3. 检查配置优先级

### 采样后数据集为空

**症状**: 采样后数据集长度为0

**可能原因**:
1. `sample_ratio` 或 `max_samples` 配置错误
2. 原始数据集为空

**解决方案**:
1. 检查原始数据集是否正常加载
2. 验证采样配置值是否合理

## 相关文档

- [DataLoader使用文档](dataloader_usage.md)
- [数据集配置指南](dataset_config.md)
- [测试文档](testing.md)

