# 快速参考指南

## 🚀 常用命令

### 测试

```bash
# 快速测试（约30秒）
conda activate ntrain
python scripts/run_quick_test.py

# 完整测试（约2-5分钟）
python scripts/run_tests.py
```

### 数据集采样配置

#### 方式1: 直接配置
```yaml
dataset:
  sample_ratio:
    train: 0.1
    test: 0.2
```

#### 方式2: 嵌套配置
```yaml
dataset:
  sampling:
    sample_ratio:
      train: 0.05
    max_samples:
      test: 20
```

### DataLoader创建

```python
from src.config import set_config
from src.dataset import get_all_dataloader

set_config(config)
train_loader, valid_loader, test_loader = get_all_dataloader(use_valid=True)
```

## 📋 配置模板

### 基础配置
```yaml
dataset:
  name: cifar10
  root_dir: data/cifar10
  config:
    download: true
    valid_ratio: 0.1

train:
  batch_size: 32

test:
  batch_size: 32

dataloader:
  num_workers: 4
  shuffle: true
  pin_memory: true
```

### 带采样配置
```yaml
dataset:
  name: cifar10
  root_dir: data/cifar10
  config:
    download: true
  sample_ratio:
    train: 0.1
    test: 0.2
  max_samples:
    train: 100
    test: 50
```

## 🔧 常用代码片段

### 获取数据集
```python
from src.dataset import get_dataset

train_dataset = get_dataset("train")
test_dataset = get_dataset("test")
valid_dataset = get_dataset("valid")
```

### 创建DataLoader
```python
from src.dataset import get_all_dataloader

train_loader, valid_loader, test_loader = get_all_dataloader(use_valid=True)
```

### 手动采样
```python
dataset = get_dataset("train")
dataset.mininalize(dataset_size=0.1, random_sample=True)
```

### 直接使用dataloader方法
```python
dataset = get_dataset("train")
loader = dataset.dataloader(
    batch_size=32,
    shuffle=True,
    num_workers=4
)
```

## 📚 文档链接

- [数据集采样配置](dataset_sampling.md)
- [DataLoader使用指南](dataloader_usage.md)
- [测试文档](testing.md)

## ⚠️ 常见问题

### Q: 采样配置未生效？
A: 检查数据集是否支持 `mininalize()` 方法，查看日志中的警告信息。

### Q: DataLoader创建失败？
A: 检查配置格式是否正确，确认数据集已成功加载。

### Q: DDP/DeepSpeed测试失败？
A: 单GPU环境下会显示警告，但不影响其他功能。多GPU环境下需要正确配置。

## 🔗 相关资源

- 项目主README: [../README.md](../README.md)
- 脚本目录: [../scripts/](../scripts/)

