# 数据集缓存功能 - 功能总结

## 新增文件

本次更新添加了数据集缓存功能，以下是新增和修改的文件：

### 核心模块

1. **`src/dataset/cache_manager.py`** (新增)
   - 数据集缓存管理器核心实现
   - 支持多种缓存格式（pickle, torch, json）
   - 提供缓存的创建、加载、验证、清除等功能
   - 自动管理缓存元数据和版本

2. **`src/dataset/custom_dataset.py`** (修改)
   - 在CustomDataset基类中集成缓存功能
   - 添加缓存相关的初始化参数
   - 添加 `save_to_cache()`, `load_from_cache()`, `clear_cache()` 方法
   - 所有继承CustomDataset的数据集类自动获得缓存能力

### 工具和示例

3. **`tools/dataset_cache_tool.py`** (新增)
   - 命令行缓存管理工具
   - 支持查看、清除、验证缓存
   - 提供便捷的缓存信息查询

4. **`examples/dataset_cache_example.py`** (新增)
   - 完整的缓存使用示例
   - 涵盖各种使用场景
   - 包括6个具体示例

5. **`examples/mnist_with_cache_demo.py`** (新增)
   - MNIST数据集的缓存演示
   - 性能对比测试
   - 实际应用示例

### 文档

6. **`docs/dataset_cache.md`** (新增)
   - 完整的缓存功能文档
   - API参考
   - 最佳实践和故障排查

7. **`docs/cache_feature_summary.md`** (本文件, 新增)
   - 功能总结和快速开始指南

## 功能特性

### 主要功能

✅ **多格式支持**
- Pickle格式（.pkl）: 适合Python对象
- PyTorch格式（.pt）: 适合tensor数据
- JSON格式（.json）: 适合简单数据，可读性好

✅ **智能缓存管理**
- 基于配置自动生成缓存键
- 支持多版本并存
- 自动元数据管理
- 缓存完整性验证

✅ **易于集成**
- 无缝集成到CustomDataset
- 向后兼容，不影响现有代码
- 简单的API设计

✅ **灵活配置**
- 可选的缓存启用/禁用
- 自定义缓存目录
- 版本号管理
- 强制重建选项

✅ **命令行工具**
- 查看所有缓存信息
- 管理和清理缓存
- 验证缓存完整性

## 目录结构

```
cache/                          # 缓存根目录
└── datasets/                   # 数据集缓存目录
    ├── {dataset_name}/         # 数据集名称目录
    │   ├── v1/                 # 版本目录
    │   │   ├── train_xxx.pkl       # 训练集缓存
    │   │   ├── train_xxx.meta.json # 训练集元数据
    │   │   ├── valid_xxx.pkl       # 验证集缓存
    │   │   ├── valid_xxx.meta.json # 验证集元数据
    │   │   ├── test_xxx.pkl        # 测试集缓存
    │   │   └── test_xxx.meta.json  # 测试集元数据
    │   └── v2/                 # 其他版本
    │       └── ...
    └── {other_dataset}/
        └── ...
```

## 快速开始

### 1. 基本使用

```python
from pathlib import Path
from src.dataset.mnist_dataset import MNISTDataset

# 创建数据集时启用缓存
dataset = MNISTDataset(
    root_dir=Path("data/mnist"),
    split='train',
    enable_cache=True,      # 启用缓存
    cache_version='v1'       # 指定版本
)

# 尝试从缓存加载
if not dataset.load_from_cache():
    # 缓存不存在，正常加载后保存
    dataset.save_to_cache()
```

### 2. 在配置文件中启用

```yaml
dataset:
  name: mnist
  root_dir: data/mnist
  config:
    enable_cache: true
    cache_version: v1
```

### 3. 使用命令行工具

```bash
# 查看所有缓存
python tools/dataset_cache_tool.py list

# 查看特定数据集
python tools/dataset_cache_tool.py info mnist

# 清除缓存
python tools/dataset_cache_tool.py clear mnist --split train

# 验证缓存
python tools/dataset_cache_tool.py verify mnist
```

## 使用示例

### 示例1: 简单的缓存使用

```python
from pathlib import Path
from src.dataset.your_dataset import YourDataset

def load_with_cache(root_dir, split):
    dataset = YourDataset(
        root_dir=root_dir,
        split=split,
        enable_cache=True
    )
    
    if not dataset.load_from_cache():
        # 正常加载数据
        print("Loading from source...")
        # 数据集的初始化会自动进行
        
        # 保存到缓存
        dataset.save_to_cache()
    
    return dataset

train_ds = load_with_cache(Path("data/my_dataset"), 'train')
```

### 示例2: 使用缓存管理器

```python
from src.dataset.cache_manager import DatasetCacheManager

# 创建管理器
cache_mgr = DatasetCacheManager(
    dataset_name='mnist',
    version='v1',
    enable_cache=True
)

# 查看缓存信息
info = cache_mgr.get_cache_info()
print(f"文件数: {info['total_files']}")
print(f"大小: {info['total_size_mb']:.2f} MB")

# 清除特定缓存
cache_mgr.clear(split='train')
```

### 示例3: 版本管理

```python
# 创建不同版本的缓存
dataset_v1 = MNISTDataset(
    root_dir=Path("data/mnist"),
    split='train',
    enable_cache=True,
    cache_version='v1'  # 原始版本
)
dataset_v1.save_to_cache()

dataset_v2 = MNISTDataset(
    root_dir=Path("data/mnist"),
    split='train',
    enable_cache=True,
    cache_version='v2',  # 新版本
    # 额外的预处理参数...
)
dataset_v2.save_to_cache()
```

## 在自定义数据集中集成

如果你有自定义的数据集类，可以很容易地添加缓存支持：

```python
from pathlib import Path
from src.dataset.custom_dataset import CustomDataset

class MyDataset(CustomDataset):
    @staticmethod
    def name() -> str:
        return "my_dataset"
    
    def __init__(self, root_dir: Path, split: str, **kwargs):
        # 调用父类构造函数（会处理缓存参数）
        super().__init__(root_dir, split, **kwargs)
        
        # 尝试从缓存加载
        if self.enable_cache and self.load_from_cache():
            return  # 成功从缓存加载
        
        # 正常加载数据
        self._load_data()
        
        # 保存到缓存
        if self.enable_cache:
            self.save_to_cache()
    
    def _load_data(self):
        """实际的数据加载逻辑"""
        self.samples = [...]  # 加载数据
        self.n = len(self.samples)
    
    def __getitem__(self, index):
        return self.samples[index]
    
    # 实现必需的静态方法
    @staticmethod
    def get_train_dataset(root_dir: Path, **kwargs):
        return MyDataset(root_dir, 'train', **kwargs)
    
    @staticmethod
    def get_valid_dataset(root_dir: Path, **kwargs):
        return MyDataset(root_dir, 'valid', **kwargs)
    
    @staticmethod
    def get_test_dataset(root_dir: Path, **kwargs):
        return MyDataset(root_dir, 'test', **kwargs)
```

## API参考

### CustomDataset缓存相关参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_cache` | bool | False | 是否启用缓存 |
| `cache_root` | Path | None | 缓存根目录，默认为./cache |
| `cache_version` | str | 'v1' | 缓存版本号 |
| `force_rebuild_cache` | bool | False | 是否强制重建缓存 |

### CustomDataset缓存方法

| 方法 | 说明 |
|------|------|
| `save_to_cache(format='pkl', metadata=None)` | 保存到缓存 |
| `load_from_cache(format='pkl')` | 从缓存加载 |
| `clear_cache()` | 清除缓存 |

### DatasetCacheManager方法

| 方法 | 说明 |
|------|------|
| `exists(split, config, format)` | 检查缓存是否存在 |
| `save(data, split, config, format, metadata)` | 保存数据到缓存 |
| `load(split, config, format, check_validity)` | 从缓存加载数据 |
| `clear(split, config, format)` | 清除缓存 |
| `get_cache_info(split)` | 获取缓存信息 |

## 性能提升

使用缓存可以显著提高数据加载速度，特别是对于：

- 需要大量预处理的数据集
- 从网络或慢速存储加载的数据
- 复杂的数据增强流程
- 多次训练/实验的场景

典型的性能提升：
- **首次加载**: 可能略慢（需要保存缓存）
- **后续加载**: 通常可以提速 **2-10倍**，取决于数据集复杂度

## 最佳实践

1. **开发阶段**: 禁用缓存或使用 `force_rebuild_cache=True`
2. **训练阶段**: 启用缓存以加速数据加载
3. **版本管理**: 为不同的预处理配置使用不同版本号
4. **定期清理**: 使用命令行工具定期清理不需要的缓存

## 注意事项

⚠️ **数据更新**: 原始数据更新时记得清除或重建缓存  
⚠️ **磁盘空间**: 缓存会占用额外的磁盘空间  
⚠️ **线程安全**: 避免多进程同时写入相同的缓存  
⚠️ **配置一致**: 确保加载时的配置与保存时一致

## 故障排查

### 问题1: 缓存加载失败
```bash
# 验证缓存
python tools/dataset_cache_tool.py verify dataset_name

# 清除损坏的缓存
python tools/dataset_cache_tool.py clear dataset_name
```

### 问题2: 缓存占用过多空间
```bash
# 查看所有缓存
python tools/dataset_cache_tool.py list

# 清除不需要的缓存
python tools/dataset_cache_tool.py clear old_dataset
```

### 问题3: 数据不是最新的
```python
# 强制重建缓存
dataset = MyDataset(
    root_dir=path,
    split='train',
    enable_cache=True,
    force_rebuild_cache=True  # 忽略现有缓存
)
```

## 运行演示

```bash
# 运行基本示例
cd /home/rczx/workspace/sxy/lab/NeuroTrain
python examples/dataset_cache_example.py

# 运行MNIST演示（包含性能对比）
python examples/mnist_with_cache_demo.py

# 使用命令行工具
python tools/dataset_cache_tool.py list
```

## 更多信息

详细文档请参考：
- [`docs/dataset_cache.md`](dataset_cache.md) - 完整的功能文档
- [`examples/dataset_cache_example.py`](../examples/dataset_cache_example.py) - 使用示例
- [`examples/mnist_with_cache_demo.py`](../examples/mnist_with_cache_demo.py) - MNIST演示

## 贡献

如有问题或建议，请提交Issue或Pull Request。

---

**版本**: 1.0.0  
**更新日期**: 2025-10-29

