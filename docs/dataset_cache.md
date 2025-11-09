# 数据集缓存功能文档

## 概述

数据集缓存功能允许你将预处理后的数据集保存到磁盘，避免每次训练时都重新加载和预处理数据，从而显著提高数据加载速度。

缓存文件统一存储在 `cache/datasets/{dataset_name}/{version}` 目录下，支持多种格式和版本管理。

## 主要特性

- **多种缓存格式**: 支持 pickle (`.pkl`)、PyTorch (`.pt`)、JSON (`.json`) 格式
- **自动缓存键生成**: 基于数据集配置自动生成唯一的缓存键
- **版本管理**: 支持多个缓存版本，便于实验管理
- **元数据管理**: 自动保存缓存元信息，包括创建时间、样本数等
- **完整性检查**: 自动验证缓存文件的有效性
- **易于集成**: 无缝集成到现有的 `CustomDataset` 系统

## 快速开始

### 1. 基本使用

在创建数据集时启用缓存：

```python
from pathlib import Path
from src.dataset.mnist_dataset import MNISTDataset

# 创建数据集并启用缓存
dataset = MNISTDataset(
    root_dir=Path("data/mnist"),
    split='train',
    enable_cache=True,      # 启用缓存
    cache_version='v1'       # 缓存版本
)

# 正常加载数据...
# (数据集的初始化和加载逻辑)

# 保存到缓存
dataset.save_to_cache(format='pkl')
```

### 2. 从缓存加载

```python
# 创建数据集
dataset = MNISTDataset(
    root_dir=Path("data/mnist"),
    split='train',
    enable_cache=True
)

# 尝试从缓存加载
if dataset.load_from_cache(format='pkl'):
    print("成功从缓存加载！")
else:
    print("缓存不存在，正常加载数据集")
    # 正常加载数据...
    # 然后保存到缓存
    dataset.save_to_cache(format='pkl')
```

### 3. 典型的工作流程

```python
from pathlib import Path
from src.dataset.your_dataset import YourDataset

def load_dataset_with_cache(root_dir, split):
    """带缓存的数据集加载函数"""
    dataset = YourDataset(
        root_dir=root_dir,
        split=split,
        enable_cache=True,
        cache_version='v1'
    )
    
    # 尝试从缓存加载
    if not dataset.load_from_cache(format='pkl'):
        # 缓存不存在，正常加载数据集
        print(f"Loading {split} dataset from scratch...")
        # 这里会执行数据集的 __init__ 中的加载逻辑
        
        # 保存到缓存
        print(f"Saving to cache...")
        dataset.save_to_cache(format='pkl')
    
    return dataset

# 使用
train_dataset = load_dataset_with_cache(Path("data/my_dataset"), 'train')
```

## 使用缓存管理器

### 创建缓存管理器

```python
from src.dataset.cache_manager import DatasetCacheManager

cache_manager = DatasetCacheManager(
    dataset_name='mnist',
    version='v1',
    enable_cache=True
)
```

### 查看缓存信息

```python
# 获取缓存信息
info = cache_manager.get_cache_info()

print(f"数据集: {info['dataset_name']}")
print(f"缓存目录: {info['cache_dir']}")
print(f"文件数: {info.get('total_files', 0)}")
print(f"总大小: {info.get('total_size_mb', 0):.2f} MB")

# 查看所有缓存文件
for file_info in info.get('files', []):
    print(f"文件: {file_info['filename']}")
    print(f"  大小: {file_info['size'] / 1024 / 1024:.2f} MB")
    meta = file_info.get('metadata', {})
    print(f"  划分: {meta.get('split', 'unknown')}")
    print(f"  样本数: {meta.get('num_samples', 'unknown')}")
```

### 手动保存和加载

```python
# 保存数据
data = {"samples": [...], "labels": [...]}
cache_manager.save(
    data,
    split='train',
    config={'root_dir': 'data/mnist'},
    format='pkl',
    metadata={'num_samples': len(data['samples'])}
)

# 加载数据
cached_data = cache_manager.load(
    split='train',
    config={'root_dir': 'data/mnist'},
    format='pkl'
)
```

### 清除缓存

```python
# 清除特定划分的缓存
cache_manager.clear(split='train')

# 清除所有缓存
cache_manager.clear()
```

## 命令行工具

项目提供了 `dataset_cache_tool.py` 命令行工具来管理缓存。

### 查看所有缓存

```bash
python tools/dataset_cache_tool.py list
```

### 查看特定数据集的缓存信息

```bash
python tools/dataset_cache_tool.py info mnist
python tools/dataset_cache_tool.py info mnist --version v2
```

### 清除缓存

```bash
# 清除特定数据集的所有缓存
python tools/dataset_cache_tool.py clear mnist

# 清除特定划分的缓存
python tools/dataset_cache_tool.py clear mnist --split train

# 清除所有缓存
python tools/dataset_cache_tool.py clear-all
```

### 验证缓存完整性

```bash
python tools/dataset_cache_tool.py verify mnist
```

## 高级用法

### 1. 使用不同的缓存格式

```python
# Pickle格式 (默认，适合Python对象)
dataset.save_to_cache(format='pkl')
dataset.load_from_cache(format='pkl')

# PyTorch格式 (适合tensor数据)
dataset.save_to_cache(format='pt')
dataset.load_from_cache(format='pt')

# JSON格式 (适合简单数据结构，可读性好)
dataset.save_to_cache(format='json')
dataset.load_from_cache(format='json')
```

### 2. 强制重建缓存

有时你可能需要忽略现有缓存并重新构建：

```python
dataset = MNISTDataset(
    root_dir=Path("data/mnist"),
    split='train',
    enable_cache=True,
    force_rebuild_cache=True  # 忽略现有缓存
)

# 重新加载数据...
# 保存新的缓存
dataset.save_to_cache(format='pkl')
```

### 3. 自定义缓存目录

```python
dataset = MNISTDataset(
    root_dir=Path("data/mnist"),
    split='train',
    enable_cache=True,
    cache_root=Path("/path/to/custom/cache")  # 自定义缓存目录
)
```

### 4. 版本管理

使用不同的版本号来管理不同配置的缓存：

```python
# 版本1: 原始数据
dataset_v1 = MNISTDataset(
    root_dir=Path("data/mnist"),
    split='train',
    enable_cache=True,
    cache_version='v1'
)
dataset_v1.save_to_cache()

# 版本2: 带数据增强
dataset_v2 = MNISTDataset(
    root_dir=Path("data/mnist"),
    split='train',
    enable_cache=True,
    cache_version='v2',
    augmentation=True  # 额外的参数
)
dataset_v2.save_to_cache()
```

### 5. 使用缓存包装器

`DatasetCacheManager.cache_dataset` 提供了一个通用的缓存包装器：

```python
from src.dataset.cache_manager import DatasetCacheManager
from src.dataset.mnist_dataset import MNISTDataset

cache_manager = DatasetCacheManager(
    dataset_name='mnist',
    version='v1',
    enable_cache=True
)

# 使用缓存包装器
dataset = DatasetCacheManager.cache_dataset(
    dataset_cls=MNISTDataset,
    cache_manager=cache_manager,
    split='train',
    root_dir=Path('data/mnist'),
    format='pkl',
    force_rebuild=False
)
```

## 在自定义数据集中实现缓存

如果你想在自己的数据集类中使用缓存功能：

```python
from pathlib import Path
from src.dataset.custom_dataset import CustomDataset

class MyDataset(CustomDataset):
    @staticmethod
    def name() -> str:
        return "my_dataset"
    
    def __init__(self, root_dir: Path, split: str, **kwargs):
        # 调用父类构造函数（会处理缓存相关参数）
        super().__init__(root_dir, split, **kwargs)
        
        # 尝试从缓存加载
        if self.enable_cache and self.load_from_cache(format='pkl'):
            # 成功从缓存加载，直接返回
            return
        
        # 缓存不存在，正常加载数据
        self._load_data()
        
        # 保存到缓存
        if self.enable_cache:
            self.save_to_cache(format='pkl')
    
    def _load_data(self):
        """实际的数据加载逻辑"""
        # 加载数据...
        self.samples = [...]  # 加载样本
        self.n = len(self.samples)
    
    def __getitem__(self, index):
        # 实现数据访问逻辑
        return self.samples[index]
    
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

## 配置示例

在配置文件中启用缓存：

```yaml
dataset:
  name: mnist
  root_dir: data/mnist
  config:
    enable_cache: true
    cache_version: v1
    force_rebuild_cache: false
```

## 最佳实践

1. **开发阶段**: 
   - 建议禁用缓存或使用 `force_rebuild_cache=True`
   - 便于快速迭代和调试

2. **训练阶段**:
   - 启用缓存以加速数据加载
   - 使用稳定的 `cache_version`

3. **实验管理**:
   - 为不同的数据预处理配置使用不同的版本号
   - 定期清理不用的缓存

4. **缓存格式选择**:
   - **pickle**: 适合复杂的Python对象，通用性最好
   - **torch**: 适合主要包含tensor的数据
   - **json**: 适合简单数据结构，可读性好

5. **磁盘空间管理**:
   - 定期使用 `dataset_cache_tool.py list` 查看缓存占用
   - 清理不需要的旧版本缓存

## 注意事项

1. **缓存键生成**: 
   - 缓存键基于数据集名称、划分、版本和配置参数
   - 相同配置会使用相同的缓存

2. **数据一致性**:
   - 当原始数据更新时，记得清除或重建缓存
   - 使用 `force_rebuild_cache=True` 强制重建

3. **内存使用**:
   - 大型数据集的缓存文件可能很大
   - 注意磁盘空间

4. **线程安全**:
   - 当前实现不保证多进程同时写入的安全性
   - 建议先构建好缓存再进行多进程训练

## 故障排查

### 缓存加载失败

```python
# 检查缓存是否存在
if cache_manager.exists(split='train'):
    print("缓存存在")
else:
    print("缓存不存在")

# 验证缓存有效性
info = cache_manager.get_cache_info()
# 查看详细信息...
```

### 清除损坏的缓存

```bash
# 验证缓存
python tools/dataset_cache_tool.py verify mnist

# 清除损坏的缓存
python tools/dataset_cache_tool.py clear mnist

# 或在代码中
dataset.clear_cache()
```

### 缓存占用过多磁盘空间

```bash
# 查看所有缓存占用
python tools/dataset_cache_tool.py list

# 清除不需要的缓存
python tools/dataset_cache_tool.py clear old_dataset
```

## 示例代码

完整的使用示例请参考：
- `examples/dataset_cache_example.py`: 各种使用场景的示例
- `tools/dataset_cache_tool.py`: 命令行工具

## API参考

### CustomDataset缓存相关参数

- `enable_cache` (bool): 是否启用缓存，默认False
- `cache_root` (Path): 缓存根目录，默认为 `./cache`
- `cache_version` (str): 缓存版本号，默认'v1'
- `force_rebuild_cache` (bool): 是否强制重建缓存，默认False

### CustomDataset缓存相关方法

- `save_to_cache(format='pkl', metadata=None)`: 保存到缓存
- `load_from_cache(format='pkl')`: 从缓存加载
- `clear_cache()`: 清除缓存

### DatasetCacheManager方法

- `exists(split, config=None, format='pkl')`: 检查缓存是否存在
- `save(data, split, config=None, format='pkl', metadata=None)`: 保存数据
- `load(split, config=None, format='pkl', check_validity=True)`: 加载数据
- `clear(split=None, config=None, format=None)`: 清除缓存
- `get_cache_info(split=None)`: 获取缓存信息

## 相关文档

- [CustomDataset基类文档](custom_dataset.md)
- [数据集开发指南](dataset_development.md)

