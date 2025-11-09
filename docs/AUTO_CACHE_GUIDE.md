# 自动缓存功能使用指南

## 🎯 设计理念

缓存功能现在是**完全自动化**的：
- ✅ 第一次加载数据集时，自动创建缓存
- ✅ 之后的加载自动从缓存读取
- ✅ 完全透明，无需任何手动操作
- ✅ 默认启用，开箱即用

## 🚀 使用方法

### 基本使用（推荐）

```python
from pathlib import Path
from src.dataset.mnist_dataset import MNISTDataset

# 第一次运行
dataset = MNISTDataset(
    root_dir=Path("data/mnist"),
    split='train'
)
# ✓ 自动检测缓存不存在
# ✓ 加载数据
# ✓ 自动保存到缓存

# 第二次运行
dataset = MNISTDataset(
    root_dir=Path("data/mnist"),
    split='train'
)
# ✓ 自动检测缓存存在
# ✓ 直接从缓存加载（快速！）
```

**就这么简单！** 无需调用任何缓存方法。

---

## 📖 详细说明

### 工作流程

```
初始化数据集
    │
    ├─→ 检查缓存是否存在
    │       │
    │       ├─ 存在 → 从缓存加载 → 完成 ✓
    │       │
    │       └─ 不存在 ↓
    │
    ├─→ 正常加载数据
    │
    └─→ 自动保存到缓存 → 完成 ✓
```

### 配置选项

#### 1. 默认行为（推荐）

```python
# 缓存默认启用，无需任何配置
dataset = MNISTDataset(
    root_dir=Path("data/mnist"),
    split='train'
)
```

#### 2. 禁用缓存

```python
# 开发调试时可能需要禁用
dataset = MNISTDataset(
    root_dir=Path("data/mnist"),
    split='train',
    enable_cache=False  # 禁用缓存
)
```

#### 3. 强制重建缓存

```python
# 数据更新后需要重建缓存
dataset = MNISTDataset(
    root_dir=Path("data/mnist"),
    split='train',
    force_rebuild_cache=True  # 忽略现有缓存，重新构建
)
```

#### 4. 版本管理

```python
# 实验1
dataset_exp1 = MNISTDataset(
    root_dir=Path("data/mnist"),
    split='train',
    cache_version='exp1'  # 使用独立的版本
)

# 实验2
dataset_exp2 = MNISTDataset(
    root_dir=Path("data/mnist"),
    split='train',
    cache_version='exp2'  # 不同版本互不干扰
)
```

#### 5. 自定义缓存目录

```python
dataset = MNISTDataset(
    root_dir=Path("data/mnist"),
    split='train',
    cache_root=Path("/path/to/custom/cache")
)
```

#### 6. 指定缓存格式

```python
dataset = MNISTDataset(
    root_dir=Path("data/mnist"),
    split='train',
    cache_format='pt'  # 可选: 'pkl', 'pt', 'json'
)
```

---

## 📂 缓存目录结构

```
cache/
└── datasets/                # 数据集缓存根目录
    ├── MNIST/               # 数据集名称（自动创建）
    │   ├── v1/              # 版本号（默认v1）
    │   │   ├── train_xxx.pkl         # 训练集缓存
    │   │   ├── train_xxx.meta.json   # 元数据
    │   │   ├── valid_xxx.pkl
    │   │   ├── valid_xxx.meta.json
    │   │   ├── test_xxx.pkl
    │   │   └── test_xxx.meta.json
    │   └── v2/
    │       └── ...
    └── {其他数据集}/
```

---

## 🎓 适用场景

### ✅ 适合使用缓存

- 需要预处理的大型数据集
- 从网络或慢速存储加载的数据
- 多次训练/实验的场景
- 复杂的数据增强流程

### ❌ 建议禁用缓存

- 开发调试阶段（数据集代码在修改）
- 磁盘空间非常有限
- 数据频繁更新
- 单次训练

---

## 🔧 在自定义数据集中实现

如果你要创建自己的数据集类，按以下模式实现：

```python
from pathlib import Path
from src.dataset.custom_dataset import CustomDataset

class MyDataset(CustomDataset):
    @staticmethod
    def name() -> str:
        return "my_dataset"
    
    def __init__(self, root_dir: Path, split: str, **kwargs):
        # 1. 调用父类构造函数（会自动尝试从缓存加载）
        super().__init__(root_dir, split, **kwargs)
        
        # 2. 如果从缓存加载成功，直接返回
        if self._cache_loaded:
            return
        
        # 3. 缓存不存在，正常加载数据
        self._load_data()
        
        # 4. 自动保存到缓存
        self._save_to_cache_if_needed()
    
    def _load_data(self):
        """实际的数据加载逻辑"""
        # 加载数据...
        self.samples = [...]  # 加载样本
        self.n = len(self.samples)
    
    def __getitem__(self, index):
        """获取数据项"""
        # 从缓存加载的情况
        if self._cache_loaded:
            return self.samples[index]
        # 正常加载的情况
        return self._process_sample(index)
    
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

### 实现要点

1. **调用父类构造函数**: `super().__init__(root_dir, split, **kwargs)` 会自动处理缓存加载
2. **检查 `_cache_loaded`**: 如果为True，说明数据已从缓存加载，直接返回
3. **加载数据**: 正常加载数据到 `self.samples`
4. **调用 `_save_to_cache_if_needed()`**: 自动保存到缓存

---

## 🛠️ 命令行工具

虽然缓存是自动的，但你仍可以使用命令行工具管理：

```bash
# 查看所有缓存
python tools/dataset_cache_tool.py list

# 查看特定数据集的缓存
python tools/dataset_cache_tool.py info MNIST

# 清除缓存
python tools/dataset_cache_tool.py clear MNIST --split train

# 清除所有缓存
python tools/dataset_cache_tool.py clear-all

# 验证缓存完整性
python tools/dataset_cache_tool.py verify MNIST
```

---

## 📊 性能提升

典型的性能提升：

| 场景 | 时间 | 说明 |
|------|------|------|
| 第1次加载（创建缓存） | 2.5s | 正常加载 + 保存缓存 |
| 第2次加载（从缓存）   | 0.3s | 直接从缓存读取 |
| **加速比** | **8.3x** | 显著提升 |

---

## ⚠️ 注意事项

### 1. 数据更新

当原始数据更新时，记得重建缓存：

```python
dataset = MyDataset(
    root_dir=path,
    split='train',
    force_rebuild_cache=True  # 强制重建
)
```

或使用命令行清除缓存：

```bash
python tools/dataset_cache_tool.py clear my_dataset
```

### 2. 磁盘空间

缓存会占用额外的磁盘空间，定期检查和清理：

```bash
# 查看缓存占用
python tools/dataset_cache_tool.py list

# 清除不需要的缓存
python tools/dataset_cache_tool.py clear old_dataset
```

### 3. 多进程训练

- ✅ 先在单进程中加载一次数据集（创建缓存）
- ✅ 然后在多进程训练中使用（从缓存加载）
- ❌ 避免多个进程同时创建同一个缓存

### 4. 配置一致性

确保加载时的配置与保存时一致，否则会创建新的缓存：

```python
# 第一次
dataset1 = MyDataset(root_dir=path, split='train', some_param=10)

# 第二次 - 会使用相同的缓存
dataset2 = MyDataset(root_dir=path, split='train', some_param=10)

# 第三次 - 配置不同，会创建新的缓存
dataset3 = MyDataset(root_dir=path, split='train', some_param=20)
```

---

## 🎯 最佳实践

### 开发阶段

```python
# 禁用缓存，便于快速迭代
dataset = MyDataset(
    root_dir=path,
    split='train',
    enable_cache=False
)
```

### 训练阶段

```python
# 使用默认配置，自动缓存
dataset = MyDataset(
    root_dir=path,
    split='train'
)
```

### 实验管理

```python
# 为不同实验使用不同版本
dataset_baseline = MyDataset(
    root_dir=path,
    split='train',
    cache_version='baseline'
)

dataset_improved = MyDataset(
    root_dir=path,
    split='train',
    cache_version='improved'
)
```

---

## 📚 示例程序

运行自动缓存演示：

```bash
conda run -n ntrain python examples/auto_cache_demo.py
```

该演示包含：
- ✅ 自动缓存基本用法
- ✅ 性能对比
- ✅ 禁用缓存
- ✅ 强制重建
- ✅ 多划分缓存
- ✅ 版本管理

---

## 🆚 与旧版本的对比

### 旧方式（手动）

```python
dataset = MyDataset(root_dir=path, split='train', enable_cache=True)

# 需要手动调用
if not dataset.load_from_cache():
    dataset.save_to_cache()
```

### 新方式（自动）✨

```python
# 完全自动，无需任何操作
dataset = MyDataset(root_dir=path, split='train')
```

---

## 🎉 总结

### 核心优势

1. **完全自动** - 无需任何手动操作
2. **默认启用** - 开箱即用
3. **完全透明** - 对使用者透明
4. **性能显著** - 2-10倍加速
5. **易于集成** - 继承CustomDataset即可获得

### 使用建议

- ✅ 训练时使用默认配置（自动缓存）
- ✅ 开发时可以禁用缓存
- ✅ 数据更新后强制重建
- ✅ 使用版本号管理不同实验
- ✅ 定期清理不需要的缓存

---

**版本**: 2.0.0 (自动缓存)  
**更新日期**: 2025-10-29

