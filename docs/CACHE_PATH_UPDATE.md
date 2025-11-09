# 缓存路径更新说明

## 📂 路径变更

缓存目录路径已更新为更合理的结构：

### 旧路径
```
cache/
├── {dataset_name}/
│   └── {version}/
```

### 新路径 ✨
```
cache/
└── datasets/              # 新增datasets子目录
    ├── {dataset_name}/
    │   └── {version}/
```

---

## 🎯 为什么要更改？

1. **更好的组织**: `cache/datasets/` 专门用于数据集缓存，其他类型的缓存可以放在cache的其他子目录
2. **避免冲突**: cache目录下可能还有 `models/`（预训练模型）等其他内容
3. **清晰的结构**: 一看就知道 `datasets/` 下都是数据集缓存

---

## 📊 完整的目录结构

```
cache/
├── models/              # 预训练模型缓存（已存在）
│   └── pretrained/
└── datasets/            # 数据集缓存（新增）
    ├── MNIST/
    │   ├── v1/
    │   │   ├── train_xxx.pkl
    │   │   ├── train_xxx.meta.json
    │   │   ├── valid_xxx.pkl
    │   │   ├── valid_xxx.meta.json
    │   │   ├── test_xxx.pkl
    │   │   └── test_xxx.meta.json
    │   └── v2/
    ├── CIFAR10/
    │   └── v1/
    └── {其他数据集}/
```

---

## 🔧 已更新的文件

### 核心代码
1. ✅ `src/dataset/cache_manager.py`
   - 修改缓存目录为 `cache_root / "datasets" / dataset_name / version`

### 工具脚本
2. ✅ `tools/dataset_cache_tool.py`
   - 更新 `list_all_caches()` 查找 `cache/datasets/`
   - 更新 `clear_all_caches()` 清除 `cache/datasets/`

### 文档
3. ✅ `docs/AUTO_CACHE_GUIDE.md`
4. ✅ `docs/dataset_cache.md`
5. ✅ `docs/cache_feature_summary.md`
6. ✅ `docs/IMPLEMENTATION_SUMMARY.md`

---

## 💻 使用方式

### 无需任何改变！

用户代码**完全不需要修改**，缓存会自动保存到新路径：

```python
# 代码不变
dataset = MNISTDataset(
    root_dir=Path("data/mnist"),
    split='train'
)

# 缓存自动保存到: cache/datasets/MNIST/v1/train_xxx.pkl
```

### 命令行工具

命令行工具会自动在正确的路径下查找：

```bash
# 查看缓存（自动在 cache/datasets/ 下查找）
python tools/dataset_cache_tool.py list

# 查看特定数据集
python tools/dataset_cache_tool.py info MNIST

# 清除缓存
python tools/dataset_cache_tool.py clear MNIST
```

---

## 🔄 迁移旧缓存（如果有）

如果你之前已经有缓存文件在旧路径 `cache/{dataset_name}/`，可以手动迁移：

```bash
# 方案1: 移动到新路径
mkdir -p cache/datasets
mv cache/MNIST cache/datasets/
mv cache/CIFAR10 cache/datasets/
# ... 移动其他数据集

# 方案2: 清除旧缓存，自动重建
rm -rf cache/MNIST cache/CIFAR10
# 下次运行时会自动在新路径重建缓存
```

或者使用命令行工具清除所有缓存：

```bash
python tools/dataset_cache_tool.py clear-all
```

---

## ✅ 测试验证

```bash
# 运行测试验证新路径
conda run -n ntrain python tests/test_dataset_cache.py
```

所有测试应该通过 ✓

---

## 📝 示例

### 自动缓存位置

```python
from pathlib import Path
from src.dataset.mnist_dataset import MNISTDataset

# 创建数据集
dataset = MNISTDataset(
    root_dir=Path("data/mnist"),
    split='train',
    cache_version='v1'
)

# 缓存会自动保存到:
# cache/datasets/MNIST/v1/train_xxx.pkl
# cache/datasets/MNIST/v1/train_xxx.meta.json
```

### 自定义缓存根目录

```python
# 如果指定自定义缓存目录
dataset = MNISTDataset(
    root_dir=Path("data/mnist"),
    split='train',
    cache_root=Path("/custom/cache")
)

# 缓存会保存到:
# /custom/cache/datasets/MNIST/v1/train_xxx.pkl
```

---

## 🎊 总结

- ✅ 缓存路径更新为 `cache/datasets/{dataset_name}/{version}`
- ✅ 更好的目录结构和组织
- ✅ 用户代码无需修改
- ✅ 所有功能正常工作
- ✅ 向后兼容（只是路径变了）

---

**更新日期**: 2025-10-29  
**版本**: V2.0.1

