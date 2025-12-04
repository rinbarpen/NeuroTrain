# 脚本目录

## 📜 脚本列表

### 训练脚本

- **`train.sh`** - 训练脚本（支持普通训练、DDP多卡训练、DeepSpeed训练）
  ```bash
  # 单GPU训练
  bash scripts/train.sh -c configs/single/train.toml --train

  # DDP多卡训练 (4卡)
  bash scripts/train.sh -c configs/ddp_example.toml -t ddp -g 4

  # DeepSpeed训练 (2卡)
  bash scripts/train.sh -c configs/deepspeed_example.yaml -t deepspeed -g 2
  ```

### 分析器脚本

- **`analyze.sh`** - 分析器脚本（支持各种analyzer）
  ```bash
  # 运行指标分析器
  bash scripts/analyze.sh metrics --run_id experiment_001

  # 运行数据集分析器
  bash scripts/analyze.sh dataset --root_dir data/cifar10

  # 运行注意力分析器
  bash scripts/analyze.sh attention --model_path runs/model.pth
  ```

### 统一运行脚本

- **`run_all.sh`** - 统一运行脚本（支持训练、测试、预测、分析）
  ```bash
  # 训练
  bash scripts/run_all.sh train -c configs/single/train.toml

  # 测试
  bash scripts/run_all.sh test -c configs/single/train.toml

  # 分析
  bash scripts/run_all.sh analyze metrics --run_id experiment_001
  ```

### 测试脚本

- **`run_quick_test.py`** - 快速测试脚本
- **`run_tests.py`** - 完整测试套件

## 🚀 使用方法

### 训练

#### 单GPU训练
```bash
bash scripts/train.sh -c configs/single/train.toml --train
```

#### DDP多卡训练
```bash
# 4卡训练
bash scripts/train.sh -c configs/ddp_example.toml -t ddp -g 4

# 8卡训练
bash scripts/train.sh -c configs/ddp_example.toml -t ddp -g 8
```

#### DeepSpeed训练
```bash
# 2卡训练
bash scripts/train.sh -c configs/deepspeed_example.yaml -t deepspeed -g 2

# 4卡训练
bash scripts/train.sh -c configs/deepspeed_example.yaml -t deepspeed -g 4
```

### 分析器

#### 指标分析器
```bash
bash scripts/analyze.sh metrics --run_id experiment_001
```

#### 数据集分析器
```bash
bash scripts/analyze.sh dataset --root_dir data/cifar10
```

#### 注意力分析器
```bash
bash scripts/analyze.sh attention --model_path runs/model.pth
```

#### 掩码分析器
```bash
bash scripts/analyze.sh mask --input_dir data/images --output_dir outputs/masks
```

#### 关系分析器
```bash
bash scripts/analyze.sh relation --config configs/relation.yaml
```

#### LoRA分析器
```bash
bash scripts/analyze.sh lora --model_path runs/model.pth --lora_path runs/lora.pt
```

### 统一运行

```bash
# 训练
bash scripts/run_all.sh train -c configs/single/train.toml

# 测试
bash scripts/run_all.sh test -c configs/single/train.toml

# 预测
bash scripts/run_all.sh predict -c configs/single/train.toml

# 分析
bash scripts/run_all.sh analyze metrics --run_id experiment_001

# 快速测试
bash scripts/run_all.sh quick-test
```

## 📋 参数说明

### train.sh

- `-c, --config FILE`: 配置文件路径（必需）
- `-m, --mode MODE`: 运行模式（train/test/predict，默认：train）
- `-g, --gpus N`: 使用的GPU数量（默认：1）
- `-t, --train-mode MODE`: 训练模式（single/ddp/deepspeed，默认：single）
- `-e, --env ENV`: Conda环境名称（默认：ntrain）
- `-d, --device DEVICE`: 设备（默认：cuda:0）

### analyze.sh

- `<analyzer_name>`: 分析器名称（必需）
  - `metrics`: 指标分析器
  - `dataset`: 数据集分析器
  - `attention`: 注意力分析器
  - `mask`: 掩码分析器
  - `relation`: 关系分析器
  - `lora`: LoRA分析器
- `-e, --env ENV`: Conda环境名称（默认：ntrain）

### run_all.sh

- `<action>`: 操作类型（必需）
  - `train`: 训练
  - `test`: 测试
  - `predict`: 预测
  - `analyze`: 分析
  - `quick-test`: 快速测试

## ⚠️ 注意事项

1. **环境要求**: 所有脚本需要在 `ntrain` conda环境中运行
2. **配置文件**: 确保配置文件路径正确
3. **GPU数量**: DDP和DeepSpeed模式需要至少2个GPU
4. **DeepSpeed**: DeepSpeed模式需要安装 `pip install deepspeed`

## 📚 相关文档

- [训练文档](../docs/)
- [数据集采样配置](../docs/dataset_sampling.md)
- [DataLoader使用指南](../docs/dataloader_usage.md)
