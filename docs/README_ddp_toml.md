# DDP训练TOML配置文件说明

## 概述

本文档介绍了DDP（Distributed Data Parallel）训练的TOML配置文件格式和使用方法。TOML格式相比YAML更加简洁，适合配置复杂的嵌套结构。

## 配置文件列表

### 1. `ddp_example.toml` - 通用DDP配置
- **用途**：通用的DDP训练配置模板
- **适用场景**：图像分类、目标检测等任务
- **特点**：包含完整的配置选项，适合大多数深度学习任务

### 2. `ddp_segmentation.toml` - 图像分割配置
- **用途**：专门用于图像分割任务的DDP训练
- **适用场景**：医学图像分割、语义分割等
- **特点**：使用UNet模型，配置了Dice和BCE损失函数

### 3. `ddp_large_model.toml` - 大模型训练配置
- **用途**：用于大型深度学习模型的分布式训练
- **适用场景**：ResNet50、EfficientNet等大型模型
- **特点**：优化的学习率调度、混合精度训练

### 4. `ddp_quick_test.toml` - 快速测试配置
- **用途**：快速验证DDP功能
- **适用场景**：MNIST数据集测试
- **特点**：配置简单，训练轮数少，适合快速验证

## 配置项说明

### 基本配置
```toml
output_dir = "./runs"        # 输出目录
task = "DDP_Training"        # 任务名称
entity = "Lab"               # 实体名称
run_id = ""                  # 运行ID（程序自动生成）
device = "cuda"              # 设备类型
seed = 42                    # 随机种子
classes = ["class_0", "class_1"]  # 类别列表
metrics = ['accuracy', 'precision']  # 评估指标
postprocess = "classification"  # 后处理类型
```

### DDP配置
```toml
[ddp]
enabled = true              # 启用DDP
log_level = "INFO"         # 日志级别
```

### 模型配置
```toml
[model]
name = "resnet18"          # 模型名称
continue_checkpoint = ""   # 继续训练检查点
pretrained = ""            # 预训练模型路径

[model.config]
arch = "resnet18"          # 模型架构
n_channels = 3             # 输入通道数
n_classes = 10            # 类别数
input_sizes = [
    [1, 3, 224, 224]       # 输入尺寸
]
dtypes = ["float32"]       # 数据类型
pretrained = true          # 使用预训练权重
```

### 损失函数配置
```toml
[[criterion]]
type = 'cross_entropy'     # 损失函数类型
weight = 1.0               # 权重
config = {}                # 额外配置
```

### 数据变换配置
```toml
[transform]
RESIZE = [224, 224]        # 调整大小
HFLIP = [0.5]              # 水平翻转概率
VFLIP = [0.1]              # 垂直翻转概率
ROTATION = [10]            # 旋转角度
PIL_TO_TENSOR = []         # 转换为张量
CONVERT_IMAGE_DTYPE = ['float32']  # 数据类型转换
NORMALIZE = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]  # 归一化
```

### 数据集配置
```toml
[dataset]
name = "CustomDataset"     # 数据集名称
root_dir = "./data"        # 数据根目录
config = {}                # 数据集配置
ref = "custom.yaml"        # 参考配置文件

[dataset.split]
type = "train_valid_test_split"  # 分割类型
train = 0.8               # 训练集比例
valid = 0.1               # 验证集比例
test = 0.1                # 测试集比例
shuffle = true            # 是否打乱
random_state = 42         # 随机种子
```

### 数据加载器配置
```toml
[dataloader]
num_workers = 4            # 工作进程数
shuffle = true             # 是否打乱
pin_memory = true          # 是否固定内存
drop_last = true           # 是否丢弃最后一批
```

### 训练配置
```toml
[train]
batch_size = 32            # 批次大小
epoch = 100                # 训练轮数
save_period = 0.1          # 保存周期
save_recovery_period = 0.1 # 恢复信息保存周期
grad_accumulation_steps = 1 # 梯度累积步数

[train.optimizer]
type = 'adamw'             # 优化器类型
learning_rate = 1e-3       # 学习率
weight_decay = 1e-4        # 权重衰减

[train.lr_scheduler]
type = 'cosine'            # 学习率调度器
warmup = 10                # 预热轮数
warmup_lr = 1e-5           # 预热学习率
update_policy = 'epoch'    # 更新策略

[train.scaler]
compute_type = 'float16'   # 混合精度类型

[train.early_stopping]
patience = 10              # 早停耐心值
min_delta = 0.001         # 最小改善阈值
```

### 测试和预测配置
```toml
[test]
batch_size = 32            # 测试批次大小

[predict]
input = "data/test"        # 预测输入路径

[predict.config]
output_format = "json"      # 输出格式
save_images = true         # 是否保存图像
```

### 私有配置
```toml
[private]
wandb = false              # 是否使用wandb
mode = 0                   # 运行模式

[private.log]
verbose = true             # 详细日志
debug = false              # 调试模式
log_file_format = '%Y-%m-%d %H_%M_%S'  # 日志文件格式
log_format = '%(asctime)s %(levelname)s | %(name)s | %(message)s'  # 日志格式
```

### 输出配置
```toml
[output]
save_best = true           # 保存最佳模型
save_last = true           # 保存最后模型
save_interval = 10         # 保存间隔
```

## 使用方法

### 1. 选择配置文件
根据任务类型选择合适的配置文件：
- 通用任务：`ddp_example.toml`
- 图像分割：`ddp_segmentation.toml`
- 大模型训练：`ddp_large_model.toml`
- 快速测试：`ddp_quick_test.toml`

### 2. 修改配置
根据实际需求修改配置项：
- 数据集路径
- 模型架构
- 训练参数
- 输出路径

### 3. 启动训练
```bash
# 使用torchrun启动
torchrun --nproc_per_node=4 main.py --config configs/ddp_example.toml --task train

# 使用启动脚本
./run_ddp.sh
```

## 配置优化建议

### 1. 批次大小
- 根据GPU内存调整批次大小
- 使用梯度累积增加有效批次大小
- 确保批次大小能被GPU数量整除

### 2. 学习率
- 大模型使用较小的学习率
- 使用学习率预热
- 根据任务调整学习率调度策略

### 3. 数据加载
- 根据CPU核心数调整`num_workers`
- 启用`pin_memory`提高数据传输效率
- 使用适当的数据增强策略

### 4. 混合精度
- 启用混合精度训练节省内存
- 选择合适的计算类型（float16/bfloat16）
- 注意数值稳定性

## 故障排除

### 1. 内存不足
- 减少批次大小
- 增加梯度累积步数
- 使用混合精度训练

### 2. 训练不稳定
- 调整学习率
- 使用学习率预热
- 检查数据预处理

### 3. 性能问题
- 调整`num_workers`
- 启用`pin_memory`
- 检查网络配置

## 示例命令

```bash
# 使用通用配置训练
torchrun --nproc_per_node=4 main.py --config configs/ddp_example.toml --task train

# 使用分割配置训练
torchrun --nproc_per_node=4 main.py --config configs/ddp_segmentation.toml --task train

# 使用大模型配置训练
torchrun --nproc_per_node=8 main.py --config configs/ddp_large_model.toml --task train

# 快速测试
torchrun --nproc_per_node=2 main.py --config configs/ddp_quick_test.toml --task train
```
