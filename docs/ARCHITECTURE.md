# NeuroTrain 架构文档

## 项目概览

NeuroTrain 是一个专业的深度学习训练框架，专为多种深度学习任务设计，包括医学图像分割、目标检测、图像分类、扩散模型等。本文档详细介绍了项目的整体架构和各个模块的职责。

## 目录结构

```
NeuroTrain/
├── configs/              # 配置文件目录
│   ├── single/          # 单次训练配置
│   └── pipeline/        # 管道配置
├── src/                  # 核心源代码
│   ├── dataset/         # 数据集模块
│   ├── models/          # 模型模块
│   ├── engine/          # 训练引擎
│   ├── metrics/         # 评估指标
│   ├── utils/           # 工具函数
│   ├── quantization/    # 模型量化
│   ├── monitor/         # 训练监控
│   ├── recorder/        # 数据记录
│   ├── visualizer/      # 可视化工具
│   └── prompts/         # LLM提示模板
├── data/                # 数据集存储目录
├── runs/                # 训练结果输出目录
├── cache/               # 模型缓存目录
│   └── models/
│       └── pretrained/  # 预训练模型
├── tests/               # 测试文件
├── docs/                # 文档目录
├── tools/               # 分析工具和脚本
│   ├── analyzers/       # 各种分析器
│   └── toolkit/         # 工具集
├── scripts/             # 辅助脚本
├── examples/            # 使用示例
├── libs/                # 第三方库
└── notebooks/           # Jupyter notebooks
```

## 核心模块

### 1. 数据集模块 (src/dataset)

数据集模块负责数据加载、预处理和增强。

#### 主要组件

- **CustomDataset**: 自定义数据集基类
- **HybridDataset**: 混合数据集，支持多数据集混合训练
- **DiffusionDataset**: 扩散模型数据集
- **Medical Datasets**: 医学图像数据集集合
  - Drive (视网膜血管分割)
  - ChaseDB1 (视网膜血管分割)
  - STARE (视网膜血管分割)
  - ISIC (皮肤病变)
  - BTCV (多器官分割)
  - BraTS (脑肿瘤分割)
- **Standard Datasets**: 标准计算机视觉数据集
  - MNIST (手写数字)
  - CIFAR-10/100 (图像分类)
  - COCO (目标检测/分割)
  - ImageNet (图像分类)

#### 特性

- 数据集注册表 (`DATASET_REGISTRY`)
- 智能采样策略（权重采样、平衡采样、优先级采样）
- 丰富的数据增强选项
- LLM驱动的数据分析 (`llm_data_analyzer.py`)

#### 关键函数

```python
get_dataset(config)              # 根据配置获取数据集
get_train_dataset(config)        # 获取训练数据集
get_test_dataset(config)         # 获取测试数据集
get_valid_dataset(config)        # 获取验证数据集
random_sample(dataset, n)        # 随机采样
get_train_valid_test_dataloader() # 获取数据加载器
```

### 2. 模型模块 (src/models)

模型模块提供了多种深度学习模型的实现和接口。

#### 主要组件

- **models.py**: 模型工厂函数，统一的模型获取接口
- **sample/**: 示例模型实现
  - UNet (医学图像分割)
  - SimpleNet (简单网络示例)
- **llm/**: 大语言模型和多模态模型
  - CLIP (图像-文本对比学习)
  - LLaVA (视觉语言助手)
- **like/**: 类库风格的模型（ResNet-like, VGG-like等）
- **attention/**: 注意力机制实现
- **transformer/**: Transformer相关组件
- **conv/**: 卷积层变体
- **norm/**: 归一化层
- **gate/**: 门控机制
- **embedding.py**: 嵌入层实现
- **position_encoding.py**: 位置编码

#### 支持的模型后端

1. **自定义模型**: UNet, SimpleNet等
2. **TorchVision模型**: ResNet, VGG, DenseNet, EfficientNet等
3. **TIMM模型**: 数百种预训练模型
4. **CLIP**: 多模态图像-文本模型
5. **自定义架构**: 可通过配置灵活组合各种组件

#### 关键函数

```python
get_model(model_name, config)  # 获取模型实例
```

### 3. 训练引擎 (src/engine)

训练引擎负责训练、测试和推理流程的管理。

#### 主要组件

- **Trainer**: 标准训练器
  - 支持混合精度训练
  - 自动保存检查点
  - 早停机制
  - 恢复训练
  - 训练指标记录
  
- **DeepSpeedTrainer**: DeepSpeed分布式训练器
  - 支持大规模模型训练
  - ZeRO优化
  - 梯度累积
  
- **Tester**: 模型测试器
  - 评估模型性能
  - 生成详细指标报告
  - 可视化预测结果
  
- **Predictor**: 模型预测器
  - 批量预测
  - 单样本预测
  - 结果后处理

#### 训练流程

1. 初始化模型和优化器
2. 加载数据集
3. 训练循环
   - 前向传播
   - 计算损失
   - 反向传播
   - 更新参数
   - 记录指标
4. 验证
5. 保存检查点
6. 早停判断

### 4. 指标模块 (src/metrics)

提供各种评估指标的计算函数。

#### 分割任务指标

- `dice`: Dice系数
- `dice_coefficient`: Dice系数（别名）
- `iou_seg`: IoU (Intersection over Union)
- `nsd`: 归一化表面距离 (Normalized Surface Dice)
- `mAP_at_iou_seg`: 分割任务的mAP
- `mF1_at_iou_seg`: 分割任务的mF1

#### 检测任务指标

- `iou_bbox`: 边界框IoU
- `mAP_at_iou_bbox`: 检测任务的mAP
- `mF1_at_iou_bbox`: 检测任务的mF1

#### 分类任务指标

- `accuracy`: 准确率
- `precision`: 精确率
- `recall`: 召回率
- `f1`: F1分数
- `auc`: AUC (Area Under Curve)
- Top-K指标: `top1_accuracy`, `top3_accuracy`, `top5_accuracy`等

#### 多模态任务指标

- **CLIP指标**: 
  - `clip_accuracy`: CLIP准确率
  - `image_retrieval_recall_at_k`: 图像检索召回率
  - `text_retrieval_recall_at_k`: 文本检索召回率
  - `image_text_similarity`: 图像-文本相似度
  
- **BLEU指标**: 
  - `bleu_1`, `bleu_2`, `bleu_3`, `bleu_4`: BLEU-N分数
  - `corpus_bleu`: 语料库级别BLEU
  - `sentence_bleu`: 句子级别BLEU

#### 阈值相关指标

- `at_threshold`: 在指定阈值下的指标
- `at_accuracy_threshold`, `at_recall_threshold`等
- `mAP`: Mean Average Precision

### 5. 工具模块 (src/utils)

提供各种辅助工具函数。

#### 主要组件

- **criterion.py**: 损失函数
  - CrossEntropyLoss
  - BCEWithLogitsLoss
  - DiceLoss
  - FocalLoss
  - 组合损失函数

- **transform.py**: 数据增强和转换
  - 标准化
  - 随机裁剪
  - 随机翻转
  - 旋转
  - 弹性变形
  
- **early_stopping.py**: 早停机制
  
- **image_utils.py**: 图像处理工具
  - 读取/保存图像
  - 颜色空间转换
  - 图像预处理
  
- **postprocess.py**: 后处理函数
  - 阈值化
  - 形态学操作
  - 连通域分析
  
- **ddp_utils.py**: 分布式训练工具
  
- **deepspeed_utils.py**: DeepSpeed工具
  
- **timer.py**: 计时器
  
- **db.py**: 数据库工具（SQLModel）
  
- **llm/**: LLM相关工具
  
- **medical/**: 医学图像处理工具

### 6. 量化模块 (src/quantization)

提供模型量化功能，减小模型大小和推理时间。

#### 支持的量化方法

- **动态量化**: 无需校准数据
- **静态量化**: 需要校准数据集
- **量化感知训练 (QAT)**: 训练时考虑量化
- **GPTQ**: 适用于大语言模型
- **AWQ**: 保持激活精度
- **BitsAndBytes**: 4bit/8bit量化

#### 主要组件

- **config.py**: 量化配置
- **core.py**: 量化核心功能
- **trainer.py**: 量化感知训练

### 7. 监控模块 (src/monitor)

提供训练过程监控和可视化。

- 实时训练进度
- GPU使用率监控
- 内存使用监控
- 训练曲线可视化

### 8. 记录模块 (src/recorder)

负责训练过程中的数据记录和保存。

- **MeterRecorder**: 指标记录器
- **DataSaver**: 数据保存器
- 支持恢复训练时加载历史记录

### 9. 可视化模块 (src/visualizer)

提供各种可视化功能。

- **painter.py**: 绘图工具
  - 训练曲线
  - 指标对比
  - 混淆矩阵
  
- 预测结果可视化
- 特征图可视化
- 注意力图可视化

## 工具集 (tools)

### 分析工具 (tools/analyzers)

- **metrics_analyzer.py**: 指标分析
- **dataset_analyzer.py**: 数据集分析
- **attention_analyzer.py**: 注意力分析
- **mask_analyzer.py**: 掩码分析
- **lora_analyzer.py**: LoRA分析
- **relation_analyzer.py**: 关系分析

### 其他工具

- **config_converter.py**: 配置文件转换
- **onnx_export.py**: ONNX导出
- **quantization_cli.py**: 量化命令行工具
- **lora_merge.py**: LoRA合并工具
- **to_parquet.py**: 数据格式转换
- **checker.py**: 配置检查器
- **cleanup.py**: 清理工具

## 配置系统

NeuroTrain使用TOML格式的配置文件来管理实验参数。

### 配置文件结构

```toml
[basic]
task_name = "任务名称"
run_id = "实验ID"

[model]
name = "模型名称"
# 模型特定配置

[dataset]
name = "数据集名称"
# 数据集特定配置

[training]
epochs = 100
batch_size = 8
learning_rate = 0.001
# 其他训练参数

[testing]
# 测试参数

[augmentation]
# 数据增强参数
```

### 配置加载

```python
from src.config import get_config, get_config_value

# 获取完整配置
config = get_config()

# 获取特定配置值
batch_size = get_config_value('training.batch_size')
```

## 训练流程

### 标准训练流程

```python
from src.engine import Trainer
from src.models import get_model
from src.dataset import get_train_valid_test_dataloader

# 1. 加载配置
config = get_config()

# 2. 创建模型
model = get_model(config['model']['name'], config['model'])

# 3. 准备数据
train_loader, valid_loader, test_loader = get_train_valid_test_dataloader(config)

# 4. 创建训练器
trainer = Trainer(output_dir, model)

# 5. 开始训练
trainer.train(train_loader, valid_loader)
```

### 分布式训练流程

```python
from src.engine import DeepSpeedTrainer

# 创建DeepSpeed训练器
trainer = DeepSpeedTrainer(output_dir, model, deepspeed_config)

# 训练
trainer.train(train_loader, valid_loader)
```

## 测试和评估流程

```python
from src.engine import Tester

# 创建测试器
tester = Tester(output_dir, model)

# 运行测试
results = tester.test(test_loader)

# 查看结果
print(results)
```

## 扩展和自定义

### 添加自定义数据集

1. 在 `src/dataset/` 中创建新的数据集类
2. 继承 `CustomDataset` 或 `torch.utils.data.Dataset`
3. 在 `DATASET_REGISTRY` 中注册数据集
4. 在配置文件中使用

### 添加自定义模型

1. 在 `src/models/` 中创建新的模型类
2. 在 `get_model()` 函数中添加模型选项
3. 在配置文件中指定模型名称和参数

### 添加自定义指标

1. 在 `src/metrics/` 中创建新的指标函数
2. 在 `__init__.py` 中导出
3. 在配置文件中添加到 `metrics` 列表

### 添加自定义损失函数

1. 在 `src/utils/criterion.py` 中添加损失函数
2. 在配置文件中指定损失函数名称

## 最佳实践

1. **使用配置文件**: 通过TOML文件管理所有实验参数
2. **模块化设计**: 每个模块职责单一，易于维护和扩展
3. **日志记录**: 使用logging模块记录训练过程
4. **检查点保存**: 定期保存模型检查点，支持恢复训练
5. **混合精度**: 使用混合精度训练加速训练过程
6. **数据增强**: 合理使用数据增强提高模型泛化能力
7. **早停机制**: 使用早停避免过拟合
8. **分布式训练**: 对于大规模模型，使用分布式训练

## 性能优化

1. **数据加载优化**
   - 使用 `num_workers` 多进程加载
   - 使用 `pin_memory` 加速数据传输
   - 预加载数据到内存

2. **训练优化**
   - 混合精度训练 (AMP)
   - 梯度累积
   - 梯度裁剪
   - 学习率调度

3. **模型优化**
   - 模型量化
   - 模型剪枝
   - 知识蒸馏
   - ONNX导出

4. **分布式优化**
   - 数据并行 (DDP)
   - 模型并行
   - DeepSpeed ZeRO

## 常见问题

### 内存不足

- 减小 `batch_size`
- 使用梯度累积
- 使用混合精度训练
- 使用DeepSpeed ZeRO

### 训练速度慢

- 增加 `num_workers`
- 使用混合精度训练
- 使用分布式训练
- 优化数据增强流程

### 模型不收敛

- 调整学习率
- 尝试不同的优化器
- 检查损失函数
- 检查数据预处理
- 增加训练轮数

## 未来计划

- [ ] 更多模型架构支持
- [ ] 自动超参数调优 (AutoML)
- [ ] Web界面管理工具
- [ ] 更多医学图像数据集支持
- [ ] 联邦学习支持
- [ ] 增量学习支持
- [ ] 神经架构搜索 (NAS)

## 参考资源

- [PyTorch文档](https://pytorch.org/docs/)
- [TorchVision文档](https://pytorch.org/vision/)
- [TIMM文档](https://timm.fast.ai/)
- [DeepSpeed文档](https://www.deepspeed.ai/)
- [Hugging Face文档](https://huggingface.co/docs)

---

本架构文档随项目发展持续更新。如有问题或建议，请提交Issue或Pull Request。

