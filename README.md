# NeuroTrain

一个专业的深度学习训练框架，专为医学图像分割任务设计，提供灵活的数据集管理、模型训练和结果分析功能。

## 🚀 特性

- **多数据集支持**: 支持DRIVE、CHASEDB1、STARE等医学图像数据集，以及COCO目标检测/分割数据集和CIFAR图像分类数据集
- **Diffusion模型支持**: 完整支持扩散模型训练，包括无条件生成、条件生成和文本到图像生成 🆕
- **LLM驱动的数据分析**: 使用自然语言查询分析、筛选和处理数据集 🆕
- **灵活的数据集配置**: 单一数据集、混合数据集、增强版混合数据集
- **智能采样策略**: 权重采样、平衡采样、优先级采样
- **丰富的数据增强**: 旋转、翻转、亮度调整、弹性变换等
- **混合精度训练**: 支持bfloat16混合精度训练，提升训练效率
- **模型量化支持**: 支持动态量化、静态量化、QAT、GPTQ、AWQ、BitsAndBytes等多种量化方法
- **模块化设计**: 易于扩展和自定义
- **完整的实验管理**: 自动保存训练日志、模型检查点和结果分析
- **配置驱动**: 使用 TOML 配置文件管理实验参数
- **支持训练中断后继续**: 训练过程中可以中断，后续可以从检查点继续训练
- **分布式训练**: 支持多GPU训练，加速模型训练过程

## 📦 安装

### 环境要求
- Python 3.10–3.13（推荐，CI 已覆盖）
- CUDA 11.8+ (可选，用于GPU训练)
- Conda 或 Miniconda

### 安装步骤

1. **克隆项目**
   ```bash
   git clone https://github.com/rinbarpen/NeuroTrain.git
   cd NeuroTrain
   ```

2. **创建并激活Conda环境**
   ```bash
   conda create -n ntrain python=3.10
   conda activate ntrain
   ```

3. **安装依赖**
   ```bash
   # 使用uv安装依赖（推荐）
   uv pip install -e '.[cu128]'  # 支持CUDA 12.8
   # 或者选择其他版本：cpu, cu118, cu126
   
   # 或者使用传统方式
   pip install -e '.[cu128]'
   ```

4. **验证安装**
   ```bash
   python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
   python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
   ```

## 🚀 快速开始

### 1. 准备数据

将数据集放在 `data` 目录下，例如：
```
data/
├── drive/
│   ├── images/
│   └── masks/
├── chasedb1/
│   ├── images/
│   └── masks/
├── stare/
│   ├── images/
│   └── masks/
└── coco/                    # COCO数据集（可选）
    ├── annotations/
    ├── train2017/
    └── val2017/
```

**下载COCO数据集（可选）**：
```bash
# 使用自动下载脚本
bash scripts/download_coco.sh 2017 all

# 或者手动下载，详见 docs/COCO_DATASET_GUIDE.md
```

### 2. 配置训练参数

复制并修改配置文件：
```bash
cp configs/single/train.template.toml configs/my_training.toml
```

### 3. 开始训练

```bash
# 激活环境
conda activate ntrain

# 检查配置
python main.py -c configs/my_training.toml --check

# 开始训练
python main.py -c configs/my_training.toml --train

# 训练并测试
python main.py -c configs/my_training.toml --train --test
```

## 📖 详细文档

我们提供了完整的文档来帮助您使用NeuroTrain框架：

- **[训练指南](docs/training_guide.md)** - 详细的模型训练教程
- **[数据集配置指南](docs/dataset_configuration_guide.md)** - 如何配置各种数据集
- **[增强版混合数据集指南](docs/enhanced_hybrid_dataset_guide.md)** - 高级数据集配置
- **[Diffusion数据集指南](docs/DIFFUSION_DATASET.md)** - Diffusion模型数据集使用指南 🆕
- **[LLM数据分析器指南](docs/LLM_DATA_ANALYZER.md)** - LLM驱动的数据分析和筛选 🆕
- **[结果分析指南](docs/results_analysis_guide.md)** - 如何分析训练结果
- **[故障排除指南](docs/troubleshooting_guide.md)** - 常见问题解决方案

## 📂 项目结构

```
NeuroTrain/
├── configs/              # 配置文件
│   ├── single/          # 单次训练配置
│   └── pipeline/        # 管道配置
├── src/                  # 核心源代码
│   ├── dataset/         # 数据集模块
│   ├── engine/          # 训练引擎
│   ├── metrics/         # 评估指标
│   └── utils/           # 工具函数
├── data/                # 数据集存储
├── runs/                # 训练结果输出
├── cache/               # 模型缓存
├── tests/               # 测试文件
├── docs/                # 文档
├── tools/               # 分析工具
├── scripts/             # 脚本文件
└── examples/            # 使用示例
```

## 🔧 配置示例

### 基础训练配置

```toml
[basic]
task_name = "Retina_Vessel_Segmentation"
run_id = "experiment_001"

[model]
name = "UNet"
n_channels = 3
n_classes = 2

[dataset]
name = "drive"
root_dir = "data/drive"
is_rgb = true

[training]
epochs = 100
batch_size = 8
learning_rate = 0.001
```

### 增强版混合数据集配置

```toml
[dataset]
name = "enhanced_hybrid"
datasets = ["drive", "chasedb1", "stare"]
sampling_strategy = "weighted"
ratios = [0.5, 0.3, 0.2]
weights = [1.0, 1.2, 0.8]

[dataset.drive]
root_dir = "data/drive"
is_rgb = true

[dataset.chasedb1]
root_dir = "data/chasedb1"
is_rgb = true

[dataset.stare]
root_dir = "data/stare"
is_rgb = true
```

## 🚀 使用场景

### 1. 单数据集训练
适用于在单个数据集上训练模型：
```bash
python main.py -c configs/single/train-drive.toml --train
```

### 2. 多数据集混合训练
使用多个数据集进行联合训练：
```bash
python main.py -c configs/single/train.enhanced.template.toml --train
```

### 3. 模型测试和评估
对训练好的模型进行测试：
```bash
python main.py -c configs/my_config.toml --test
```

### 4. 批量实验管道
执行一系列预定义的实验：
```bash
python main_pipeline.py -c configs/pipeline/pipeline-template.toml
```

### 5. Diffusion模型训练 🆕
训练扩散模型进行图像生成：
```bash
# 运行示例
python examples/diffusion_dataset_example.py

# 或使用配置文件
python main.py -c examples/config_diffusion_example.toml --train
```

了解更多：[Diffusion数据集指南](docs/DIFFUSION_DATASET.md) 和 [快速开始README](DIFFUSION_DATASET_README.md)

## 📊 结果分析

训练结果保存在 `runs/{run_id}/` 目录下：

```
runs/
└── experiment_001/
    ├── train/
    │   ├── model_summary.txt      # 模型结构摘要
    │   ├── model_flop_count.txt   # 计算复杂度分析
    │   └── training.log           # 训练日志
    ├── test/
    │   ├── metrics.json           # 测试指标
    │   └── predictions/           # 预测结果
    └── checkpoints/
        ├── best.pth              # 最佳模型
        └── last.pth              # 最新模型
```

## 🔍 监控和调试

### 查看训练日志
```bash
tail -f runs/{run_id}/train/training.log
```

### 分析模型性能
```bash
python tools/analyzers/metrics_analyzer.py --run_id {run_id}
```

### 可视化训练过程

## 🚀 模型量化

NeuroTrain支持多种模型量化方法，帮助减少模型大小和推理时间：

### 支持的量化方法
- **动态量化**: PyTorch内置，无需校准数据
- **静态量化**: PyTorch内置，需要校准数据集
- **量化感知训练(QAT)**: 训练时考虑量化影响
- **GPTQ量化**: 适用于大语言模型
- **AWQ量化**: 保持激活精度
- **BitsAndBytes量化**: 4bit/8bit量化

### 快速开始量化

```python
from src.utils.quantization import QuantizationConfig, QuantizationManager

# 创建量化配置
config = QuantizationConfig(method="dynamic", dtype="qint8")

# 量化模型
manager = QuantizationManager(config)
quantized_model = manager.quantize_model(your_model)

# 获取模型信息
size_info = manager.get_model_size_info(quantized_model)
print(f"模型大小: {size_info['model_size_mb']:.2f}MB")
```

### 命令行量化工具

```bash
# 量化模型
python tools/quantization_cli.py quantize model.pt output/ --method dynamic

# 分析量化效果
python tools/quantization_cli.py analyze original.pt quantized.pt analysis/

# 运行示例
python tools/quantization_cli.py example --method dynamic
```

### 配置文件量化

```yaml
# config.yaml
quantization:
  enabled: true
  method: "dynamic"
  dtype: "qint8"
```

详细使用说明请参考 [量化模块文档](src/quantization/README.md)。

## 🧪 测试

运行测试以确保框架正常工作：

```bash
# 激活环境
conda activate ntrain

# 运行所有测试
python -m pytest tests/ -v

# 运行特定测试
python -m pytest tests/test_dataset_creation.py -v

# 测试数据集创建
python test_dataset_creation.py

# 测试增强配置
python test_enhanced_config.py
```

## 🤝 贡献

欢迎贡献代码！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📝 更新日志

### v1.2.0 (最新)
- ✨ 新增CIFAR-10和CIFAR-100数据集支持，用于图像分类任务
- 📊 数据集统计信息（mean/std）作为静态变量，便于访问
- 🔧 新增metadata静态方法，提供类别信息和推荐metrics
- 📚 添加CIFAR数据集使用指南和示例代码
- ✅ 支持自动下载CIFAR数据集

### v1.1.0
- ✨ 新增COCO数据集支持，支持目标检测、实例分割、关键点检测和图像描述任务
- 📚 添加COCO数据集使用指南和示例代码
- 🛠️ 提供COCO数据集自动下载脚本
- ✅ 添加COCO数据集单元测试

### v1.0.0
- ✨ 新增增强版混合数据集支持
- ✨ 支持多种采样策略（权重、平衡、优先级）
- ✨ 完善的文档系统
- 🐛 修复RGB通道不匹配问题
- 🔧 优化配置文件结构

### v0.9.0
- ✨ 基础混合数据集功能
- ✨ UNet模型支持
- ✨ 基础训练和测试流程

## 🔮 计划功能

- [x] 通用目标检测数据集支持（COCO）✅
- [x] 图像分类数据集支持（CIFAR-10/100）✅
- [ ] 更多深度学习模型支持（ResNet、DenseNet、Transformer等）
- [ ] 自动超参数调优
- [ ] 分布式训练优化
- [ ] 模型量化和剪枝
- [ ] 更多医学图像数据集支持
- [ ] Web界面管理工具

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

感谢以下开源项目的支持：
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [torchvision](https://pytorch.org/vision/) - 计算机视觉工具
- [Albumentations](https://albumentations.ai/) - 图像增强库
- [MONAI](https://monai.io/) - 医学图像分析工具

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 [Issue](https://github.com/rinbarpen/NeuroTrain/issues)
- 发送邮件至：[your-email@example.com]
- 项目主页：[https://github.com/rinbarpen/NeuroTrain]

---

**NeuroTrain** - 让医学图像分割变得简单高效！ 🚀
    │   │   ├── config[.json|.toml|.yaml]  # 实验配置文件
    │   │   ├── [mean|std]_metric.csv  # 所有类别的均值指标和标准差指标
    │   │   ├── mean_metrics_per_classes.png  # 每个类别的均值指标可视化图
    │   │   └── mean_metrics.png  # 所有类别的均值指标可视化图
    │   ├── predict/          # 预测相关输出
    │   │   ├── {predicted_files}  # 预测输出文件
    │   │   └── config[.json|.toml|.yaml]  # 实验配置文件
    ├── logs/                 # 训练日志
    |   └── [train|test|predict].log  # 每个任务的日志文件
    │   model_flop_count.txt  # 模型 FLOP 统计
    └── model_summary.txt     # 模型参数统计
```

## 贡献

我们欢迎所有形式的贡献，包括但不限于代码、文档、问题报告和功能建议。请通过提交 Pull Request 或打开 Issue 来参与贡献。

## 许可证

本项目基于 MIT 许可证开源。详情请见 `LICENSE` 文件。