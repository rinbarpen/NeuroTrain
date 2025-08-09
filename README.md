# NeuroTrain

NeuroTrain 是一个功能强大的、基于 PyTorch 的深度学习框架，旨在简化和加速医学影像分析等领域的科研和开发流程。它通过高度可配置化的设计，支持从数据处理、模型训练、测试到推理的完整工作流，并提供管道功能以实现全自动化实验。

## ✨ 功能特性

- **配置驱动**: 使用 TOML、YAML 或 JSON 管理实验参数，且支持命令行参数部分重载。
- **单次与管道运行**: 支持通过 `main.py` 执行单个独立的任务，也支持通过 `main_pipeline.py` 自动化执行一系列任务。
- **实用工具集**: 提供实验的可解释性分析工具，如注意力图可视化、模型分析等。
- **支持训练中断后继续**: 训练过程中可以中断，后续可以通过配置文件指定从中断点继续训练。
- **支持分布式训练**: 利用多个 GPU 训练，加速模型训练过程。

## 🚀 安装

项目使用 `pyproject.toml` 管理依赖。首先，请确保您已安装 Python 3.10 或更高版本。

1.  克隆本仓库并进入项目目录：
    ```bash
    git clone https://github.com/rinbarpen/NeuroTrain.git
    cd NeuroTrain
    ```

2.  安装基础依赖：
    ```bash
    uv pip install -e '.[cu128]' # cpu cu118 cu126
    ```
    or
    ```bash
    uv sync --extra cu128 # cpu cu118 cu126
    ```

## 💡 如何使用

### 1. 单次运行

使用 `main.py` 执行单个任务（例如训练、测试或预测）。您需要提供一个任务配置文件。

#### 测试配置是否有效
```bash
python main.py -c configs/single/train.template.toml --check
```
#### 训练模型并测试推理
```bash
python main.py -c configs/single/train.template.toml --train --test --predict
```

您可以在 `configs/single/` 目录中找到更多单次运行的配置示例。

### 2. 管道运行

使用 `main_pipeline.py` 执行一系列预定义的任务。这需要一个管道配置文件。

```bash
python main_pipeline.py -c configs/pipeline/pipeline-template.toml
```

您可以在 `configs/pipeline/` 目录中找到管道配置的示例。

## 📂 项目结构

```
NeuroTrain/
├── configs/              # 存放所有 TOML/YAML/JSON 配置文件
├── src/                  # 核心源代码
│   ├── dataset/          # 数据集加载和处理模块
│   ├── engine/           # 训练、测试、预测的核心逻辑
│   └── utils/            # 各种工具函数
├── tools/                # 独立的分析和辅助工具
├── scripts/              # 用于快速执行任务的 Shell 脚本
├── main.py               # 单次任务执行入口
├── main_pipeline.py      # 管道任务执行入口
├── pyproject.toml        # 项目依赖和元数据配置文件
└── README.md             # 本文档
```

## 🔧 配置

所有实验都由配置文件驱动。我们强烈建议您将所有自定义配置保存在 `configs/` 目录下，并参考其中的模板文件（如 `configs/single/train.template.toml`）来创建您自己的实验配置。

## 训练与推理输出

所有实验的输出（如模型检查点、训练日志、推理结果等）将被保存到 `outputs/` 目录下。每个实验都有一个唯一的子目录，目录名基于实验配置文件的名称。

### 输出目录结构

`outputs/` 目录用于存放所有实验的产出，其结构通常如下：

```
outputs/
└── {task_name}/              # 实验任务名
    ├── {run_id}/             # 实验运行ID，默认使用运行的时间戳
    │   ├── train/            # 训练相关输出
    │   │   ├── recovery/     # 训练中断恢复相关输出
    │   │   ├── weights/      # 模型检查点输出
    │   │   |   ├── {model_name}-{epoch}of{num_epochs}{.ext}.pth  # 每个 epoch 的模型检查点
    │   │   |   └── [best|last]{.ext}.pth  # 最佳模型和最后一个模型的检查点
    │   │   ├── {class_labels}/          # 每个类别的指标输出
    |   │   │   ├── [all|mean|std]_metric.csv  # 每个类别的所有指标、均值指标和标准差指标
    |   │   │   ├── {metrics}.png  # 每个类别的指标可视化图
    |   │   │   ├── metrics.png  # 所有类别的指标可视化图
    |   │   │   └── mean_metric.png  # 所有类别的均值指标可视化图
    │   │   ├── config[.json|.toml|.yaml]  # 实验配置文件
    │   │   ├── epoch_metrics_curve_per_classes.png  # 每个类别的指标曲线可视化图
    │   │   ├── epoch_metrics_curve.png  # 所有类别的指标曲线可视化图
    │   │   ├── [mean|std]_metric.csv  # 所有类别的均值指标和标准差指标
    │   │   ├── mean_metrics_per_classes.png  # 每个类别的均值指标可视化图
    │   │   ├── mean_metrics.png  # 所有类别的均值指标可视化图
    │   │   ├── train_epoch_loss.png  # 每个 epoch 的训练损失可视化图
    │   │   └── train_loss.csv  # 每个 epoch 的训练损失
    │   ├── test/             # 测试相关输出
    │   │   ├── {metrics}/    # 每个指标的输出
    │   │   │   ├── [all|mean|std]_metric.csv  # 每个类别的所有指标、均值指标和标准差指标
    │   │   │   └── mean_metric.png  # 每个指标的可视化图
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