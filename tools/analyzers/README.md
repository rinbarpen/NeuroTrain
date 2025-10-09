# Analyzer Module

这是一个综合的模型分析工具模块，提供多种分析功能来帮助理解和优化深度学习模型。

## 模块结构

```
tools/analyzers/
├── __init__.py              # 统一接口和便捷函数
├── attention_analyzer.py    # 注意力机制分析器（支持SE模块等）
├── data_analyzer.py         # 训练数据和指标分析器
├── dataset_analyzer.py      # 数据集质量分析器
├── mask_analyzer.py         # Mask信息分析器（图像和文本）
├── relation_analyzer.py     # 跨模态关系分析器（类似CLIP）
└── README.md               # 本文档
```

## 🔧 核心分析器

### 1. AttentionAnalyzer - 注意力分析器

专门用于分析和可视化深度学习模型中的注意力机制。

**主要功能：**
- 注意力权重提取和分析
- 多头注意力可视化
- 注意力模式分析
- 注意力流可视化
- 注意力热图生成

**使用示例：**
```python
from tools.analyzers import AttentionAnalyzer

# 创建分析器
analyzer = AttentionAnalyzer(model=your_model)

# 分析注意力模式
results = analyzer.analyze_attention_patterns(input_data)

# 生成可视化
analyzer.visualize_attention_weights(input_data, layer_name='attention')
```

### 2. MetricsAnalyzer - 数据指标分析器

提供全面的模型性能指标计算和分析功能。

**主要功能：**
- 分类任务指标（准确率、精确率、召回率、F1等）
- 分割任务指标（IoU、Dice系数等）
- 检测任务指标（mAP、精确率-召回率曲线等）
- 回归任务指标（MSE、MAE、R²等）
- 多类别指标分析和可视化
- 模型性能比较

**使用示例：**
```python
from tools.analyzers import MetricsAnalyzer

# 创建分析器
analyzer = MetricsAnalyzer()

# 分析分类任务
results = analyzer.analyze_predictions(
    y_true=true_labels, 
    y_pred=predictions, 
    task_type='classification'
)

# 生成混淆矩阵
analyzer.plot_confusion_matrix(y_true, y_pred)
```

### 3. DatasetAnalyzer - 数据集分析器

用于数据集特征分析、质量检查和统计信息生成。

**主要功能：**
- 数据集类别分布分析
- 数据质量检查与统计
- 数据集特征分析
- 数据平衡性评估
- 可视化与报告生成

**使用示例：**
```python
from tools.analyzers import DatasetAnalyzer

# 创建分析器
analyzer = DatasetAnalyzer(
    dataset_name='CIFAR10',
    dataset_config={'data_dir': './data'}
)

# 运行完整分析
results = analyzer.run_full_analysis(splits=['train', 'test'])
```

## 🚀 统一接口

### UnifiedAnalyzer - 统一分析器

整合三个核心分析器，提供一站式分析服务。

**使用示例：**
```python
from tools.analyzers import UnifiedAnalyzer

# 创建统一分析器
analyzer = UnifiedAnalyzer(
    model=your_model,
    dataset_name='CIFAR10',
    dataset_config={'data_dir': './data'}
)

# 运行综合分析
results = analyzer.run_comprehensive_analysis(
    input_data=sample_data,
    y_true=true_labels,
    y_pred=predictions,
    task_type='classification'
)
```

### 便捷函数

模块还提供了便捷的函数接口：

```python
from tools.analyzers import (
    analyze_model_attention,
    analyze_model_metrics,
    analyze_dataset,
    run_comprehensive_analysis
)

# 快速注意力分析
attention_results = analyze_model_attention(
    model=your_model,
    input_data=sample_data
)

# 快速指标分析
metrics_results = analyze_model_metrics(
    y_true=true_labels,
    y_pred=predictions,
    task_type='classification'
)

# 快速数据集分析
dataset_results = analyze_dataset(
    dataset_name='CIFAR10',
    dataset_config={'data_dir': './data'}
)

# 综合分析
comprehensive_results = run_comprehensive_analysis(
    model=your_model,
    dataset_name='CIFAR10',
    input_data=sample_data,
    y_true=true_labels,
    y_pred=predictions
)
```

## 📊 输出结果

所有分析器都会在指定的输出目录中生成以下内容：

### 文件结构
```
runs/analysis_output/
├── attention/              # 注意力分析结果
│   ├── attention_weights.png
│   ├── attention_patterns.json
│   └── attention_report.txt
├── metrics/                # 指标分析结果
│   ├── confusion_matrix.png
│   ├── metrics_summary.json
│   └── performance_report.txt
├── dataset/                # 数据集分析结果
│   ├── class_distribution.png
│   ├── data_quality.png
│   └── dataset_report.txt
├── comprehensive_analysis.json  # 综合分析结果
└── comprehensive_report.txt     # 综合分析报告
```

### 报告内容
- **JSON格式**：结构化的分析结果，便于程序处理
- **文本报告**：人类可读的分析摘要和建议
- **可视化图表**：直观的图表和图像
- **统计数据**：详细的数值统计信息

## ⚙️ 配置选项

### 通用配置
```python
# 输出目录配置
output_dir = "runs/my_analysis"

# 日志级别配置
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
```

### 分析器特定配置

**AttentionAnalyzer配置：**
```python
analyzer = AttentionAnalyzer(
    model=model,
    device='cuda',
    layer_names=['attention.0', 'attention.1'],
    head_fusion='mean',  # 'mean', 'max', 'min'
    output_dir='runs/attention_analysis'
)
```

**MetricsAnalyzer配置：**
```python
analyzer = MetricsAnalyzer(
    class_names=['cat', 'dog', 'bird'],
    average='weighted',  # 'micro', 'macro', 'weighted'
    output_dir='runs/metrics_analysis'
)
```

**DatasetAnalyzer配置：**
```python
analyzer = DatasetAnalyzer(
    dataset_name='CustomDataset',
    dataset_config={'batch_size': 32},
    label_extractor=custom_label_function,
    image_extractor=custom_image_function,
    output_dir='runs/dataset_analysis'
)
```

## 🔧 自定义扩展

### 自定义标签提取器
```python
def custom_label_extractor(sample):
    """自定义标签提取函数"""
    # 实现你的标签提取逻辑
    return extracted_label

analyzer = DatasetAnalyzer(
    dataset_name='MyDataset',
    label_extractor=custom_label_extractor
)
```

### 自定义指标计算
```python
def custom_metric(y_true, y_pred):
    """自定义指标计算函数"""
    # 实现你的指标计算逻辑
    return metric_value

analyzer = MetricsAnalyzer()
analyzer.add_custom_metric('my_metric', custom_metric)
```

## 📋 依赖要求

```python
# 核心依赖
torch>=1.8.0
numpy>=1.19.0
matplotlib>=3.3.0
seaborn>=0.11.0
pandas>=1.2.0
scikit-learn>=0.24.0

# NeuroTrain模块
src.dataset
src.metrics
src.utils
```

## 🚨 注意事项

1. **内存使用**：大型数据集分析可能消耗大量内存，建议设置合适的采样参数
2. **GPU支持**：注意力分析器支持GPU加速，确保CUDA环境正确配置
3. **文件权限**：确保输出目录具有写入权限
4. **数据格式**：确保输入数据格式与分析器期望的格式一致

## 🔍 故障排除

### 常见问题

**Q: 导入模块失败**
```python
# 确保NeuroTrain项目根目录在Python路径中
import sys
sys.path.append('/path/to/NeuroTrain')
```

**Q: 注意力分析失败**
```python
# 检查模型是否包含注意力层
print([name for name, module in model.named_modules() if 'attention' in name.lower()])
```

**Q: 数据集加载失败**
```python
# 检查数据集配置
try:
    from src.dataset import get_dataset
    dataset = get_dataset(**your_config)
    print(f"Dataset loaded successfully: {len(dataset)} samples")
except Exception as e:
    print(f"Dataset loading failed: {e}")
```

## 📚 更多示例

查看 `examples/` 目录获取更多详细的使用示例和最佳实践。

## 🤝 贡献指南

欢迎贡献代码和改进建议！请遵循以下步骤：

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request

## 📄 许可证

本模块遵循 NeuroTrain 项目的许可证协议。