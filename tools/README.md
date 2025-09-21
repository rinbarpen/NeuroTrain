# 工具使用说明

本目录包含了NeuroTrain项目的各种工具和脚本，用于模型分析、可视化和导出。

## 工具列表

### 1. 分析工具

#### `analyzer.py` - 模型性能分析器
- **功能**: 分析模型参数、性能指标，生成对比报告
- **主要特性**:
  - 实现 `_comparative_score_table` 函数，生成跨模型、跨任务的性能对比表
  - 支持Markdown和PDF报告导出
  - 自动分析最佳性能模型

**使用示例**:
```python
from tools.analyzer import Analyzer
from pathlib import Path

# 创建分析器
analyzer = Analyzer(Path("results"))

# 性能数据示例
scores_map = {
    'model1': {
        'task1': {
            'accuracy': {'score': 0.95, 'up': True},
            'f1_score': {'score': 0.92, 'up': True}
        }
    },
    'model2': {
        'task1': {
            'accuracy': {'score': 0.93, 'up': True},
            'f1_score': {'score': 0.90, 'up': True}
        }
    }
}

# 生成对比表
table = analyzer._comparative_score_table(scores_map)
print(table)

# 生成完整报告
report = analyzer.generate_comparative_report(scores_map, Path("report.md"))

# 导出PDF（需要安装reportlab）
analyzer.export_to_pdf(scores_map, Path("report.pdf"))
```

#### `onnx_export.py` - ONNX模型导出器
- **功能**: 将PyTorch模型导出为ONNX格式
- **主要特性**:
  - 支持多输入模型
  - 自动验证导出的ONNX模型
  - 提供详细的模型信息

**使用示例**:
```python
from tools.onnx_export import OnnxExport, export_model_to_onnx
import torch
import torch.nn as nn

# 创建示例模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.linear(x)

model = SimpleModel()

# 方法1: 使用类
exporter = OnnxExport(model, input_sizes=(10,))
exporter.save(Path("model.onnx"))

# 方法2: 使用便捷函数
export_model_to_onnx(model, (10,), Path("model.onnx"))
```

### 2. 可视化工具

#### `advanced_visualizer.py` - 高级可视化工具
- **功能**: 提供多种高级可视化功能
- **主要特性**:
  - 注意力权重热力图
  - 特征图可视化
  - 训练曲线绘制
  - 混淆矩阵可视化
  - 交互式仪表板（需要Plotly）
  - Streamlit应用支持

**使用示例**:
```python
from tools.advanced_visualizer import AdvancedVisualizer
import numpy as np

# 创建可视化器
visualizer = AdvancedVisualizer(Path("output/visualizations"))

# 绘制训练曲线
train_losses = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2]
valid_losses = [1.1, 0.9, 0.7, 0.5, 0.4, 0.3]

visualizer.plot_training_curves(
    train_losses=train_losses,
    valid_losses=valid_losses,
    save_path=Path("training_curves.png")
)

# 绘制注意力热力图
attention_weights = np.random.rand(10, 10)
tokens = [f"token_{i}" for i in range(10)]

visualizer.plot_attention_heatmap(
    attention_weights, tokens=tokens,
    save_path=Path("attention_heatmap.png")
)
```

#### `attention_visualizer.py` - 注意力机制可视化
- **功能**: 专门用于可视化Transformer模型的注意力机制
- **主要特性**:
  - 多头注意力可视化
  - 注意力流分析
  - 注意力模式分析
  - 交互式注意力图

**使用示例**:
```python
from tools.attention_visualizer import AttentionVisualizer, visualize_model_attention
import torch

# 创建注意力可视化器
visualizer = AttentionVisualizer(Path("output/attention"))

# 示例注意力权重
attention_weights = torch.randn(8, 10, 10)  # 8个头，10x10的注意力矩阵
tokens = [f"token_{i}" for i in range(10)]

# 可视化多头注意力
visualizer.visualize_attention_heads(
    attention_weights, tokens=tokens,
    save_path=Path("attention_heads.png")
)

# 可视化注意力流
visualizer.visualize_attention_flow(
    attention_weights, tokens=tokens,
    save_path=Path("attention_flow.png")
)
```

#### `error_analyzer.py` - 错误样本分析工具
- **功能**: 分析模型预测错误的样本，找出错误模式
- **主要特性**:
  - 错误样本可视化
  - 特征空间分析
  - 置信度分布分析
  - 交互式错误分析仪表板

**使用示例**:
```python
from tools.error_analyzer import ErrorAnalyzer, analyze_model_errors
import numpy as np

# 创建错误分析器
analyzer = ErrorAnalyzer(Path("output/error_analysis"))

# 示例数据
y_true = np.array([0, 1, 2, 0, 1, 2])
y_pred = np.array([0, 1, 0, 0, 2, 2])  # 包含一些错误
y_prob = np.random.dirichlet([1, 1, 1], 6)

# 分析错误
analysis = analyzer.analyze_prediction_errors(
    y_true, y_pred, y_prob,
    class_names=['Class A', 'Class B', 'Class C']
)

# 可视化错误分布
analyzer.visualize_error_distribution(
    y_true, y_pred, ['Class A', 'Class B', 'Class C'],
    save_path=Path("error_distribution.png")
)
```

### 3. 命令行集成

#### ONNX导出集成
在训练或测试时自动导出ONNX模型：

```bash
# 训练时导出ONNX
python main.py --train --export-onnx --onnx-path model.onnx

# 测试时导出ONNX
python main.py --test --export-onnx --onnx-opset 11

# 详细导出信息
python main.py --train --export-onnx --onnx-verbose
```

## 依赖安装

### 必需依赖
```bash
pip install matplotlib seaborn pandas scikit-learn
```

### 可选依赖
```bash
# PDF导出支持
pip install reportlab

# 交互式可视化支持
pip install plotly

# Streamlit应用支持
pip install streamlit

# ONNX验证支持
pip install onnx

# 高级可视化支持
pip install opencv-python pillow
```

## 使用建议

1. **性能分析**: 使用 `analyzer.py` 进行模型性能对比和报告生成
2. **模型部署**: 使用 `onnx_export.py` 将模型导出为ONNX格式
3. **可视化分析**: 使用各种可视化工具深入分析模型行为
4. **错误调试**: 使用 `error_analyzer.py` 找出模型的问题所在

## 注意事项

1. 某些功能需要额外的依赖库，请根据需要进行安装
2. 交互式功能需要相应的库支持
3. 大模型的可视化可能需要较多内存
4. 建议在Jupyter Notebook中使用这些工具以获得更好的交互体验