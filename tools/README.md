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

### 3. 数据处理工具

#### `data_to_latex.py` - 数据文件转LaTeX格式工具
- **功能**: 将各种数据文件（CSV、Excel、JSON等）转换为LaTeX格式
- **主要特性**:
  - 支持多种数据格式：CSV、Excel、JSON、Parquet、TSV
  - 支持多种LaTeX格式：table、longtable、itemize、enumerate、description
  - 自动转义LaTeX特殊字符
  - 支持列选择和自定义模板
  - 支持行数限制和数据预览

**命令行使用**:
```bash
# 转换CSV为标准表格
python tools/data_to_latex.py -i data.csv -t table --caption "实验结果" --label "tab:results"

# 转换为无序列表（自定义模板）
python tools/data_to_latex.py -i data.xlsx -t itemize --template "{name}: {value}"

# 转换为描述列表
python tools/data_to_latex.py -i data.json -t description --key-column name

# 只选择特定列并保存
python tools/data_to_latex.py -i data.csv -o output.tex -t table -c col1 col2 col3

# 长表格（支持跨页）
python tools/data_to_latex.py -i data.csv -t longtable --max-rows 100

# 查看数据信息
python tools/data_to_latex.py -i data.csv --show-info
```

**支持的LaTeX格式**:
- `table`: 标准表格 (tabular)
- `longtable`: 长表格（支持跨页，需要 `\usepackage{longtable}`）
- `itemize`: 无序列表
- `enumerate`: 有序列表
- `description`: 描述列表

**Python使用**:
```python
from tools.data_to_latex import DataToLatexConverter

# 创建转换器
converter = DataToLatexConverter(
    input_file='data.csv',
    output_file='output.tex',
    latex_type='table',
    columns=['name', 'score'],
    caption='Results',
    label='tab:results'
)

# 加载数据
converter.load_data()

# 执行转换
latex_code = converter.convert()

# 保存
converter.save(latex_code)
```

更多示例请查看 `tools/data_to_latex_examples.md`

#### `to_parquet.py` - 数据文件转Parquet格式工具
- **功能**: 将CSV、Excel等文件转换为高效的Parquet格式
- **主要特性**:
  - 支持单文件和批量转换
  - 自动识别文件类型

```bash
# 转换单个文件
python tools/to_parquet.py -i data.csv

# 批量转换目录
python tools/to_parquet.py -i data/
```

#### `list_datasets.py` - 数据集查询工具
- **功能**: 查询和浏览项目中支持的数据集
- **主要特性**:
  - 列出所有可用数据集
  - 按任务类型查询数据集
  - 查看数据集详细信息
  - 显示统计信息

```bash
# 列出所有数据集
python tools/list_datasets.py --list-all

# 列出所有任务类型
python tools/list_datasets.py --list-tasks

# 按任务查询
python tools/list_datasets.py --task classification

# 查看数据集信息
python tools/list_datasets.py --info imagenet

# 显示统计信息
python tools/list_datasets.py --statistics
```

### 4. 命令行集成

#### LoRA 适配器合并工具
支持多种合并策略的 LoRA 适配器合并工具，可将一个或多个 LoRA 适配器合并回基础模型。

```bash
# 基本顺序合并
python tools/lora_merge.py \
  --base THUDM/chatglm3-6b \
  --adapters ./outputs/run1-lora ./outputs/run2-lora \
  --output ./merged_model \
  --merge-dtype float16 \
  --trust-remote-code \
  --save-tokenizer \
  --use-proxy

# 加权合并
python tools/lora_merge.py \
  --base THUDM/chatglm3-6b \
  --adapters ./outputs/run1-lora ./outputs/run2-lora \
  --output ./merged_model \
  --merge-strategy weighted \
  --weights 0.7 0.3 \
  --merge-dtype float16

# 平均合并
python tools/lora_merge.py \
  --base THUDM/chatglm3-6b \
  --adapters ./outputs/run1-lora ./outputs/run2-lora \
  --output ./merged_model \
  --merge-strategy average

# 仅使用本地缓存
python tools/lora_merge.py \
  --base mistralai/Mistral-7B-v0.1 \
  --adapters ./lora_out \
  --output ./merged_mistral \
  --local-files-only
```

参数说明：
- `--base`: 基础模型路径或 Hugging Face 模型 ID
- `--adapters`: 一个或多个 LoRA 适配器目录
- `--output`: 合并后模型保存目录
- `--merge-strategy`: 合并策略（`sequential|weighted|average`）
- `--weights`: 权重列表（用于加权合并）
- `--merge-dtype`: 合并时 dtype（`auto|float16|bfloat16|float32`）
- `--device-map`: 设备映射（默认 `auto`）
- `--trust-remote-code`: 允许加载自定义代码（多模态模型常用）
- `--local-files-only`: 仅本地加载，禁止联网
- `--no-safetensors`: 不使用 safetensors 保存
- `--save-tokenizer`: 如可用则同时保存 tokenizer
- `--no-report`: 禁用生成合并报告
- `--use-proxy`: 下载前调用 `proxy_on` 开启代理（按用户环境约定）

合并策略说明：
- **sequential**: 顺序合并，按提供顺序依次合并适配器
- **weighted**: 加权合并，使用指定权重合并多个适配器
- **average**: 平均合并，使用相等权重合并所有适配器

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
