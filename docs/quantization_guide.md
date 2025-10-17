# NeuroTrain量化工具文档

## 概述

NeuroTrain量化工具是一个完整的模型量化解决方案，支持多种量化方法，包括动态量化、静态量化、量化感知训练(QAT)、GPTQ、AWQ和BitsAndBytes量化。该工具旨在帮助用户减少模型大小、提高推理速度，同时保持模型性能。

## 功能特性

### 支持的量化方法

1. **动态量化 (Dynamic Quantization)**
   - PyTorch内置方法
   - 无需校准数据
   - 推理时动态量化权重
   - 适用于大多数模型

2. **静态量化 (Static Quantization)**
   - PyTorch内置方法
   - 需要校准数据集
   - 预计算量化参数
   - 更好的压缩效果

3. **量化感知训练 (QAT)**
   - PyTorch内置方法
   - 训练时考虑量化影响
   - 最佳量化效果
   - 需要重新训练

4. **GPTQ量化**
   - 基于GPTQ算法
   - 需要auto-gptq库
   - 适用于大语言模型
   - 4bit量化支持

5. **AWQ量化**
   - 基于AWQ算法
   - 需要awq库
   - 适用于大语言模型
   - 保持激活精度

6. **BitsAndBytes量化**
   - 4bit和8bit量化
   - 需要bitsandbytes库
   - 支持训练和推理
   - 内存效率高

## 安装依赖

### 基础依赖
```bash
pip install torch>=1.8.0
```

### 可选依赖
```bash
# GPTQ量化
pip install auto-gptq>=0.4.0

# AWQ量化
pip install awq>=0.1.0

# BitsAndBytes量化
pip install bitsandbytes>=0.39.0
```

## 快速开始

### 1. 基础量化使用

```python
from src.utils.quantization import QuantizationConfig, QuantizationManager

# 创建量化配置
config = QuantizationConfig(method="dynamic", dtype="qint8")

# 创建量化管理器
manager = QuantizationManager(config)

# 量化模型
quantized_model = manager.quantize_model(your_model)

# 获取模型信息
size_info = manager.get_model_size_info(quantized_model)
print(f"模型大小: {size_info['model_size_mb']:.2f}MB")
```

### 2. 配置文件方式

```yaml
# config.yaml
quantization:
  enabled: true
  method: "dynamic"
  dtype: "qint8"
  device: "auto"
```

```python
from src.utils.quantization_config import setup_quantization_from_config

# 基于配置量化模型
quantized_model = setup_quantization_from_config(model, config)
```

### 3. 命令行工具

```bash
# 量化模型
python tools/quantization_cli.py quantize model.pt output/ --method dynamic

# 分析量化效果
python tools/quantization_cli.py analyze original.pt quantized.pt analysis/

# 列出可用方法
python tools/quantization_cli.py list

# 运行示例
python tools/quantization_cli.py example --method dynamic
```

## 详细使用指南

### 量化配置

#### QuantizationConfig参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| method | str | "dynamic" | 量化方法 |
| dtype | str | "qint8" | 量化数据类型 |
| device | str | "auto" | 设备 |
| trust_remote_code | bool | False | 是否信任远程代码 |
| bits | int | 4 | GPTQ量化位数 |
| group_size | int | 128 | GPTQ分组大小 |
| desc_act | bool | False | GPTQ是否使用desc_act |
| load_in_4bit | bool | False | BitsAndBytes 4bit量化 |
| load_in_8bit | bool | False | BitsAndBytes 8bit量化 |

#### 预定义配置

```python
from src.utils.quantization import (
    DYNAMIC_QUANTIZATION_CONFIG,
    STATIC_QUANTIZATION_CONFIG,
    QAT_CONFIG,
    GPTQ_4BIT_CONFIG,
    BITSANDBYTES_4BIT_CONFIG
)

# 使用预定义配置
quantized_model = manager.quantize_model(model, DYNAMIC_QUANTIZATION_CONFIG)
```

### 量化感知训练

```python
from src.utils.quantization_trainer import QuantizationTrainer

# 创建QAT配置
qat_config = QuantizationConfig(method="qat")

# 创建量化训练器
trainer = QuantizationTrainer(
    model=model,
    quantization_config=qat_config,
    output_dir="outputs/qat"
)

# 设置量化
quantized_model = trainer.setup_quantization()

# 进行量化感知训练
trainer.train_with_quantization(
    train_loader=train_loader,
    valid_loader=valid_loader,
    num_epochs=10,
    optimizer=optimizer,
    criterion=criterion
)
```

### 量化效果分析

```python
from src.utils.quantization_trainer import QuantizationAnalyzer

# 创建分析器
analyzer = QuantizationAnalyzer(original_model, quantized_model)

# 比较模型大小
size_comparison = analyzer.compare_model_sizes()
print(f"压缩比: {size_comparison['compression_ratio']:.2f}x")

# 比较推理速度
speed_comparison = analyzer.compare_inference_speed(test_input)
print(f"加速比: {speed_comparison['speedup']:.2f}x")

# 比较准确率
accuracy_comparison = analyzer.compare_accuracy(test_loader)
print(f"准确率下降: {accuracy_comparison['accuracy_drop']:.4f}")

# 生成完整报告
report = analyzer.generate_report(
    test_loader=test_loader,
    test_input=test_input,
    output_path="analysis_output"
)
```

## 配置示例

### 动态量化配置

```yaml
quantization:
  enabled: true
  method: "dynamic"
  dtype: "qint8"
```

### 静态量化配置

```yaml
quantization:
  enabled: true
  method: "static"
  dtype: "qint8"
  num_calibration_samples: 100
```

### GPTQ量化配置

```yaml
quantization:
  enabled: true
  method: "gptq"
  bits: 4
  group_size: 128
  desc_act: false
```

### BitsAndBytes量化配置

```yaml
quantization:
  enabled: true
  method: "bnb_4bit"
  load_in_4bit: true
  bnb_4bit_compute_dtype: "float16"
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_use_double_quant: true
```

## 最佳实践

### 1. 选择合适的量化方法

- **CNN模型**: 推荐使用静态量化或QAT
- **Transformer模型**: 推荐使用动态量化或QAT
- **大语言模型**: 推荐使用GPTQ、AWQ或BitsAndBytes
- **快速原型**: 推荐使用动态量化

### 2. 量化策略

```python
from src.utils.quantization_config import get_recommended_quantization_method

# 获取推荐方法
method = get_recommended_quantization_method("cnn", "inference")
print(f"推荐方法: {method}")
```

### 3. 依赖检查

```python
from src.utils.quantization_config import check_quantization_dependencies

# 检查依赖
if check_quantization_dependencies("gptq"):
    print("GPTQ量化可用")
else:
    print("GPTQ量化不可用，请安装auto-gptq")
```

### 4. 模型保存和加载

```python
# 保存量化模型
manager.save_quantized_model(quantized_model, "outputs/quantized_model")

# 加载量化模型
loaded_model = manager.load_quantized_model("outputs/quantized_model")
```

## 性能对比

### 典型压缩效果

| 量化方法 | 压缩比 | 速度提升 | 精度损失 |
|----------|--------|----------|----------|
| 动态量化 | 2-4x | 1.5-2x | <1% |
| 静态量化 | 3-5x | 2-3x | 1-3% |
| QAT | 4-6x | 2-4x | <1% |
| GPTQ | 4-8x | 2-4x | 1-5% |
| BitsAndBytes | 4-8x | 2-4x | 1-3% |

### 内存使用

量化可以显著减少内存使用：

- **模型权重**: 减少50-75%
- **激活值**: 减少25-50%
- **总体内存**: 减少30-60%

## 故障排除

### 常见问题

1. **ImportError: auto-gptq not available**
   ```bash
   pip install auto-gptq>=0.4.0
   ```

2. **ImportError: awq not available**
   ```bash
   pip install awq>=0.1.0
   ```

3. **ImportError: bitsandbytes not available**
   ```bash
   pip install bitsandbytes>=0.39.0
   ```

4. **CUDA out of memory**
   - 减少batch size
   - 使用CPU量化
   - 使用梯度检查点

5. **量化后精度下降严重**
   - 尝试QAT方法
   - 调整量化参数
   - 使用校准数据

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查量化信息
print(manager.quantization_info)

# 验证模型输出
original_output = model(test_input)
quantized_output = quantized_model(test_input)
print(f"输出差异: {torch.norm(original_output - quantized_output)}")
```

## API参考

### QuantizationConfig

量化配置类，用于设置量化参数。

#### 构造函数
```python
QuantizationConfig(
    method: str = "dynamic",
    dtype: str = "qint8",
    device: str = "auto",
    **kwargs
)
```

### QuantizationManager

量化管理器，负责执行量化操作。

#### 主要方法
- `quantize_model(model)`: 量化模型
- `save_quantized_model(model, path)`: 保存量化模型
- `load_quantized_model(path)`: 加载量化模型
- `get_model_size_info(model)`: 获取模型大小信息

### QuantizationTrainer

量化训练器，支持量化感知训练。

#### 主要方法
- `setup_quantization()`: 设置量化
- `train_with_quantization()`: 进行量化感知训练

### QuantizationAnalyzer

量化分析器，用于分析量化效果。

#### 主要方法
- `compare_model_sizes()`: 比较模型大小
- `compare_inference_speed()`: 比较推理速度
- `compare_accuracy()`: 比较准确率
- `generate_report()`: 生成分析报告

## 示例代码

完整的使用示例请参考：
- `examples/quantization_examples.py` - Python代码示例
- `examples/quantization_demo.ipynb` - Jupyter notebook示例
- `configs/quantization_example.yaml` - 配置文件示例

## 更新日志

### v1.0.0
- 初始版本发布
- 支持动态量化、静态量化、QAT
- 支持GPTQ、AWQ、BitsAndBytes量化
- 提供完整的分析工具
- 集成到NeuroTrain框架

## 贡献指南

欢迎贡献代码和建议！请参考项目的贡献指南。

## 许可证

本项目采用MIT许可证。
