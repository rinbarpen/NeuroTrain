# NeuroTrain量化模块

## 模块结构

```
src/quantization/
├── __init__.py          # 模块入口，导出主要API
├── core.py              # 核心量化功能（QuantizationConfig, QuantizationManager）
├── trainer.py            # 量化训练器（QuantizationTrainer, QuantizationAnalyzer）
└── config.py             # 配置管理（配置模板、验证、依赖检查）
```

## 支持的平台

- **PyTorch**: 动态量化、静态量化、QAT
- **ONNX**: 模型导出、静态量化
- **TensorRT**: FP16/INT8量化（需要额外安装TensorRT）

## 快速使用

### 1. 导入模块

```python
from src.quantization import (
    QuantizationConfig, 
    QuantizationManager, 
    QuantizationTrainer,
    QuantizationAnalyzer
)
```

### 2. PyTorch量化

```python
# 动态量化
config = QuantizationConfig(platform="pytorch", method="dynamic")
manager = QuantizationManager(config)
quantized_model = manager.quantize_model(model)

# 静态量化
config = QuantizationConfig(platform="pytorch", method="static", calibration_dataset=data)
manager = QuantizationManager(config)
quantized_model = manager.quantize_model(model)

# QAT量化感知训练
config = QuantizationConfig(platform="pytorch", method="qat")
trainer = QuantizationTrainer(model, config, "outputs/qat")
quantized_model = trainer.setup_quantization()
```

### 3. ONNX导出和量化

```python
# ONNX导出
config = QuantizationConfig(platform="onnx", method="static")
manager = QuantizationManager(config)
onnx_path = manager.quantize_model(model, input_shape=(1, 3, 224, 224))

# 或使用训练器导出
trainer = QuantizationTrainer(model, config, "outputs/onnx")
onnx_path = trainer.export_to_onnx(input_shape=(1, 3, 224, 224))
```

### 4. TensorRT量化

```python
# TensorRT引擎
config = QuantizationConfig(platform="tensorrt", tensorrt_precision="fp16")
manager = QuantizationManager(config)
engine = manager.quantize_model(model, input_shape=(1, 3, 224, 224))

# 或使用训练器导出
trainer = QuantizationTrainer(model, config, "outputs/tensorrt")
engine_path = trainer.export_to_tensorrt(input_shape=(1, 3, 224, 224))
```

### 5. 量化效果分析

```python
# 创建分析器
analyzer = QuantizationAnalyzer(original_model, quantized_model)

# 比较模型大小
size_comparison = analyzer.compare_model_sizes()
print(f"压缩比: {size_comparison['compression_ratio']:.2f}x")

# 比较推理速度
speed_comparison = analyzer.compare_inference_speed(test_input)
print(f"加速比: {speed_comparison['speedup']:.2f}x")

# 生成完整报告
report = analyzer.generate_report(test_loader, test_input, "analysis_output")
```

## 命令行工具

```bash
# 列出可用平台和方法
python tools/quantization_cli.py list

# PyTorch动态量化
python tools/quantization_cli.py quantize model.pt output/ --platform pytorch --method dynamic

# ONNX导出和量化
python tools/quantization_cli.py quantize model.pt output/ --platform onnx --onnx-mode static --input-shape 1 3 224 224

# TensorRT量化
python tools/quantization_cli.py quantize model.pt output/ --platform tensorrt --trt-precision fp16 --input-shape 1 3 224 224

# 分析量化效果
python tools/quantization_cli.py analyze original.pt quantized.pt analysis/
```

## 配置示例

### PyTorch配置

```yaml
quantization:
  enabled: true
  platform: "pytorch"
  method: "dynamic"  # dynamic, static, qat
  dtype: "qint8"
  device: "cpu"
```

### ONNX配置

```yaml
quantization:
  enabled: true
  platform: "onnx"
  method: "static"
  onnx_opset_version: 11
  onnx_quantization_mode: "static"
```

### TensorRT配置

```yaml
quantization:
  enabled: true
  platform: "tensorrt"
  tensorrt_precision: "fp16"  # fp16, int8
  tensorrt_workspace_size: 1073741824  # 1GB
```

## 依赖要求

### 基础依赖
- PyTorch >= 1.8.0

### 可选依赖
- ONNX: `pip install onnx onnxruntime`
- TensorRT: 需要从NVIDIA官网下载安装

## 示例文件

- `examples/quantization_examples.py` - 完整使用示例
- `examples/ptq_qat_examples.py` - PTQ和QAT专门示例
- `examples/quantization_demo.ipynb` - Jupyter notebook示例
- `tools/quantization_cli.py` - 命令行工具
