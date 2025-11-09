# Utils与Tools 模块文档

## Utils模块

### 概述

Utils模块提供了各种辅助工具函数，包括损失函数、数据变换、早停、图像处理等。

## 损失函数 (criterion.py)

### 常用损失函数

```python
from src.utils.criterion import get_criterion

# 交叉熵损失
criterion = get_criterion('cross_entropy')

# BCE损失（带Logits）
criterion = get_criterion('bce_with_logits')

# Dice损失
criterion = get_criterion('dice')

# Focal损失
criterion = get_criterion('focal', alpha=0.25, gamma=2.0)
```

### 组合损失

```python
from src.utils.criterion import CombinedLoss

# Dice + BCE
combined_loss = CombinedLoss(
    losses=['dice', 'bce_with_logits'],
    weights=[0.5, 0.5]
)

loss = combined_loss(predictions, targets)
```

## 早停机制 (early_stopping.py)

```python
from src.utils import EarlyStopping

early_stopping = EarlyStopping(
    patience=10,        # 等待轮数
    min_delta=1e-4,     # 最小改善量
    mode='min'          # 'min' 或 'max'
)

for epoch in range(num_epochs):
    val_loss = validate()
    
    if early_stopping(val_loss):
        print("Early stopping triggered!")
        break
    
    if early_stopping.counter == 0:
        # 保存最佳模型
        save_model(model, 'best.pth')
```

## 图像处理 (image_utils.py)

```python
from src.utils.image_utils import (
    read_image,
    save_image,
    normalize_image,
    denormalize_image,
    resize_image
)

# 读取图像
image = read_image('path/to/image.jpg')

# 标准化
normalized = normalize_image(image, mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])

# 反标准化
denorm = denormalize_image(normalized, mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])

# 调整大小
resized = resize_image(image, size=(512, 512))

# 保存图像
save_image(resized, 'output.jpg')
```

## 后处理 (postprocess.py)

```python
from src.utils.postprocess import (
    threshold,
    morphology_ops,
    remove_small_objects,
    fill_holes
)

# 阈值化
binary = threshold(predictions, threshold=0.5)

# 形态学操作
opened = morphology_ops(binary, operation='opening', kernel_size=3)
closed = morphology_ops(binary, operation='closing', kernel_size=3)

# 移除小对象
cleaned = remove_small_objects(binary, min_size=100)

# 填充孔洞
filled = fill_holes(cleaned)
```

## 计时器 (timer.py)

```python
from src.utils import Timer

# 使用上下文管理器
with Timer("Data loading"):
    data = load_data()

# 手动计时
timer = Timer()
timer.start()
# ... 执行操作 ...
elapsed = timer.stop()
print(f"Elapsed: {elapsed:.2f}s")
```

## 分布式训练工具 (ddp_utils.py)

```python
from src.utils.ddp_utils import (
    setup_ddp,
    cleanup_ddp,
    reduce_tensor,
    gather_tensors
)

# 初始化DDP
setup_ddp(rank, world_size)

try:
    # 训练代码
    loss = train_step()
    
    # 聚合损失
    avg_loss = reduce_tensor(loss, world_size)
    
finally:
    cleanup_ddp()
```

---

## Tools模块

### 概述

Tools模块提供了各种分析和转换工具，包括配置转换、ONNX导出、量化、数据分析等。

## 配置转换 (config_converter.py)

```python
from tools.config_converter import convert_config

# JSON转TOML
convert_config('config.json', 'config.toml', target_format='toml')

# TOML转YAML
convert_config('config.toml', 'config.yaml', target_format='yaml')

# YAML转JSON
convert_config('config.yaml', 'config.json', target_format='json')
```

## ONNX导出 (onnx_export.py)

```python
from tools.onnx_export import export_model_to_onnx

# 导出模型
export_model_to_onnx(
    model=model,
    input_size=(3, 224, 224),
    output_path='model.onnx',
    opset_version=11,
    dynamic_axes={'input': {0: 'batch_size'}}
)
```

## 量化工具 (quantization_cli.py)

```bash
# 量化模型
python tools/quantization_cli.py quantize \
    model.pt output/ \
    --method dynamic \
    --dtype qint8

# 分析量化效果
python tools/quantization_cli.py analyze \
    original.pt quantized.pt analysis/

# 运行示例
python tools/quantization_cli.py example --method static
```

## LoRA合并 (lora_merge.py)

```bash
# 基本合并
python tools/lora_merge.py \
    --base THUDM/chatglm3-6b \
    --adapters ./lora_adapter \
    --output ./merged_model

# 加权合并
python tools/lora_merge.py \
    --base model \
    --adapters adapter1 adapter2 \
    --output merged \
    --merge-strategy weighted \
    --weights 0.7 0.3
```

## 数据分析器

### Metrics分析器 (analyzers/metrics_analyzer.py)

```python
from tools.analyzers.metrics_analyzer import MetricsAnalyzer

analyzer = MetricsAnalyzer('runs/experiment_001')

# 加载指标
metrics = analyzer.load_metrics()

# 生成报告
report = analyzer.generate_report()
analyzer.save_report(report, 'analysis_report.md')

# 可视化
analyzer.plot_training_curves()
analyzer.plot_metrics_comparison()
```

### 数据集分析器 (analyzers/dataset_analyzer.py)

```python
from tools.analyzers.dataset_analyzer import DatasetAnalyzer

analyzer = DatasetAnalyzer(dataset)

# 统计信息
stats = analyzer.compute_statistics()
print(f"Mean: {stats['mean']}")
print(f"Std: {stats['std']}")

# 类别分布
dist = analyzer.class_distribution()
analyzer.plot_distribution(dist)

# 数据质量检查
issues = analyzer.check_quality()
print(f"Found {len(issues)} issues")
```

### 注意力分析器 (analyzers/attention_analyzer.py)

```python
from tools.analyzers.attention_analyzer import AttentionAnalyzer

analyzer = AttentionAnalyzer(model)

# 提取注意力权重
attention_weights = analyzer.extract_attention(images)

# 可视化注意力
analyzer.visualize_attention(
    attention_weights,
    save_path='attention.png'
)

# 分析注意力模式
patterns = analyzer.analyze_patterns(attention_weights)
```

### Mask分析器 (analyzers/mask_analyzer.py)

```python
from tools.analyzers.mask_analyzer import MaskAnalyzer

analyzer = MaskAnalyzer()

# 分析预测掩码
analysis = analyzer.analyze_mask(predicted_mask, ground_truth)

print(f"Coverage: {analysis['coverage']:.2%}")
print(f"Fragmentation: {analysis['fragmentation']:.2f}")
print(f"Boundary smoothness: {analysis['smoothness']:.2f}")

# 可视化对比
analyzer.visualize_comparison(predicted_mask, ground_truth)
```

### LoRA分析器 (analyzers/lora_analyzer.py)

```python
from tools.analyzers.lora_analyzer import LoRAAnalyzer

analyzer = LoRAAnalyzer(model_with_lora)

# 分析LoRA参数
lora_info = analyzer.analyze_lora_parameters()
print(f"LoRA rank: {lora_info['rank']}")
print(f"Trainable params: {lora_info['trainable_params']:,}")
print(f"Percentage: {lora_info['percentage']:.2%}")

# 可视化权重分布
analyzer.plot_weight_distribution()

# 比较LoRA效果
comparison = analyzer.compare_with_full_finetuning(
    base_model, lora_model, full_ft_model
)
```

## 数据格式转换 (to_parquet.py)

```python
from tools.to_parquet import convert_to_parquet

# CSV转Parquet
convert_to_parquet(
    input_path='data.csv',
    output_path='data.parquet',
    compression='snappy'
)

# JSON转Parquet
convert_to_parquet(
    input_path='data.json',
    output_path='data.parquet',
    format='json'
)
```

## 配置检查器 (checker.py)

```python
from tools.checker import ConfigChecker

checker = ConfigChecker('config.toml')

# 检查配置
issues = checker.check()

if issues:
    print("Configuration issues found:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("Configuration is valid!")

# 验证必需字段
required_fields = ['model.name', 'dataset.root_dir', 'training.epochs']
checker.validate_required_fields(required_fields)
```

## 清理工具 (cleanup.py)

```bash
# 清理临时文件
python tools/cleanup.py --target runs/ --pattern "*.tmp"

# 清理旧检查点
python tools/cleanup.py --clean-checkpoints --keep-last 3

# 清理缓存
python tools/cleanup.py --clean-cache
```

## 数据集列表 (list_datasets.py)

```bash
# 列出所有可用数据集
python tools/list_datasets.py

# 列出特定类型的数据集
python tools/list_datasets.py --type medical

# 显示数据集详情
python tools/list_datasets.py --dataset drive --verbose
```

## 使用示例

### 完整工作流程

```bash
# 1. 检查配置
python tools/checker.py configs/my_config.toml

# 2. 列出可用数据集
python tools/list_datasets.py --type segmentation

# 3. 训练模型
python main.py --config configs/my_config.toml --train

# 4. 分析训练结果
python tools/analyzers/metrics_analyzer.py --run_id experiment_001

# 5. 导出模型
python tools/onnx_export.py \
    --model runs/experiment_001/checkpoints/best.pth \
    --output model.onnx

# 6. 量化模型
python tools/quantization_cli.py quantize \
    runs/experiment_001/checkpoints/best.pth \
    quantized/ \
    --method dynamic

# 7. 清理临时文件
python tools/cleanup.py --target runs/experiment_001 --pattern "*.tmp"
```

### 批量处理

```python
from pathlib import Path
from tools.onnx_export import export_model_to_onnx

# 批量导出模型
model_dir = Path('runs/')
for model_path in model_dir.glob('**/best.pth'):
    output_path = model_path.parent / 'model.onnx'
    export_model_to_onnx(
        model_path=model_path,
        output_path=output_path
    )
    print(f"Exported: {output_path}")
```

## 最佳实践

1. **配置管理**: 使用配置检查器验证配置文件
2. **定期分析**: 训练后使用分析器检查结果
3. **模型导出**: 训练完成后导出为ONNX格式
4. **量化优化**: 部署前进行模型量化
5. **清理维护**: 定期清理临时文件和旧检查点

## 参考资料

- [ONNX文档](https://onnx.ai/)
- [PyTorch量化](https://pytorch.org/docs/stable/quantization.html)
- [LoRA论文](https://arxiv.org/abs/2106.09685)

---

更多工具和使用方法请查看 `tools/README.md` 和 `examples/` 目录。

