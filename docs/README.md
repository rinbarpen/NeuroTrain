# NeuroTrain 文档索引

欢迎来到NeuroTrain文档中心！本页面提供了所有文档的索引和导航。

## 📚 核心文档

### 入门指南
- **[项目架构文档](ARCHITECTURE.md)** - 了解NeuroTrain的整体架构和设计理念
- **[快速开始](../README.md)** - 快速开始使用NeuroTrain

### 模块文档
- **[Dataset模块](DATASET_MODULE.md)** - 数据集加载、增强和管理
- **[Models模块](MODELS_MODULE.md)** - 模型创建、定制和管理
- **[Engine模块](ENGINE_MODULE.md)** - 训练、测试和推理引擎
- **[Metrics模块](METRICS_MODULE.md)** - 各种评估指标
- **[Utils & Tools模块](UTILS_AND_TOOLS.md)** - 工具函数和分析工具

## 🎯 按任务类型查找

### 图像分割
- Medical Image Segmentation
  - DRIVE数据集使用
  - UNet模型训练
  - Dice和IoU指标

### 图像分类
- CIFAR-10/100示例
- ImageNet迁移学习
- ResNet系列模型

### 目标检测
- COCO数据集
- mAP指标计算

### 多模态学习
- CLIP模型使用
- 图像-文本检索

## 📖 示例代码

### Python示例
位于 `examples/` 目录：

1. **dataset_basic_example.py** - 数据集基础使用
2. **models_basic_example.py** - 模型创建和使用
3. **complete_training_example.py** - 完整训练流程

### Jupyter Notebook
位于 `examples/` 目录：

1. **dataset_tutorial.ipynb** - Dataset模块完整教程
2. 更多notebook持续更新中...

## 🔧 工具使用

### 命令行工具
```bash
# 配置检查
python tools/checker.py config.toml

# ONNX导出
python tools/onnx_export.py model.pt output.onnx

# 模型量化
python tools/quantization_cli.py quantize model.pt output/

# LoRA合并
python tools/lora_merge.py --base model --adapters lora_adapter --output merged
```

### 分析工具
```bash
# 指标分析
python tools/analyzers/metrics_analyzer.py --run_id experiment_001

# 数据集分析
python tools/analyzers/dataset_analyzer.py --dataset cifar10

# 注意力可视化
python tools/analyzers/attention_analyzer.py --model model.pt
```

## 📊 工作流程

### 典型训练流程

```
1. 准备数据
   ├── 下载数据集
   ├── 配置数据增强
   └── 创建DataLoader

2. 创建模型
   ├── 选择模型架构
   ├── 加载预训练权重
   └── 定制模型结构

3. 配置训练
   ├── 设置超参数
   ├── 选择损失函数
   └── 配置优化器

4. 开始训练
   ├── 训练循环
   ├── 验证
   └── 保存检查点

5. 测试评估
   ├── 加载最佳模型
   ├── 计算指标
   └── 生成报告

6. 模型部署
   ├── ONNX导出
   ├── 模型量化
   └── 性能测试
```

## 🚀 快速链接

### 常见任务

- [如何加载CIFAR-10数据集？](DATASET_MODULE.md#cifar-10)
- [如何使用ResNet进行迁移学习？](MODELS_MODULE.md#torchvision模型)
- [如何启用混合精度训练？](ENGINE_MODULE.md#混合精度训练)
- [如何计算Dice系数？](METRICS_MODULE.md#dice-coefficient)
- [如何导出ONNX模型？](UTILS_AND_TOOLS.md#onnx导出)

### 进阶主题

- [分布式训练配置](ENGINE_MODULE.md#deepspeedtrainer)
- [自定义数据集](DATASET_MODULE.md#自定义数据集)
- [模型量化](UTILS_AND_TOOLS.md#量化工具)
- [LoRA微调](UTILS_AND_TOOLS.md#lora合并)

## 💡 最佳实践

1. **数据准备**
   - 始终先可视化检查数据
   - 计算数据集统计信息（mean/std）
   - 使用适当的数据增强

2. **模型选择**
   - 小数据集：使用较小的模型 + 预训练权重
   - 大数据集：可以使用更大的模型
   - 医学图像：考虑使用UNet等专用架构

3. **训练策略**
   - 使用学习率调度
   - 启用早停机制
   - 定期保存检查点
   - 使用混合精度训练加速

4. **评估和调试**
   - 使用多个指标评估
   - 可视化训练曲线
   - 分析错误样本
   - 监控梯度和激活

## 🔍 故障排除

### 常见问题

**Q: 内存不足 (OOM)**
- 减小batch_size
- 使用梯度累积
- 启用混合精度训练
- 使用DeepSpeed ZeRO

**Q: 训练不收敛**
- 检查学习率（可能太大或太小）
- 检查数据预处理
- 尝试不同的优化器
- 检查损失函数

**Q: 验证集性能差**
- 增加数据增强
- 使用正则化（dropout, weight decay）
- 尝试早停
- 使用更多训练数据

**Q: 训练速度慢**
- 增加num_workers
- 使用pin_memory
- 启用混合精度
- 使用分布式训练

## 📚 参考资源

### 官方文档
- [PyTorch文档](https://pytorch.org/docs/)
- [TorchVision文档](https://pytorch.org/vision/)
- [TIMM文档](https://timm.fast.ai/)

### 论文和教程
- [U-Net论文](https://arxiv.org/abs/1505.04597)
- [ResNet论文](https://arxiv.org/abs/1512.03385)
- [CLIP论文](https://arxiv.org/abs/2103.00020)

### 社区资源
- [GitHub Issues](https://github.com/rinbarpen/NeuroTrain/issues)
- [Discussions](https://github.com/rinbarpen/NeuroTrain/discussions)

## 🤝 贡献文档

欢迎贡献文档改进！请：

1. Fork项目
2. 编辑文档（Markdown格式）
3. 提交Pull Request

### 文档规范

- 使用清晰的标题层次
- 提供代码示例
- 添加适当的链接
- 保持简洁明了

## 📝 文档更新日志

### 2024-10-24
- ✅ 添加项目架构文档
- ✅ 添加Dataset模块文档
- ✅ 添加Models模块文档
- ✅ 添加Engine模块文档
- ✅ 添加Metrics模块文档
- ✅ 添加Utils&Tools文档
- ✅ 添加Python示例代码
- ✅ 添加Jupyter Notebook教程

## 📧 获取帮助

如有问题或建议：

1. 查看相关文档
2. 搜索GitHub Issues
3. 提交新的Issue
4. 加入Discussions讨论

---

**NeuroTrain** - 让深度学习训练变得简单高效！ 🚀

