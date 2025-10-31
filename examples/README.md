# NeuroTrain Examples

欢迎来到NeuroTrain示例代码库！本目录包含了各种使用示例，帮助您快速上手。

## 📚 示例目录

### Python脚本示例

#### 1. dataset_basic_example.py
**数据集基础使用示例**

展示如何使用NeuroTrain的数据集模块：
- 加载单个数据集（DRIVE, CIFAR-10等）
- 混合数据集使用
- 数据增强配置
- DataLoader使用
- 随机采样

```bash
python examples/dataset_basic_example.py
```

#### 2. models_basic_example.py
**模型创建和使用示例**

展示如何使用NeuroTrain的模型模块：
- UNet模型（医学图像分割）
- TorchVision模型（ResNet, VGG等）
- TIMM模型库
- CLIP多模态模型
- 模型对比和定制
- 模型保存和加载

```bash
python examples/models_basic_example.py
```

#### 3. complete_training_example.py
**完整训练流程示例**

展示从头到尾的完整训练流程：
- 配置管理
- 数据准备
- 模型创建
- 训练循环
- 验证和测试
- 结果保存
- 日志记录

```bash
python examples/complete_training_example.py
```

### Jupyter Notebook教程

#### 1. dataset_tutorial.ipynb
**Dataset模块完整教程**

交互式教程，涵盖：
- 基础数据集加载（CIFAR-10, DRIVE）
- 数据增强演示
- DataLoader配置
- 混合数据集
- 数据集统计分析

```bash
jupyter notebook examples/dataset_tutorial.ipynb
```

#### 2. complete_workflow_tutorial.ipynb
**完整工作流程教程**

端到端的深度学习项目教程：
- 环境配置
- 数据探索和可视化
- 模型选择和配置
- 训练过程监控
- 结果分析
- 混淆矩阵
- 模型导出

```bash
jupyter notebook examples/complete_workflow_tutorial.ipynb
```

## 🚀 快速开始

### 前置要求

```bash
# 激活环境
conda activate ntrain

# 安装依赖
uv pip install -e '.[cu128]'

# 对于notebooks，还需要安装
pip install jupyter ipykernel
```

### 运行Python示例

```bash
# 进入项目根目录
cd /path/to/NeuroTrain

# 运行示例
python examples/dataset_basic_example.py
python examples/models_basic_example.py
python examples/complete_training_example.py
```

### 运行Jupyter Notebooks

```bash
# 启动Jupyter
jupyter notebook

# 在浏览器中打开
# 导航到 examples/ 目录
# 打开任意 .ipynb 文件
```

## 📖 按任务类型分类

### 图像分类
- `complete_training_example.py` - CIFAR-10分类
- `complete_workflow_tutorial.ipynb` - ResNet18训练
- `models_basic_example.py` - 各种分类模型

### 医学图像分割
- `dataset_basic_example.py` - DRIVE数据集
- 查看 `configs/single/train-drive.toml` 配置示例

### 数据处理
- `dataset_basic_example.py` - 数据集操作
- `dataset_tutorial.ipynb` - 交互式数据探索

### 模型相关
- `models_basic_example.py` - 模型创建和使用
- 查看 `docs/MODELS_MODULE.md` 了解更多

## 🎯 学习路径

### 初学者路径

1. **开始**: 阅读 `README.md` 了解项目
2. **数据**: 运行 `dataset_basic_example.py`
3. **模型**: 运行 `models_basic_example.py`
4. **训练**: 阅读 `complete_workflow_tutorial.ipynb`
5. **实践**: 修改配置文件进行自己的实验

### 进阶路径

1. **深入数据**: 学习自定义数据集
2. **模型定制**: 修改模型架构
3. **高级训练**: 分布式训练、混合精度
4. **优化部署**: 模型量化、ONNX导出
5. **工具使用**: 学习分析工具

## 💡 示例修改指南

### 修改数据集

```python
# 在示例中找到数据集配置
config = {
    'dataset': {
        'name': 'cifar10',  # 改为 'drive', 'coco' 等
        'root_dir': 'data/cifar10',  # 修改数据路径
        ...
    }
}
```

### 修改模型

```python
# 修改模型配置
model_config = {
    'arch': 'resnet18',  # 改为 'resnet50', 'efficientnet_b0' 等
    'pretrained': True,  # 是否使用预训练权重
    'n_classes': 10,     # 修改类别数
}
```

### 修改训练参数

```python
# 修改训练配置
num_epochs = 20          # 训练轮数
learning_rate = 0.001    # 学习率
batch_size = 128         # 批大小
```

## 🔍 示例输出

所有示例的输出将保存在以下位置：

```
examples/output/          # Python脚本输出
├── cifar10_samples.png
├── augmentation_examples.png
├── sample_visualization.png
└── models/
    └── *.pth

runs/tutorial_example/    # 训练结果
├── best_model.pth
├── results.json
└── loss_curves.png
```

## 📝 常见问题

### Q: 运行示例时找不到数据集？

**A**: 首先下载所需数据集到 `data/` 目录，或在配置中启用 `download=True`。

### Q: 内存不足错误？

**A**: 减小 `batch_size` 参数或使用更小的模型。

### Q: 示例运行很慢？

**A**: 
- 确保CUDA可用：`torch.cuda.is_available()`
- 减少 `num_epochs`
- 使用更小的数据集

### Q: 如何保存训练结果？

**A**: 所有示例都会自动保存结果到指定目录，检查输出路径即可。

## 🤝 贡献示例

欢迎贡献新的示例！请：

1. 创建清晰的示例代码
2. 添加详细的注释
3. 更新本README
4. 提交Pull Request

### 示例代码规范

- 使用清晰的变量名
- 添加适当的注释
- 包含错误处理
- 提供输出示例
- 记录依赖项

## 📚 相关文档

- [项目README](../README.md)
- [架构文档](../docs/ARCHITECTURE.md)
- [Dataset模块文档](../docs/DATASET_MODULE.md)
- [Models模块文档](../docs/MODELS_MODULE.md)
- [Engine模块文档](../docs/ENGINE_MODULE.md)
- [工具文档](../docs/UTILS_AND_TOOLS.md)

## 📧 获取帮助

遇到问题？

1. 查看示例代码注释
2. 阅读相关文档
3. 搜索GitHub Issues
4. 提交新的Issue

---

**Happy Learning with NeuroTrain! 🚀**

