# Dataset 模块文档

## 概述

Dataset模块是NeuroTrain的核心组件之一，负责数据的加载、预处理和增强。该模块提供了灵活的数据集管理系统，支持多种医学图像数据集、标准计算机视觉数据集，以及扩散模型数据集。

## 主要特性

- 🎯 **统一的数据集接口**: 所有数据集都遵循统一的接口设计
- 🔄 **混合数据集支持**: 可以混合使用多个数据集进行训练
- 🎲 **智能采样策略**: 支持权重采样、平衡采样、优先级采样
- 🔧 **丰富的数据增强**: 集成多种数据增强方法
- 🤖 **LLM驱动分析**: 使用大语言模型分析和筛选数据
- 📊 **数据集注册表**: 自动管理所有可用数据集
- 🌐 **自动下载**: 支持自动下载标准数据集

## 数据集注册表

所有可用的数据集都在 `DATASET_REGISTRY` 中注册，可以通过名称快速访问。

### 医学图像数据集

#### 视网膜血管分割
- **drive**: DRIVE数据集
- **medical/chasedb1**: CHASE-DB1数据集
- **medical/stare**: STARE数据集

#### 皮肤病变
- **medical/isic2016**: ISIC 2016 皮肤病变分割
- **medical/isic2017**: ISIC 2017 皮肤病变分类
- **medical/isic2018**: ISIC 2018 皮肤病变分析

#### 3D医学图像
- **medical/btcv**: BTCV多器官分割
- **medical/brats2020**: BraTS 2020脑肿瘤分割

#### 细胞分割
- **medical/bowl2018**: Data Science Bowl 2018核分割

#### 多模态医学数据
- **medical/vqarad**: VQA-RAD医学视觉问答
- **medical/mri_brain_clip**: MRI脑部CLIP数据集

### 标准计算机视觉数据集

#### 图像分类
- **mnist**: MNIST手写数字
- **cifar10**: CIFAR-10图像分类
- **cifar100**: CIFAR-100图像分类
- **imagenet**: ImageNet大规模图像分类

#### 目标检测和分割
- **coco**: COCO数据集（检测、分割、关键点）
- **coco/detection**: COCO目标检测
- **coco/segmentation**: COCO实例分割
- **coco/keypoint**: COCO关键点检测
- **coco/caption**: COCO图像描述

### 扩散模型数据集

- **diffusion**: 通用扩散模型数据集
- **unconditional_diffusion**: 无条件扩散模型
- **conditional_diffusion**: 条件扩散模型
- **text_to_image_diffusion**: 文本到图像扩散模型

## 核心类和函数

### 1. 数据集获取函数

```python
from src.dataset import (
    get_dataset,
    get_train_dataset,
    get_test_dataset,
    get_valid_dataset,
    get_train_valid_test_dataloader
)

# 获取完整数据集
dataset = get_dataset(config)

# 获取训练/测试/验证数据集
train_dataset = get_train_dataset(config)
test_dataset = get_test_dataset(config)
valid_dataset = get_valid_dataset(config)

# 获取数据加载器
train_loader, valid_loader, test_loader = get_train_valid_test_dataloader(config)
```

### 2. CustomDataset - 自定义数据集基类

`CustomDataset` 是所有自定义数据集的基类，提供了标准的数据集接口。

```python
from src.dataset.custom_dataset import CustomDataset

class MyDataset(CustomDataset):
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        # 初始化数据列表
        self._load_data()
    
    def _load_data(self):
        # 加载数据路径等
        pass
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 加载和处理单个样本
        image, mask = self._load_sample(idx)
        
        if self.transform:
            image, mask = self.transform(image, mask)
        
        return image, mask
```

### 3. HybridDataset - 混合数据集

`HybridDataset` 允许混合使用多个数据集进行训练，支持多种采样策略。

```python
from src.dataset.hybrid_dataset import HybridDataset, create_hybrid_dataset_from_config

# 方法1: 直接创建
dataset1 = get_dataset(config1)
dataset2 = get_dataset(config2)

hybrid_dataset = HybridDataset(
    datasets=[dataset1, dataset2],
    sampling_strategy="weighted",  # 采样策略
    weights=[0.7, 0.3],             # 采样权重
    ratios=[0.6, 0.4]               # 数据集比例
)

# 方法2: 从配置创建
hybrid_dataset = create_hybrid_dataset_from_config(config)
```

#### 采样策略

1. **weighted (权重采样)**
   - 根据指定权重从各数据集采样
   - 适用于数据集质量不同的情况

2. **balanced (平衡采样)**
   - 确保各数据集的样本数量平衡
   - 适用于数据集大小差异大的情况

3. **priority (优先级采样)**
   - 按优先级顺序采样
   - 适用于有主次数据集的情况

4. **sequential (顺序采样)**
   - 按顺序依次采样各数据集
   - 适用于课程学习场景

### 4. DiffusionDataset - 扩散模型数据集

专门为扩散模型设计的数据集类。

```python
from src.dataset.diffusion_dataset import (
    DiffusionDataset,
    UnconditionalDiffusionDataset,
    ConditionalDiffusionDataset,
    TextToImageDiffusionDataset,
    get_mnist_diffusion_dataset,
    get_cifar10_diffusion_dataset
)

# 无条件扩散
unconditional_dataset = UnconditionalDiffusionDataset(
    image_dir="path/to/images",
    image_size=(64, 64)
)

# 条件扩散（类别条件）
conditional_dataset = ConditionalDiffusionDataset(
    image_dir="path/to/images",
    label_file="path/to/labels.json",
    image_size=(64, 64),
    num_classes=10
)

# 文本到图像扩散
text_to_image_dataset = TextToImageDiffusionDataset(
    image_dir="path/to/images",
    caption_file="path/to/captions.json",
    image_size=(256, 256)
)

# 使用预定义数据集
mnist_diffusion = get_mnist_diffusion_dataset(
    root="./data",
    train=True,
    download=True
)
```

### 5. LLM数据分析器

使用大语言模型分析和筛选数据集。

```python
from src.dataset.llm_data_analyzer import LLMDataAnalyzer

analyzer = LLMDataAnalyzer(
    model_name="gpt-4",
    api_key="your_api_key"
)

# 分析数据集
analysis = analyzer.analyze_dataset(dataset)

# 根据条件筛选数据
filtered_indices = analyzer.filter_by_criteria(
    dataset,
    criteria="Find images with clear blood vessels"
)

# 生成数据集统计报告
report = analyzer.generate_report(dataset)
```

## 配置示例

### 单数据集配置

```toml
[dataset]
name = "drive"
root_dir = "data/drive"
is_rgb = true
train_split = 0.8
image_size = [512, 512]

[dataset.augmentation]
random_flip = true
random_rotation = true
rotation_range = 15
brightness_range = [0.8, 1.2]
```

### 混合数据集配置

```toml
[dataset]
name = "enhanced_hybrid"
datasets = ["drive", "medical/chasedb1", "medical/stare"]
sampling_strategy = "weighted"
ratios = [0.5, 0.3, 0.2]
weights = [1.0, 1.2, 0.8]

[dataset.drive]
root_dir = "data/drive"
is_rgb = true

[dataset."medical/chasedb1"]
root_dir = "data/chasedb1"
is_rgb = true

[dataset."medical/stare"]
root_dir = "data/stare"
is_rgb = true

[dataset.augmentation]
random_flip = true
random_rotation = true
elastic_deformation = true
```

### 扩散模型数据集配置

```toml
[dataset]
name = "conditional_diffusion"
image_dir = "data/images"
label_file = "data/labels.json"
image_size = [64, 64]
num_classes = 10

[dataset.augmentation]
random_flip = true
color_jitter = true
```

## 数据增强

NeuroTrain支持多种数据增强方法，可以通过配置文件灵活控制。

### 常用增强方法

```python
from src.utils.transform import get_transforms

# 获取增强变换
transforms = get_transforms(config['dataset']['augmentation'])

# 常用增强包括：
augmentation = {
    # 几何变换
    "random_flip": True,           # 随机翻转
    "random_rotation": True,       # 随机旋转
    "rotation_range": 15,          # 旋转角度范围
    "random_crop": True,           # 随机裁剪
    "crop_size": [256, 256],      # 裁剪大小
    
    # 颜色变换
    "brightness_range": [0.8, 1.2],  # 亮度调整
    "contrast_range": [0.8, 1.2],    # 对比度调整
    "saturation_range": [0.8, 1.2],  # 饱和度调整
    "hue_range": [-0.1, 0.1],        # 色调调整
    "color_jitter": True,             # 颜色抖动
    
    # 形变
    "elastic_deformation": True,   # 弹性变形
    "grid_distortion": True,       # 网格扭曲
    
    # 噪声
    "gaussian_noise": True,        # 高斯噪声
    "gaussian_blur": True,         # 高斯模糊
    
    # 医学图像特定
    "normalize": True,             # 标准化
    "clahe": True,                 # CLAHE对比度增强
}
```

### 自定义增强

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 自定义增强流程
custom_transform = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# 在数据集中使用
dataset = MyDataset(root_dir, transform=custom_transform)
```

## 数据采样

### 随机采样

```python
from src.dataset import random_sample

# 从数据集中随机采样n个样本
subset = random_sample(dataset, n=100)
```

### 分层采样

```python
from torch.utils.data import WeightedRandomSampler

# 为不平衡数据集创建采样器
class_counts = [1000, 500, 200]  # 各类别样本数
weights = [1.0/c for c in class_counts]
sample_weights = [weights[label] for label in labels]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# 在DataLoader中使用
loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

## 数据预处理

### 标准化

```python
# 计算数据集的均值和标准差
from src.dataset import compute_dataset_stats

mean, std = compute_dataset_stats(dataset)

# 使用计算的统计值进行标准化
normalize = A.Normalize(mean=mean, std=std)
```

### 调整大小和裁剪

```python
import albumentations as A

# 固定大小
resize = A.Resize(height=512, width=512)

# 保持长宽比
resize_keep_ratio = A.LongestMaxSize(max_size=512)
pad = A.PadIfNeeded(min_height=512, min_width=512, border_mode=0)

# 随机裁剪
random_crop = A.RandomCrop(height=256, width=256)

# 中心裁剪
center_crop = A.CenterCrop(height=256, width=256)
```

## 数据加载优化

### DataLoader配置

```python
from torch.utils.data import DataLoader

# 优化的DataLoader配置
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,        # 多进程加载
    pin_memory=True,      # 固定内存，加速传输
    persistent_workers=True,  # 保持workers活跃
    prefetch_factor=2     # 预取数据
)
```

### 数据预加载

```python
from src.dataset import PrefetchDataLoader

# 预取数据加载器
prefetch_loader = PrefetchDataLoader(
    loader,
    device='cuda'
)

# 使用
for batch in prefetch_loader:
    # 数据已经在GPU上
    images, labels = batch
    ...
```

## 常用数据集使用示例

### DRIVE数据集

```python
from src.dataset import get_dataset

config = {
    'dataset': {
        'name': 'drive',
        'root_dir': 'data/drive',
        'is_rgb': True,
        'train_split': 0.8,
        'image_size': [512, 512]
    }
}

train_dataset = get_train_dataset(config)
test_dataset = get_test_dataset(config)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
```

### COCO数据集

```python
config = {
    'dataset': {
        'name': 'coco/detection',
        'root_dir': 'data/coco',
        'annotation_file': 'data/coco/annotations/instances_train2017.json',
        'year': 2017
    }
}

dataset = get_dataset(config)

# 获取一个样本
image, target = dataset[0]
print(f"Image shape: {image.shape}")
print(f"Bounding boxes: {target['boxes']}")
print(f"Labels: {target['labels']}")
```

### CIFAR-10数据集

```python
config = {
    'dataset': {
        'name': 'cifar10',
        'root_dir': 'data/cifar10',
        'train': True,
        'download': True
    }
}

dataset = get_dataset(config)

# 查看数据集信息
print(f"Classes: {dataset.classes}")
print(f"Number of samples: {len(dataset)}")
```

## 高级功能

### 数据集拆分

```python
from torch.utils.data import random_split

# 按比例拆分数据集
total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, 
    [train_size, val_size, test_size]
)
```

### 数据集合并

```python
from torch.utils.data import ConcatDataset

# 合并多个数据集
combined_dataset = ConcatDataset([dataset1, dataset2, dataset3])
```

### 数据集子集

```python
from torch.utils.data import Subset

# 创建数据集子集
indices = [0, 1, 2, 10, 11, 12]  # 选择的索引
subset = Subset(dataset, indices)
```

### 自定义Collate函数

```python
def custom_collate_fn(batch):
    """自定义batch处理函数"""
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # 处理不同大小的图像
    images = torch.stack([pad_image(img, target_size) for img in images])
    labels = torch.tensor(labels)
    
    return images, labels

# 在DataLoader中使用
loader = DataLoader(dataset, batch_size=32, collate_fn=custom_collate_fn)
```

## 性能优化建议

1. **使用适当的num_workers**
   - CPU核心数的2-4倍
   - 太少：数据加载成为瓶颈
   - 太多：内存占用过高

2. **启用pin_memory**
   - 使用GPU训练时启用
   - 加速CPU到GPU的数据传输

3. **数据预处理**
   - 将耗时的预处理操作离线完成
   - 保存处理后的数据到磁盘

4. **数据缓存**
   - 小数据集可以完全加载到内存
   - 使用RAM disk加速IO

5. **混合精度**
   - 使用float16减少内存占用
   - 加速数据传输

## 故障排除

### 常见问题

1. **数据加载速度慢**
   - 增加num_workers
   - 使用SSD而非HDD
   - 减少数据增强操作

2. **内存不足**
   - 减小batch_size
   - 减少num_workers
   - 使用数据生成器而非预加载

3. **数据不平衡**
   - 使用WeightedRandomSampler
   - 使用过采样/欠采样
   - 调整损失函数权重

4. **数据增强效果不好**
   - 检查增强参数范围
   - 可视化增强后的样本
   - 逐步增加增强强度

## 最佳实践

1. **数据验证**: 加载数据后先可视化检查
2. **统计分析**: 计算数据集的统计信息（均值、标准差、分布等）
3. **版本控制**: 记录数据集版本和预处理步骤
4. **文档记录**: 记录数据集来源、格式、特点
5. **增量开发**: 先在小数据集上测试，再用完整数据集
6. **错误处理**: 添加异常处理，避免个别损坏数据影响训练

## 参考资料

- [PyTorch Dataset和DataLoader文档](https://pytorch.org/docs/stable/data.html)
- [Albumentations数据增强库](https://albumentations.ai/)
- [MONAI医学图像处理库](https://monai.io/)
- [TorchVision数据集](https://pytorch.org/vision/stable/datasets.html)

---

更多示例请查看 `examples/` 目录中的代码和Jupyter Notebooks。

