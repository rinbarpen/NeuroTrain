# Models 模块文档

## 概述

Models模块是NeuroTrain的核心组件，提供了统一的模型接口和丰富的模型选择。支持自定义模型、TorchVision预训练模型、TIMM模型库、以及多模态模型（CLIP、LLaVA等）。

## 主要特性

- 🎯 **统一的模型接口**: 通过 `get_model()` 函数统一获取各种模型
- 🔄 **多种模型后端**: 支持自定义、TorchVision、TIMM、Hugging Face等
- 🎨 **模块化设计**: 提供各种可组合的网络组件
- 🚀 **预训练支持**: 轻松加载预训练权重
- 🔧 **灵活定制**: 方便修改模型结构和参数
- 📊 **模型分析**: 集成模型摘要和FLOPs计算

## 核心函数

### get_model()

统一的模型获取接口，根据配置返回相应的模型实例。

```python
from src.models import get_model

# 获取模型
model = get_model(model_name, config)
```

**参数：**
- `model_name` (str): 模型名称，如 'unet', 'torchvision', 'timm', 'clip'等
- `config` (dict): 模型配置字典

**返回：**
- `nn.Module`: PyTorch模型实例

## 支持的模型类型

### 1. 自定义模型

#### UNet - 医学图像分割

UNet是经典的医学图像分割模型，采用编码器-解码器架构。

```python
config = {
    'n_channels': 3,      # 输入通道数
    'n_classes': 2,       # 输出类别数
    'bilinear': False     # 是否使用双线性插值上采样
}

model = get_model('unet', config)
```

**特点：**
- U型架构，具有跳跃连接
- 适用于医学图像分割任务
- 支持多尺度特征融合

**应用场景：**
- 视网膜血管分割
- 细胞核分割
- 器官分割

#### SimpleNet - 简单示例网络

用于演示和快速原型开发的简单网络。

```python
model = get_model('simple-net', {})
```

### 2. TorchVision模型

使用TorchVision提供的预训练模型，支持ResNet、VGG、DenseNet、EfficientNet等。

```python
config = {
    'arch': 'resnet18',         # 模型架构
    'pretrained': True,         # 使用预训练权重
    'n_classes': 10,            # 目标类别数
    'n_channels': 3             # 输入通道数（默认3）
}

model = get_model('torchvision', config)
```

**支持的架构：**

#### 分类模型
- **ResNet系列**: resnet18, resnet34, resnet50, resnet101, resnet152
- **VGG系列**: vgg11, vgg13, vgg16, vgg19 (可带bn)
- **DenseNet系列**: densenet121, densenet161, densenet169, densenet201
- **EfficientNet系列**: efficientnet_b0 到 efficientnet_b7
- **MobileNet系列**: mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
- **Vision Transformer**: vit_b_16, vit_b_32, vit_l_16, vit_l_32

**特点：**
- 自动加载ImageNet预训练权重
- 自动调整分类头适应目标类别数
- 支持灰度图像（自动修改第一层）

**示例：**

```python
# ResNet18用于CIFAR-10分类
config = {
    'arch': 'resnet18',
    'pretrained': True,
    'n_classes': 10
}
model = get_model('torchvision', config)

# EfficientNet用于ImageNet分类
config = {
    'arch': 'efficientnet_b0',
    'pretrained': True,
    'n_classes': 1000
}
model = get_model('torchvision', config)

# Vision Transformer
config = {
    'arch': 'vit_b_16',
    'pretrained': True,
    'n_classes': 100  # CIFAR-100
}
model = get_model('torchvision', config)
```

### 3. TIMM模型库

TIMM (PyTorch Image Models) 提供了数百种预训练模型。

```python
config = {
    'model_name': 'efficientnet_b0',  # TIMM模型名称
    'pretrained': True,                # 使用预训练权重
    'n_classes': 100,                  # 目标类别数
    'n_channels': 3                    # 输入通道数
}

model = get_model('timm', config)
```

**热门模型：**
- EfficientNet系列
- RegNet系列
- NFNet系列
- ConvNeXt系列
- Swin Transformer系列
- 各种ViT变体

**安装：**
```bash
pip install timm
```

**查看可用模型：**
```python
import timm
available_models = timm.list_models()
print(f"Available models: {len(available_models)}")
```

### 4. CLIP - 多模态模型

CLIP (Contrastive Language-Image Pre-training) 是OpenAI的多模态模型，能够理解图像和文本的关系。

```python
config = {
    'model_name': 'openai/clip-vit-base-patch32',
    'cache_dir': 'cache/models/pretrained',
    'device': 'cuda',
    'dtype': torch.float16
}

model = get_model('clip', config)
```

**可用的CLIP模型：**
- openai/clip-vit-base-patch32
- openai/clip-vit-base-patch16
- openai/clip-vit-large-patch14

**功能：**
- 图像编码
- 文本编码
- 图像-文本相似度计算
- 零样本分类
- 图像检索

**示例：**
```python
from transformers import CLIPProcessor

processor = CLIPProcessor.from_pretrained(config['model_name'])
model = get_model('clip', config)

# 图像编码
images = torch.randn(4, 3, 224, 224)
image_features = model.get_image_features(images)

# 文本编码
texts = ["a photo of a cat", "a photo of a dog"]
text_inputs = processor(text=texts, return_tensors="pt", padding=True)
text_features = model.get_text_features(**text_inputs)

# 计算相似度
similarity = torch.matmul(image_features, text_features.T)
```

## 模型组件

Models模块还提供了丰富的可组合组件，用于构建自定义模型。

### 注意力机制 (attention/)

```python
from src.models.attention import SelfAttention, MultiHeadAttention

# 自注意力
self_attn = SelfAttention(embed_dim=512, num_heads=8)

# 多头注意力
multi_head_attn = MultiHeadAttention(embed_dim=512, num_heads=8)
```

**可用的注意力模块：**
- SelfAttention: 自注意力
- MultiHeadAttention: 多头注意力
- CrossAttention: 交叉注意力
- SpatialAttention: 空间注意力
- ChannelAttention: 通道注意力
- CBAM: 卷积块注意力模块

### Transformer组件 (transformer/)

```python
from src.models.transformer import TransformerBlock, TransformerEncoder

# Transformer块
transformer_block = TransformerBlock(
    embed_dim=512,
    num_heads=8,
    mlp_ratio=4.0,
    dropout=0.1
)

# Transformer编码器
encoder = TransformerEncoder(
    num_layers=6,
    embed_dim=512,
    num_heads=8
)
```

### 卷积层变体 (conv/)

```python
from src.models.conv import DepthwiseSeparableConv, InvertedResidual

# 深度可分离卷积
dwconv = DepthwiseSeparableConv(in_channels=64, out_channels=128)

# 倒残差块（MobileNet）
inverted_residual = InvertedResidual(
    in_channels=64,
    out_channels=128,
    stride=1,
    expand_ratio=6
)
```

### 归一化层 (norm/)

```python
from src.models.norm import LayerNorm, GroupNorm, BatchNorm2d

# Layer Normalization
ln = LayerNorm(normalized_shape=512)

# Group Normalization
gn = GroupNorm(num_groups=32, num_channels=512)
```

### 位置编码 (position_encoding.py)

```python
from src.models.position_encoding import PositionalEncoding, LearnedPositionalEncoding

# 固定位置编码
pos_enc = PositionalEncoding(d_model=512, max_len=5000)

# 可学习位置编码
learned_pos_enc = LearnedPositionalEncoding(d_model=512, max_len=5000)
```

### 嵌入层 (embedding.py)

```python
from src.models.embedding import PatchEmbedding, TokenEmbedding

# 图像块嵌入（Vision Transformer）
patch_emb = PatchEmbedding(
    img_size=224,
    patch_size=16,
    in_channels=3,
    embed_dim=768
)

# Token嵌入
token_emb = TokenEmbedding(
    vocab_size=10000,
    embed_dim=512
)
```

## 模型定制

### 修改分类头

```python
import torch.nn as nn

# 加载预训练模型
config = {'arch': 'resnet18', 'pretrained': True, 'n_classes': 1000}
model = get_model('torchvision', config)

# 替换分类头
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 10)  # 10个类别
)
```

### 冻结部分层

```python
# 冻结除分类头外的所有层
for name, param in model.named_parameters():
    if 'fc' not in name:  # 不冻结fc层
        param.requires_grad = False

# 统计可训练参数
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
```

### 添加自定义层

```python
class CustomModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.custom_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # 提取特征
        features = self.base.features(x)
        # 自定义分类头
        output = self.custom_head(features)
        return output

# 使用
base = get_model('torchvision', {'arch': 'resnet18', 'pretrained': True})
model = CustomModel(base)
```

## 模型分析

### 模型摘要

```python
from torchinfo import summary

model = get_model('unet', {'n_channels': 3, 'n_classes': 2})

# 打印模型摘要
summary(model, 
        input_size=(1, 3, 512, 512),  # (batch_size, channels, height, width)
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        depth=4)
```

### 计算FLOPs

```python
from fvcore.nn import FlopCountAnalysis, parameter_count

model = get_model('resnet18', {'arch': 'resnet18', 'pretrained': False})
inputs = torch.randn(1, 3, 224, 224)

# 计算FLOPs
flops = FlopCountAnalysis(model, inputs)
print(f"FLOPs: {flops.total() / 1e9:.2f} G")

# 计算参数量
params = parameter_count(model)
print(f"Parameters: {params[''] / 1e6:.2f} M")
```

## 模型保存和加载

### 保存模型

```python
import torch
from pathlib import Path

# 保存完整模型
save_path = Path('models/my_model.pth')
save_path.parent.mkdir(parents=True, exist_ok=True)

torch.save({
    'model_state_dict': model.state_dict(),
    'model_name': 'unet',
    'config': config,
    'epoch': epoch,
    'optimizer_state_dict': optimizer.state_dict(),
}, save_path)

# 仅保存权重
torch.save(model.state_dict(), 'models/model_weights.pth')
```

### 加载模型

```python
# 加载完整模型
checkpoint = torch.load('models/my_model.pth', map_location='cpu')

# 重建模型
model = get_model(checkpoint['model_name'], checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 仅加载权重
model.load_state_dict(torch.load('models/model_weights.pth'))
```

### 部分加载

```python
# 加载部分权重（如预训练的backbone）
pretrained_dict = torch.load('pretrained_backbone.pth')
model_dict = model.state_dict()

# 过滤不匹配的键
pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                   if k in model_dict and model_dict[k].shape == v.shape}

# 更新模型字典
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
```

## 模型导出

### ONNX导出

```python
from pathlib import Path

model = get_model('resnet18', {'arch': 'resnet18', 'pretrained': True})
model.eval()

# 准备示例输入
dummy_input = torch.randn(1, 3, 224, 224)

# 导出为ONNX
onnx_path = Path('models/resnet18.onnx')
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    },
    opset_version=11
)

print(f"Model exported to {onnx_path}")
```

### TorchScript导出

```python
# Trace模式
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save('models/model_traced.pt')

# Script模式
scripted_model = torch.jit.script(model)
scripted_model.save('models/model_scripted.pt')

# 加载TorchScript模型
loaded_model = torch.jit.load('models/model_traced.pt')
loaded_model.eval()
```

## 最佳实践

### 1. 选择合适的模型

**图像分类：**
- 小数据集：ResNet18, MobileNetV2
- 中等数据集：ResNet50, EfficientNet-B0
- 大数据集：EfficientNet-B4, Vision Transformer

**医学图像分割：**
- 2D分割：UNet, UNet++
- 3D分割：3D UNet, V-Net

**目标检测：**
- 实时检测：YOLO, SSD
- 高精度检测：Faster R-CNN, Mask R-CNN

### 2. 预训练策略

- 尽可能使用预训练权重
- ImageNet预训练适用于大多数视觉任务
- 医学图像可能需要领域特定的预训练

### 3. 迁移学习

```python
# Step 1: 加载预训练模型
model = get_model('torchvision', {
    'arch': 'resnet50',
    'pretrained': True,
    'n_classes': 1000
})

# Step 2: 冻结早期层
for param in list(model.parameters())[:-10]:
    param.requires_grad = False

# Step 3: 修改分类头
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Step 4: 使用较小的学习率训练
optimizer = torch.optim.Adam([
    {'params': model.fc.parameters(), 'lr': 1e-3},
    {'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': 1e-4}
])
```

### 4. 模型性能优化

```python
# 混合精度训练
model = model.half()  # 转换为float16

# 梯度检查点（节省内存）
from torch.utils.checkpoint import checkpoint

class CheckpointedBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
    
    def forward(self, x):
        return checkpoint(self.block, x)

# 模型编译（PyTorch 2.0+）
compiled_model = torch.compile(model)
```

## 常见问题

### Q: 如何查看模型结构？

```python
# 方法1: 打印模型
print(model)

# 方法2: 使用torchinfo
from torchinfo import summary
summary(model, input_size=(1, 3, 224, 224))

# 方法3: 可视化
from torchviz import make_dot
y = model(x)
make_dot(y, params=dict(model.named_parameters())).render("model", format="png")
```

### Q: 模型太大，内存不足？

- 使用更小的模型（如MobileNet）
- 使用混合精度训练
- 减小batch size
- 使用梯度检查点
- 使用模型量化

### Q: 如何添加自定义层？

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用现有模型作为backbone
        self.backbone = get_model('torchvision', {
            'arch': 'resnet18',
            'pretrained': True
        })
        # 移除原始分类头
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        # 添加自定义层
        self.custom_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.custom_layers(x)
        return x
```

## 参考资料

- [PyTorch模型文档](https://pytorch.org/docs/stable/nn.html)
- [TorchVision模型](https://pytorch.org/vision/stable/models.html)
- [TIMM库文档](https://timm.fast.ai/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [ONNX文档](https://onnx.ai/)

---

更多示例请查看 `examples/models_basic_example.py` 和 `examples/` 目录中的Jupyter Notebooks。

