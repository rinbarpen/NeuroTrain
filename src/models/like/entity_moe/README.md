# Vision Models with EntityMoE

基于 `timm` 和 `transformers` 库的预训练视觉模型，集成 EntityMoE 增强特征表示能力。

## 支持的模型

1. **ViT (Vision Transformer)** - 通过 `timm` 获取
2. **Swin Transformer** - 通过 `timm` 获取
3. **ResNet** - 通过 `timm` 获取
4. **SAM (Segment Anything Model)** - 通过 `transformers` 获取

## 依赖安装

```bash
# 安装 timm（用于 ViT, Swin, ResNet）
pip install timm

# 安装 transformers（用于 SAM）
pip install transformers
```

## 快速开始

### 1. ViT with EntityMoE

```python
from models.like.entity_moe.vit_entity_moe import create_vit_entity_moe, vit_base_entity_moe

# 方法 1: 使用预定义函数
model = vit_base_entity_moe(
    pretrained=True,          # 加载预训练权重
    num_classes=1000,         # 分类类别数
    num_experts=4,            # MoE 专家数量
    num_experts_shared=2,     # 共享专家数量
    expert_k=1,               # 每次激活的专家数
    inject_layers='last'      # 在最后一层注入 EntityMoE
)

# 方法 2: 使用通用创建函数
model = create_vit_entity_moe(
    model_name='vit_base_patch16_224',  # timm 模型名称
    pretrained=True,
    num_classes=1000,
    inject_layers=[9, 10, 11]  # 在指定层注入 EntityMoE
)

# 前向传播
import torch
x = torch.randn(2, 3, 224, 224)
output = model(x)  # (2, 1000)
```

### 2. Swin Transformer with EntityMoE

```python
from models.like.entity_moe.vit_entity_moe import create_swin_entity_moe, swin_tiny_entity_moe

# 方法 1: 使用预定义函数
model = swin_tiny_entity_moe(
    pretrained=True,
    num_classes=1000,
    inject_layers='last_stage'  # 只在最后一个 stage 注入
)

# 方法 2: 指定具体的 stage 和 block
model = create_swin_entity_moe(
    model_name='swin_tiny_patch4_window7_224',
    pretrained=True,
    num_classes=1000,
    inject_layers=[(3, 0), (3, 1)]  # 在 stage 3 的 block 0 和 1 注入
)

# 前向传播
x = torch.randn(2, 3, 224, 224)
output = model(x)  # (2, 1000)
```

### 3. ResNet with EntityMoE

```python
from models.like.entity_moe.vit_entity_moe import create_resnet_entity_moe, resnet50_entity_moe

# 方法 1: 使用预定义函数
model = resnet50_entity_moe(
    pretrained=True,
    num_classes=1000,
    inject_layers='layer4'  # 只在 layer4 注入
)

# 方法 2: 指定具体的 layer 和 block
model = create_resnet_entity_moe(
    model_name='resnet50',
    pretrained=True,
    num_classes=1000,
    inject_layers=[('layer3', 5), ('layer4', 0), ('layer4', 1)]
)

# 前向传播
x = torch.randn(2, 3, 224, 224)
output = model(x)  # (2, 1000)
```

### 4. SAM with EntityMoE

```python
from models.like.entity_moe.vit_entity_moe import create_sam_entity_moe, sam_vit_base_entity_moe

# 方法 1: 使用预定义函数
model = sam_vit_base_entity_moe(
    pretrained=True,
    inject_layers='last_half'  # 在后半部分的层注入
)

# 方法 2: 指定具体的层
model = create_sam_entity_moe(
    model_name='facebook/sam-vit-base',
    pretrained=True,
    inject_layers=[6, 7, 8, 9, 10, 11]  # 在指定层注入
)

# 前向传播
x = torch.randn(1, 3, 1024, 1024)
output = model(pixel_values=x)

# 访问 vision encoder 的输出
vision_features = output.vision_outputs.last_hidden_state  # (1, num_patches, hidden_size)
```

## 参数说明

### 通用参数

- **`pretrained`**: 是否加载预训练权重（默认: `True`）
- **`num_experts`**: MoE 专家数量（默认: `4`）
- **`num_experts_shared`**: 共享专家数量（默认: `2`）
- **`expert_k`**: 稀疏路由时每次激活的专家数量（默认: `1`）
- **`dropout`**: Dropout 比例（默认: `0.1`）
- **`mlp_ratio`**: MLP 隐藏层维度倍数（默认: `4.0`）

### inject_layers 参数说明

#### ViT
- `'all'`: 在所有 Transformer blocks 中注入
- `'last'`: 只在最后一层注入
- `[6, 7, 8, 9, 10, 11]`: 在指定索引的层注入

#### Swin Transformer
- `'all'`: 在所有 blocks 中注入
- `'last_stage'`: 只在最后一个 stage 注入
- `[(3, 0), (3, 1)]`: 在指定的 (stage_idx, block_idx) 注入

#### ResNet
- `'all'`: 在所有 layers 中注入
- `'layer3'`: 只在 layer3 注入
- `'layer4'`: 只在 layer4 注入
- `[('layer3', 5), ('layer4', 0)]`: 在指定的 (layer_name, block_idx) 注入

#### SAM
- `'all'`: 在所有 encoder layers 中注入
- `'last_half'`: 在后半部分的层注入
- `[6, 7, 8, 9, 10, 11]`: 在指定索引的层注入

## 可用的预定义模型

### ViT
```python
from models.like.entity_moe.vit_entity_moe import (
    vit_base_entity_moe,    # ViT-Base
    vit_large_entity_moe,   # ViT-Large
)
```

### Swin Transformer
```python
from models.like.entity_moe.vit_entity_moe import (
    swin_tiny_entity_moe,   # Swin-Tiny
    swin_base_entity_moe,   # Swin-Base
)
```

### ResNet
```python
from models.like.entity_moe.vit_entity_moe import (
    resnet50_entity_moe,    # ResNet-50
    resnet101_entity_moe,   # ResNet-101
)
```

### SAM
```python
from models.like.entity_moe.vit_entity_moe import (
    sam_vit_base_entity_moe,   # SAM ViT-Base
    sam_vit_large_entity_moe,  # SAM ViT-Large
    sam_vit_huge_entity_moe,   # SAM ViT-Huge
)
```

## 自定义 timm 模型

你可以使用任何 timm 支持的模型：

```python
from models.like.entity_moe.vit_entity_moe import create_vit_entity_moe

# 使用其他 ViT 变体
model = create_vit_entity_moe(
    model_name='vit_small_patch16_224',
    pretrained=True,
    num_classes=1000
)

# 使用其他 Swin 变体
from models.like.entity_moe.vit_entity_moe import create_swin_entity_moe
model = create_swin_entity_moe(
    model_name='swin_base_patch4_window12_384',
    pretrained=True,
    num_classes=1000
)

# 使用其他 ResNet 变体
from models.like.entity_moe.vit_entity_moe import create_resnet_entity_moe
model = create_resnet_entity_moe(
    model_name='resnet152',
    pretrained=True,
    num_classes=1000
)
```

## 查看模型配置

```python
from models.like.entity_moe.vit_entity_moe import print_model_info

model = vit_base_entity_moe(pretrained=False)
print_model_info(model)

# 输出示例:
# ============================================================
# EntityMoE Configuration
# ============================================================
# model_type: vit
# base_model: vit_base_patch16_224
# inject_layers: [11]
# num_experts: 4
# num_experts_shared: 2
# expert_k: 1
# ============================================================
```

## 训练示例

```python
import torch
import torch.nn as nn
import torch.optim as optim
from models.like.entity_moe.vit_entity_moe import vit_base_entity_moe

# 创建模型
model = vit_base_entity_moe(
    pretrained=True,
    num_classes=10,  # 自定义类别数
    inject_layers='last'
)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# 训练循环
model.train()
for images, labels in train_loader:
    # 前向传播
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 微调建议

1. **选择合适的注入位置**：
   - 简单任务：在最后几层注入即可（`inject_layers='last'` 或 `'last_stage'`）
   - 复杂任务：在更多层注入（`inject_layers='all'`）

2. **调整 MoE 参数**：
   - 增加 `num_experts` 可以提高模型容量，但也增加计算量
   - 增加 `expert_k` 可以让模型使用更多专家，但降低稀疏性

3. **学习率设置**：
   - 预训练权重部分：使用较小的学习率（如 1e-5）
   - EntityMoE 部分：可以使用较大的学习率（如 1e-4）

4. **冻结部分参数**（可选）：
```python
# 冻结预训练的主干网络
for name, param in model.named_parameters():
    if 'entity_moe' not in name:
        param.requires_grad = False
```

## 性能优化

1. **减少注入层数**：只在关键层注入 EntityMoE
2. **使用混合精度训练**：
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)
scaler.scale(loss).backward()
```

3. **使用梯度累积**：处理大 batch size

## 注意事项

1. **内存占用**：EntityMoE 会增加模型参数量和显存占用
2. **计算开销**：MoE 会增加前向传播的计算时间
3. **预训练权重**：首次使用会自动下载预训练权重，需要网络连接
4. **SAM 模型**：SAM 模型输入尺寸固定为 1024×1024

## 常见问题

**Q: 如何只使用 vision encoder 而不加载完整的 SAM 模型？**
```python
model = sam_vit_base_entity_moe(pretrained=True)
vision_encoder = model.vision_encoder  # 只使用 encoder 部分
```

**Q: 如何在不同的 GPU 上运行？**
```python
model = vit_base_entity_moe(pretrained=True)
model = model.to('cuda:0')  # 或 'cuda:1', 'cpu' 等
```

**Q: 如何保存和加载模型？**
```python
# 保存
torch.save(model.state_dict(), 'model_entity_moe.pth')

# 加载
model = vit_base_entity_moe(pretrained=False)
model.load_state_dict(torch.load('model_entity_moe.pth'))
```

## 引用

如果你使用了这些模型，请引用相关论文：

```bibtex
# ViT
@article{dosovitskiy2020vit,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and others},
  journal={ICLR},
  year={2021}
}

# Swin Transformer
@inproceedings{liu2021swin,
  title={Swin transformer: Hierarchical vision transformer using shifted windows},
  author={Liu, Ze and others},
  booktitle={ICCV},
  year={2021}
}

# ResNet
@inproceedings{he2016resnet,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and others},
  booktitle={CVPR},
  year={2016}
}

# SAM
@article{kirillov2023sam,
  title={Segment anything},
  author={Kirillov, Alexander and others},
  journal={arXiv preprint arXiv:2304.02643},
  year={2023}
}
```

