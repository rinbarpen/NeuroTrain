# EntityMoE 视觉模型 - 快速开始指南

## 🚀 5分钟上手

### 安装依赖

```bash
# 基础依赖
pip install torch torchvision

# ViT, Swin, ResNet
pip install timm

# SAM
pip install transformers
```

### 快速示例

#### 1️⃣ ViT with EntityMoE

```python
from models.like.entity_moe.vit_entity_moe import vit_base_entity_moe
import torch

# 创建模型
model = vit_base_entity_moe(pretrained=True, num_classes=1000)

# 推理
images = torch.randn(2, 3, 224, 224)
outputs = model(images)  # (2, 1000)
```

#### 2️⃣ Swin Transformer with EntityMoE

```python
from models.like.entity_moe.vit_entity_moe import swin_tiny_entity_moe

model = swin_tiny_entity_moe(pretrained=True, num_classes=1000)
outputs = model(images)
```

#### 3️⃣ ResNet with EntityMoE

```python
from models.like.entity_moe.vit_entity_moe import resnet50_entity_moe

model = resnet50_entity_moe(pretrained=True, num_classes=1000)
outputs = model(images)
```

#### 4️⃣ SAM with EntityMoE

```python
from models.like.entity_moe.vit_entity_moe import sam_vit_base_entity_moe

model = sam_vit_base_entity_moe(pretrained=True)
outputs = model(pixel_values=torch.randn(1, 3, 1024, 1024))
```

## ⚙️ 常用配置

### 选择注入位置

```python
# 只在最后一层（推荐用于快速实验）
model = vit_base_entity_moe(inject_layers='last')

# 在多个层（更强的表达能力）
model = vit_base_entity_moe(inject_layers=[9, 10, 11])

# 在所有层（最大容量，但计算开销大）
model = vit_base_entity_moe(inject_layers='all')
```

### 调整专家配置

```python
model = vit_base_entity_moe(
    num_experts=4,           # 专家数量（2-8）
    num_experts_shared=2,    # 共享专家数（0-4）
    expert_k=1,              # 激活专家数（1-2）
)
```

## 🎓 训练示例

```python
import torch.nn as nn
import torch.optim as optim

# 1. 创建模型
model = vit_base_entity_moe(pretrained=True, num_classes=10)

# 2. 定义优化器（可选：使用差异化学习率）
optimizer = optim.AdamW([
    {'params': [p for n, p in model.named_parameters() if 'entity_moe' not in n], 
     'lr': 1e-5},  # 主干网络小学习率
    {'params': [p for n, p in model.named_parameters() if 'entity_moe' in n], 
     'lr': 1e-4},  # EntityMoE 大学习率
])

criterion = nn.CrossEntropyLoss()

# 3. 训练循环
model.train()
for images, labels in train_loader:
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 💡 常见用法

### 自定义 timm 模型

```python
from models.like.entity_moe.vit_entity_moe import create_vit_entity_moe

# 使用任何 timm 支持的模型
model = create_vit_entity_moe(
    model_name='vit_large_patch16_384',  # 任何 timm 模型名
    pretrained=True,
    inject_layers='last'
)
```

### 查看模型信息

```python
from models.like.entity_moe.vit_entity_moe import print_model_info

model = vit_base_entity_moe()
print_model_info(model)
```

### 冻结部分参数

```python
# 只训练 EntityMoE 和分类头
for name, param in model.named_parameters():
    if 'entity_moe' not in name and 'head' not in name:
        param.requires_grad = False
```

## 📚 更多资源

- **详细文档**: `README.md`
- **实现总结**: `IMPLEMENTATION_SUMMARY.md`
- **完整示例**: `examples/entity_moe_example.py`

## ❓ 快速问答

**Q: 应该在多少层注入 EntityMoE？**
- 简单任务/快速实验：`inject_layers='last'`
- 一般任务：`inject_layers=[9, 10, 11]`（后几层）
- 复杂任务：`inject_layers='all'`

**Q: 如何减少显存占用？**
- 减少注入层数
- 减小 `num_experts`
- 使用混合精度训练

**Q: 预训练权重会被保留吗？**
- 是的，只在指定层添加 EntityMoE，原始权重保持不变

**Q: 训练速度会变慢吗？**
- 会有一些影响，但可以通过减少注入层数来优化

---

✅ **就这么简单！开始使用 EntityMoE 增强你的视觉模型吧！**

