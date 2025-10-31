# Vision Models with EntityMoE - 实现总结

## 📋 实现概述

已成功实现基于 **timm** 和 **transformers** 库的四种主流视觉模型与 EntityMoE 的集成，支持加载预训练权重并灵活注入 EntityMoE 层。

## ✅ 已完成的功能

### 1. 核心模型支持

| 模型 | 数据源 | 状态 | 预定义函数 |
|------|--------|------|-----------|
| **ViT** (Vision Transformer) | timm | ✅ | `vit_base_entity_moe`, `vit_large_entity_moe` |
| **Swin Transformer** | timm | ✅ | `swin_tiny_entity_moe`, `swin_base_entity_moe` |
| **ResNet** | timm | ✅ | `resnet50_entity_moe`, `resnet101_entity_moe` |
| **SAM** (Segment Anything) | transformers | ✅ | `sam_vit_base_entity_moe`, `sam_vit_large_entity_moe`, `sam_vit_huge_entity_moe` |

### 2. EntityMoE 集成方式

采用 **包装器模式**（`EntityMoEWrapper`）实现：

```python
原始模型层 → EntityMoE 增强 → 输出
    ↓
可学习融合权重 (alpha)
```

**优势**：
- ✅ 无需修改原始模型结构
- ✅ 可加载预训练权重
- ✅ 支持灵活的层级注入
- ✅ 保持与原模型接口一致

### 3. 灵活的注入策略

#### ViT
```python
inject_layers='all'          # 所有 Transformer blocks
inject_layers='last'         # 最后一层
inject_layers=[9, 10, 11]    # 指定层索引
```

#### Swin Transformer
```python
inject_layers='all'              # 所有 blocks
inject_layers='last_stage'       # 最后一个 stage
inject_layers=[(3, 0), (3, 1)]   # (stage_idx, block_idx)
```

#### ResNet
```python
inject_layers='all'                      # 所有 layers
inject_layers='layer4'                   # 指定 layer
inject_layers=[('layer4', 0), ('layer4', 1)]  # (layer_name, block_idx)
```

#### SAM
```python
inject_layers='all'              # 所有 encoder layers
inject_layers='last_half'        # 后半部分
inject_layers=[6, 7, 8, 9, 10, 11]  # 指定层索引
```

### 4. 主要 API 函数

#### 通用创建函数
```python
create_vit_entity_moe(model_name, pretrained=True, ...)
create_swin_entity_moe(model_name, pretrained=True, ...)
create_resnet_entity_moe(model_name, pretrained=True, ...)
create_sam_entity_moe(model_name, pretrained=True, ...)
```

#### 预定义快捷函数
```python
# ViT
vit_base_entity_moe(pretrained=True, num_classes=1000)
vit_large_entity_moe(pretrained=True, num_classes=1000)

# Swin
swin_tiny_entity_moe(pretrained=True, num_classes=1000)
swin_base_entity_moe(pretrained=True, num_classes=1000)

# ResNet
resnet50_entity_moe(pretrained=True, num_classes=1000)
resnet101_entity_moe(pretrained=True, num_classes=1000)

# SAM
sam_vit_base_entity_moe(pretrained=True)
sam_vit_large_entity_moe(pretrained=True)
sam_vit_huge_entity_moe(pretrained=True)
```

#### 工具函数
```python
print_model_info(model)  # 打印 EntityMoE 配置信息
```

### 5. 可配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `pretrained` | 是否加载预训练权重 | `True` |
| `num_classes` | 分类类别数 | `1000` |
| `num_experts` | MoE 专家数量 | `4` |
| `num_experts_shared` | 共享专家数量 | `2` |
| `expert_k` | 稀疏路由激活专家数 | `1` |
| `dropout` | Dropout 比例 | `0.1` |
| `mlp_ratio` | MLP 隐藏层倍数 | `4.0` |
| `inject_layers` | 注入位置策略 | 模型相关 |

## 📁 文件结构

```
src/models/like/entity_moe/
├── EntityMoe.py              # EntityMoE 核心实现
├── vit_entity_moe.py         # 主实现文件（新增）
├── README.md                 # 详细使用文档（新增）
└── IMPLEMENTATION_SUMMARY.md # 本文件（新增）

examples/
└── entity_moe_example.py     # 使用示例（新增）
```

## 🎯 使用场景

### 1. 图像分类
```python
from models.like.entity_moe.vit_entity_moe import vit_base_entity_moe

model = vit_base_entity_moe(pretrained=True, num_classes=1000)
output = model(images)  # (B, 1000)
```

### 2. 特征提取
```python
model = vit_base_entity_moe(pretrained=True, num_classes=0)
features = model.forward_features(images)  # 提取特征
```

### 3. 迁移学习
```python
# 冻结主干网络，只训练 EntityMoE 和分类头
for name, param in model.named_parameters():
    if 'entity_moe' not in name and 'head' not in name:
        param.requires_grad = False
```

### 4. 图像分割（使用 SAM）
```python
sam_model = sam_vit_base_entity_moe(pretrained=True)
output = sam_model(pixel_values=images)
vision_features = output.vision_outputs.last_hidden_state
```

## 🔧 技术特点

### 1. EntityMoEWrapper 设计

```python
class EntityMoEWrapper(nn.Module):
    def __init__(self, original_layer, dim, ...):
        self.original_layer = original_layer
        self.entity_moe = ObjectMoELayer(...)
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)  # 可学习融合
    
    def forward(self, x, *args, **kwargs):
        # 1. 原始层处理
        out = self.original_layer(x, *args, **kwargs)
        
        # 2. EntityMoE 增强
        moe_out = self.entity_moe(out)
        
        # 3. 可学习融合
        enhanced_out = out + self.alpha * moe_out
        
        return enhanced_out
```

**关键设计点**：
- ✅ 保持原始层的输入输出接口
- ✅ 自动处理不同维度的张量（3D/4D）
- ✅ 可学习的融合权重 `alpha`
- ✅ 支持返回元组的层（如注意力层）

### 2. 自适应维度处理

```python
if main_out.dim() == 3:  # (B, N, C) - Transformer
    moe_input = main_out.unsqueeze(1)  # (B, 1, N, C)
    
elif main_out.dim() == 4:  # (B, C, H, W) - CNN
    spatial_flat = main_out.view(B, C, H*W).transpose(1, 2)
    moe_input = spatial_flat.unsqueeze(1)
```

### 3. 模型信息追踪

每个模型都会记录 EntityMoE 配置：

```python
model.entity_moe_config = {
    'model_type': 'vit',
    'base_model': 'vit_base_patch16_224',
    'inject_layers': [11],
    'num_experts': 4,
    'num_experts_shared': 2,
    'expert_k': 1,
}
```

## 📊 性能考虑

### 计算开销分析

| 模型 | 基础参数 | EntityMoE 增加 | 总计 |
|------|---------|---------------|------|
| ViT-Base (1层) | ~86M | ~2-3M | ~88-89M |
| Swin-Tiny (最后stage) | ~28M | ~1-2M | ~29-30M |
| ResNet-50 (layer4) | ~25M | ~3-4M | ~28-29M |
| SAM-ViT-Base (后半) | ~90M | ~3-5M | ~93-95M |

### 优化建议

1. **减少注入层数**：
   - 简单任务：只在最后几层注入
   - 复杂任务：在关键层注入

2. **调整专家数量**：
   - `num_experts=2-4`：适合大多数任务
   - `expert_k=1`：保持稀疏性

3. **使用混合精度训练**：
   ```python
   from torch.cuda.amp import autocast
   with autocast():
       outputs = model(images)
   ```

## 🔍 测试验证

### 模块导入测试
```bash
python -c "
from models.like.entity_moe.vit_entity_moe import *
print('✓ 所有模块导入成功')
"
```

### 完整示例测试
```bash
python examples/entity_moe_example.py
```

## 📚 依赖要求

```bash
# 核心依赖
pip install torch torchvision

# ViT, Swin, ResNet 支持
pip install timm

# SAM 支持
pip install transformers

# 可选：加速训练
pip install accelerate
```

## 🚀 快速开始

### 最简示例

```python
# 1. 导入
from models.like.entity_moe.vit_entity_moe import vit_base_entity_moe
import torch

# 2. 创建模型
model = vit_base_entity_moe(
    pretrained=True,      # 使用预训练权重
    num_classes=1000,     # ImageNet 分类
    inject_layers='last'  # 只在最后一层注入
)

# 3. 推理
x = torch.randn(1, 3, 224, 224)
output = model(x)
print(output.shape)  # (1, 1000)
```

## 🎓 进阶用法

### 1. 多阶段注入

```python
# 在多个位置注入 EntityMoE
model = create_vit_entity_moe(
    model_name='vit_base_patch16_224',
    inject_layers=[6, 9, 11],  # 在浅、中、深层都注入
)
```

### 2. 差异化学习率

```python
optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': 1e-5},
    {'params': entitymoe_params, 'lr': 1e-4},
    {'params': head_params, 'lr': 1e-3},
])
```

### 3. 动态专家数量

```python
# 浅层用少量专家，深层用更多专家
# 需要手动实现，但框架支持
```

## 📝 注意事项

1. **内存占用**：EntityMoE 会增加显存占用，建议从少量层开始
2. **训练时间**：MoE 会增加训练时间，但可以提升模型容量
3. **预训练权重**：首次使用会自动下载，需要良好的网络连接
4. **兼容性**：确保 timm 和 transformers 版本较新

## 🔗 相关资源

- **EntityMoE 论文**：[待补充]
- **timm 文档**：https://github.com/huggingface/pytorch-image-models
- **transformers 文档**：https://huggingface.co/docs/transformers
- **SAM 论文**：https://arxiv.org/abs/2304.02643

## 🤝 贡献

欢迎提出问题和改进建议！

## 📄 许可

遵循项目主许可证。

---

**最后更新**: 2025-10-28
**版本**: 1.0.0
**状态**: ✅ 稳定可用
