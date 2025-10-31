"""
Vision Models with EntityMoE Integration

基于 timm 和 transformers 库的预训练模型，集成 EntityMoE 增强特征表示。

支持的模型：
1. ViT (Vision Transformer) with EntityMoE - 通过 timm 获取
2. Swin Transformer with EntityMoE - 通过 timm 获取  
3. ResNet with EntityMoE - 通过 timm 获取
4. SAM (Segment Anything Model) Encoder with EntityMoE - 通过 transformers 获取

使用示例：
    # ViT
    model = create_vit_entity_moe('vit_base_patch16_224', num_classes=1000)
    
    # Swin Transformer
    model = create_swin_entity_moe('swin_tiny_patch4_window7_224', num_classes=1000)
    
    # ResNet
    model = create_resnet_entity_moe('resnet50', num_classes=1000)
    
    # SAM
    model = create_sam_entity_moe('facebook/sam-vit-base')
"""

import torch
import torch.nn as nn
from .EntityMoe import ObjectMoELayer
import warnings

import timm
from transformers import SamModel, SamVisionConfig

# ==================== EntityMoE Wrapper Modules ====================

class EntityMoEWrapper(nn.Module):
    """
    用于在现有模型层之后添加 EntityMoE 处理的包装器
    """
    def __init__(
        self,
        original_layer,
        dim,
        num_experts=4,
        num_experts_shared=2,
        expert_k=1,
        dropout=0.1,
        mlp_ratio=4.0,
        num_heads=8,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.entity_moe = ObjectMoELayer(
            input_dim=dim,
            output_dim=dim,
            num_experts=num_experts,
            num_experts_shared=num_experts_shared,
            expert_hidden_dim=int(dim * mlp_ratio),
            k=expert_k,
            sparse=True,
            dropout=dropout,
            ffn_hidden_dim=int(dim * mlp_ratio),
            cross_attn_heads=num_heads,
            cross_attn_dropout=dropout,
        )
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)  # 可学习的融合权重
    
    def forward(self, x, *args, **kwargs):
        # 调用原始层
        out = self.original_layer(x, *args, **kwargs)
        
        # 如果原始层返回元组（如注意力层），只处理主输出
        if isinstance(out, tuple):
            main_out = out[0]
            extra_outputs = out[1:]
        else:
            main_out = out
            extra_outputs = None
        
        # 应用 EntityMoE
        if main_out.dim() == 3:  # (B, N, C)
            B, N, C = main_out.shape
            # 将序列视为 objects
            moe_input = main_out.unsqueeze(1)  # (B, 1, N, C)
            moe_out = self.entity_moe(moe_input)  # (B, 1, N, C)
            moe_out = moe_out.squeeze(1)  # (B, N, C)
            # 使用可学习的融合权重
            enhanced_out = main_out + self.alpha * moe_out
        elif main_out.dim() == 4:  # (B, C, H, W)
            B, C, H, W = main_out.shape
            # 展平空间维度
            spatial_flat = main_out.view(B, C, H * W).transpose(1, 2)  # (B, H*W, C)
            moe_input = spatial_flat.unsqueeze(1)  # (B, 1, H*W, C)
            moe_out = self.entity_moe(moe_input)  # (B, 1, H*W, C)
            moe_out = moe_out.squeeze(1).transpose(1, 2).view(B, C, H, W)
            enhanced_out = main_out + self.alpha * moe_out
        else:
            enhanced_out = main_out
        
        # 返回与原始层相同的格式
        if extra_outputs is not None:
            return (enhanced_out,) + extra_outputs
        else:
            return enhanced_out


# ==================== ViT with EntityMoE (from timm) ====================
def create_vit_entity_moe(
    model_name='vit_base_patch16_224',
    pretrained=True,
    num_classes=1000,
    num_experts=4,
    num_experts_shared=2,
    expert_k=1,
    dropout=0.1,
    mlp_ratio=4.0,
    inject_layers='all',  # 'all', 'last', or list of layer indices
    **kwargs
):
    """
    创建带有 EntityMoE 的 ViT 模型
    
    Args:
        model_name: timm 模型名称，如 'vit_base_patch16_224', 'vit_large_patch16_224'
        pretrained: 是否加载预训练权重
        num_classes: 分类类别数
        num_experts: MoE 专家数量
        num_experts_shared: 共享专家数量
        expert_k: 每次选择的专家数量
        dropout: Dropout 比例
        mlp_ratio: MLP 隐藏层倍数
        inject_layers: 在哪些层注入 EntityMoE
            - 'all': 所有 Transformer blocks
            - 'last': 只在最后一层
            - list: 指定层的索引列表，如 [6, 7, 8, 9, 10, 11]
    
    Returns:
        带有 EntityMoE 的 ViT 模型
    """
    # 创建基础模型
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, **kwargs)
    
    # 获取模型配置
    embed_dim = model.embed_dim
    num_heads = model.blocks[0].attn.num_heads if hasattr(model.blocks[0], 'attn') else 8
    
    # 确定要注入 EntityMoE 的层
    total_layers = len(model.blocks)
    if inject_layers == 'all':
        layer_indices = list(range(total_layers))
    elif inject_layers == 'last':
        layer_indices = [total_layers - 1]
    elif isinstance(inject_layers, (list, tuple)):
        layer_indices = inject_layers
    else:
        raise ValueError(f"Invalid inject_layers: {inject_layers}")
    
    # 为指定的层添加 EntityMoE
    for idx in layer_indices:
        if idx < 0 or idx >= total_layers:
            warnings.warn(f"Layer index {idx} out of range [0, {total_layers}), skipping.")
            continue
        
        original_block = model.blocks[idx]
        model.blocks[idx] = EntityMoEWrapper(
            original_layer=original_block,
            dim=embed_dim,
            num_experts=num_experts,
            num_experts_shared=num_experts_shared,
            expert_k=expert_k,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
        )
    
    # 添加模型信息
    model.entity_moe_config = {
        'model_type': 'vit',
        'base_model': model_name,
        'inject_layers': layer_indices,
        'num_experts': num_experts,
        'num_experts_shared': num_experts_shared,
        'expert_k': expert_k,
    }
    
    return model


# ==================== Swin Transformer with EntityMoE (from timm) ====================
def create_swin_entity_moe(
    model_name='swin_tiny_patch4_window7_224',
    pretrained=True,
    num_classes=1000,
    num_experts=4,
    num_experts_shared=2,
    expert_k=1,
    dropout=0.1,
    mlp_ratio=4.0,
    inject_layers='last_stage',  # 'all', 'last_stage', or list of (stage_idx, block_idx)
    **kwargs
):
    """
    创建带有 EntityMoE 的 Swin Transformer 模型
    
    Args:
        model_name: timm 模型名称，如 'swin_tiny_patch4_window7_224', 'swin_base_patch4_window7_224'
        pretrained: 是否加载预训练权重
        num_classes: 分类类别数
        num_experts: MoE 专家数量
        num_experts_shared: 共享专家数量
        expert_k: 每次选择的专家数量
        dropout: Dropout 比例
        mlp_ratio: MLP 隐藏层倍数
        inject_layers: 在哪些层注入 EntityMoE
            - 'all': 所有 blocks
            - 'last_stage': 只在最后一个 stage
            - list: 指定 (stage_idx, block_idx) 的列表
    
    Returns:
        带有 EntityMoE 的 Swin Transformer 模型
    """
    # 创建基础模型
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, **kwargs)
    
    # Swin 的层组织方式：model.layers[stage_idx].blocks[block_idx]
    num_stages = len(model.layers)
    
    # 确定要注入 EntityMoE 的层
    if inject_layers == 'all':
        layer_positions = []
        for stage_idx in range(num_stages):
            stage = model.layers[stage_idx]
            if hasattr(stage, 'blocks'):
                for block_idx in range(len(stage.blocks)):
                    layer_positions.append((stage_idx, block_idx))
    elif inject_layers == 'last_stage':
        last_stage_idx = num_stages - 1
        stage = model.layers[last_stage_idx]
        if hasattr(stage, 'blocks'):
            layer_positions = [(last_stage_idx, i) for i in range(len(stage.blocks))]
        else:
            layer_positions = []
    elif isinstance(inject_layers, (list, tuple)):
        layer_positions = inject_layers
    else:
        raise ValueError(f"Invalid inject_layers: {inject_layers}")
    
    # 为指定的层添加 EntityMoE
    for stage_idx, block_idx in layer_positions:
        if stage_idx >= num_stages:
            warnings.warn(f"Stage index {stage_idx} out of range, skipping.")
            continue
        
        stage = model.layers[stage_idx]
        if not hasattr(stage, 'blocks') or block_idx >= len(stage.blocks):
            warnings.warn(f"Block index {block_idx} out of range in stage {stage_idx}, skipping.")
            continue
        
        original_block = stage.blocks[block_idx]
        # 获取该层的维度
        dim = original_block.dim if hasattr(original_block, 'dim') else model.num_features
        num_heads = original_block.attn.num_heads if hasattr(original_block, 'attn') else 4
        
        stage.blocks[block_idx] = EntityMoEWrapper(
            original_layer=original_block,
            dim=dim,
            num_experts=num_experts,
            num_experts_shared=num_experts_shared,
            expert_k=expert_k,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
        )
    
    # 添加模型信息
    model.entity_moe_config = {
        'model_type': 'swin',
        'base_model': model_name,
        'inject_layers': layer_positions,
        'num_experts': num_experts,
        'num_experts_shared': num_experts_shared,
        'expert_k': expert_k,
    }
    
    return model


# ==================== ResNet with EntityMoE (from timm) ====================
def create_resnet_entity_moe(
    model_name='resnet50',
    pretrained=True,
    num_classes=1000,
    num_experts=4,
    num_experts_shared=2,
    expert_k=1,
    dropout=0.1,
    mlp_ratio=2.0,
    inject_layers='layer4',  # 'all', 'layer3', 'layer4', or list of (layer_name, block_idx)
    **kwargs
):
    """
    创建带有 EntityMoE 的 ResNet 模型
    
    Args:
        model_name: timm 模型名称，如 'resnet50', 'resnet101', 'resnet152'
        pretrained: 是否加载预训练权重
        num_classes: 分类类别数
        num_experts: MoE 专家数量
        num_experts_shared: 共享专家数量
        expert_k: 每次选择的专家数量
        dropout: Dropout 比例
        mlp_ratio: MLP 隐藏层倍数
        inject_layers: 在哪些层注入 EntityMoE
            - 'all': 所有 layers
            - 'layer3': 只在 layer3
            - 'layer4': 只在 layer4
            - list: 指定 (layer_name, block_idx) 的列表，如 [('layer3', 0), ('layer4', 0)]
    
    Returns:
        带有 EntityMoE 的 ResNet 模型
    """
    # 创建基础模型
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, **kwargs)
    
    # ResNet 的层组织：model.layer1, model.layer2, model.layer3, model.layer4
    layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
    
    # 确定要注入 EntityMoE 的层
    if inject_layers == 'all':
        layer_positions = []
        for layer_name in layer_names:
            layer = getattr(model, layer_name)
            for block_idx in range(len(layer)):
                layer_positions.append((layer_name, block_idx))
    elif inject_layers in layer_names:
        layer = getattr(model, inject_layers)
        layer_positions = [(inject_layers, i) for i in range(len(layer))]
    elif isinstance(inject_layers, (list, tuple)):
        layer_positions = inject_layers
    else:
        raise ValueError(f"Invalid inject_layers: {inject_layers}")
    
    # 为指定的层添加 EntityMoE
    for layer_name, block_idx in layer_positions:
        if layer_name not in layer_names:
            warnings.warn(f"Layer name {layer_name} invalid, skipping.")
            continue
        
        layer = getattr(model, layer_name)
        if block_idx >= len(layer):
            warnings.warn(f"Block index {block_idx} out of range in {layer_name}, skipping.")
            continue
        
        original_block = layer[block_idx]
        
        # 获取该 block 的输出通道数
        if hasattr(original_block, 'conv3'):  # Bottleneck
            dim = original_block.conv3.out_channels
        elif hasattr(original_block, 'conv2'):  # BasicBlock
            dim = original_block.conv2.out_channels
        else:
            warnings.warn(f"Cannot determine dim for {layer_name}[{block_idx}], skipping.")
            continue
        
        layer[block_idx] = EntityMoEWrapper(
            original_layer=original_block,
            dim=dim,
            num_experts=num_experts,
            num_experts_shared=num_experts_shared,
            expert_k=expert_k,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            num_heads=4,  # ResNet 使用固定的 heads 数
        )
    
    # 添加模型信息
    model.entity_moe_config = {
        'model_type': 'resnet',
        'base_model': model_name,
        'inject_layers': layer_positions,
        'num_experts': num_experts,
        'num_experts_shared': num_experts_shared,
        'expert_k': expert_k,
    }
    
    return model


# ==================== SAM Encoder with EntityMoE (from transformers) ====================

def create_sam_entity_moe(
    model_name='facebook/sam-vit-base',
    pretrained=True,
    num_experts=4,
    num_experts_shared=2,
    expert_k=1,
    dropout=0.1,
    mlp_ratio=4.0,
    inject_layers='last_half',  # 'all', 'last_half', or list of layer indices
    **kwargs
):
    """
    创建带有 EntityMoE 的 SAM 模型
    
    Args:
        model_name: transformers 模型名称，如 'facebook/sam-vit-base', 'facebook/sam-vit-large', 'facebook/sam-vit-huge'
        pretrained: 是否加载预训练权重
        num_experts: MoE 专家数量
        num_experts_shared: 共享专家数量
        expert_k: 每次选择的专家数量
        dropout: Dropout 比例
        mlp_ratio: MLP 隐藏层倍数
        inject_layers: 在哪些层注入 EntityMoE
            - 'all': 所有 encoder layers
            - 'last_half': 只在后半部分 layers
            - list: 指定层的索引列表
    
    Returns:
        带有 EntityMoE 的 SAM 模型（只返回 vision encoder）
    """
    # 加载 SAM 模型
    if pretrained:
        sam_model = SamModel.from_pretrained(model_name, **kwargs)
    else:
        config = SamVisionConfig()
        sam_model = SamModel(config)
    
    # 获取 vision encoder
    vision_encoder = sam_model.vision_encoder
    
    # 获取配置
    hidden_size = vision_encoder.config.hidden_size
    num_attention_heads = vision_encoder.config.num_attention_heads
    
    # 获取 encoder layers
    encoder_layers = vision_encoder.layers
    total_layers = len(encoder_layers)
    
    # 确定要注入 EntityMoE 的层
    if inject_layers == 'all':
        layer_indices = list(range(total_layers))
    elif inject_layers == 'last_half':
        layer_indices = list(range(total_layers // 2, total_layers))
    elif isinstance(inject_layers, (list, tuple)):
        layer_indices = inject_layers
    else:
        raise ValueError(f"Invalid inject_layers: {inject_layers}")
    
    # 为指定的层添加 EntityMoE
    for idx in layer_indices:
        if idx < 0 or idx >= total_layers:
            warnings.warn(f"Layer index {idx} out of range [0, {total_layers}), skipping.")
            continue
        
        original_layer = encoder_layers[idx]
        encoder_layers[idx] = EntityMoEWrapper(
            original_layer=original_layer,
            dim=hidden_size,
            num_experts=num_experts,
            num_experts_shared=num_experts_shared,
            expert_k=expert_k,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            num_heads=num_attention_heads,
        )
    
    # 添加模型信息
    vision_encoder.entity_moe_config = {
        'model_type': 'sam',
        'base_model': model_name,
        'inject_layers': layer_indices,
        'num_experts': num_experts,
        'num_experts_shared': num_experts_shared,
        'expert_k': expert_k,
    }
    
    # 返回完整的 SAM 模型或只返回 vision encoder
    # 这里返回完整模型，用户可以通过 model.vision_encoder 访问
    return sam_model


# ==================== Convenience Functions ====================

def vit_base_entity_moe(pretrained=True, num_classes=1000, **kwargs):
    """ViT-Base with EntityMoE"""
    return create_vit_entity_moe(
        model_name='vit_base_patch16_224',
        pretrained=pretrained,
        num_classes=num_classes,
        **kwargs
    )


def vit_large_entity_moe(pretrained=True, num_classes=1000, **kwargs):
    """ViT-Large with EntityMoE"""
    return create_vit_entity_moe(
        model_name='vit_large_patch16_224',
        pretrained=pretrained,
        num_classes=num_classes,
        **kwargs
    )


def swin_tiny_entity_moe(pretrained=True, num_classes=1000, **kwargs):
    """Swin-Tiny with EntityMoE"""
    return create_swin_entity_moe(
        model_name='swin_tiny_patch4_window7_224',
        pretrained=pretrained,
        num_classes=num_classes,
        **kwargs
    )


def swin_base_entity_moe(pretrained=True, num_classes=1000, **kwargs):
    """Swin-Base with EntityMoE"""
    return create_swin_entity_moe(
        model_name='swin_base_patch4_window7_224',
        pretrained=pretrained,
        num_classes=num_classes,
        **kwargs
    )


def resnet50_entity_moe(pretrained=True, num_classes=1000, **kwargs):
    """ResNet-50 with EntityMoE"""
    return create_resnet_entity_moe(
        model_name='resnet50',
        pretrained=pretrained,
        num_classes=num_classes,
        **kwargs
    )


def resnet101_entity_moe(pretrained=True, num_classes=1000, **kwargs):
    """ResNet-101 with EntityMoE"""
    return create_resnet_entity_moe(
        model_name='resnet101',
        pretrained=pretrained,
        num_classes=num_classes,
        **kwargs
    )


def sam_vit_base_entity_moe(pretrained=True, **kwargs):
    """SAM ViT-Base with EntityMoE"""
    return create_sam_entity_moe(
        model_name='facebook/sam-vit-base',
        pretrained=pretrained,
        **kwargs
    )


def sam_vit_large_entity_moe(pretrained=True, **kwargs):
    """SAM ViT-Large with EntityMoE"""
    return create_sam_entity_moe(
        model_name='facebook/sam-vit-large',
        pretrained=pretrained,
        **kwargs
    )


def sam_vit_huge_entity_moe(pretrained=True, **kwargs):
    """SAM ViT-Huge with EntityMoE"""
    return create_sam_entity_moe(
        model_name='facebook/sam-vit-huge',
        pretrained=pretrained,
        **kwargs
    )

if __name__ == "__main__":
    # ==================== Model Information ====================
    def print_model_info(model):
        """打印模型的 EntityMoE 配置信息"""
        if hasattr(model, 'entity_moe_config'):
            config = model.entity_moe_config
            print("=" * 60)
            print("EntityMoE Configuration")
            print("=" * 60)
            for key, value in config.items():
                print(f"{key}: {value}")
            print("=" * 60)
        elif hasattr(model, 'vision_encoder') and hasattr(model.vision_encoder, 'entity_moe_config'):
            config = model.vision_encoder.entity_moe_config
            print("=" * 60)
            print("EntityMoE Configuration (Vision Encoder)")
            print("=" * 60)
            for key, value in config.items():
                print(f"{key}: {value}")
            print("=" * 60)
        else:
            print("No EntityMoE configuration found in this model.")
    print("Testing Vision Models with EntityMoE\n")
    
    # Test ViT
    print("=" * 60)
    print("Testing ViT with EntityMoE...")
    print("=" * 60)
    try:
        vit = vit_base_entity_moe(pretrained=False, num_classes=1000)
        x = torch.randn(2, 3, 224, 224)
        out = vit(x)
        print(f"✓ ViT output shape: {out.shape}")
        print_model_info(vit)
        total_params = sum(p.numel() for p in vit.parameters()) / 1e6
        print(f"✓ Total parameters: {total_params:.2f}M\n")
    except Exception as e:
        print(f"✗ ViT test failed: {e}\n")
    
    # Test Swin
    print("=" * 60)
    print("Testing Swin Transformer with EntityMoE...")
    print("=" * 60)
    try:
        swin = swin_tiny_entity_moe(pretrained=False, num_classes=1000)
        x = torch.randn(2, 3, 224, 224)
        out = swin(x)
        print(f"✓ Swin output shape: {out.shape}")
        print_model_info(swin)
        total_params = sum(p.numel() for p in swin.parameters()) / 1e6
        print(f"✓ Total parameters: {total_params:.2f}M\n")
    except Exception as e:
        print(f"✗ Swin test failed: {e}\n")
    
    # Test ResNet
    print("=" * 60)
    print("Testing ResNet with EntityMoE...")
    print("=" * 60)
    try:
        resnet = resnet50_entity_moe(pretrained=False, num_classes=1000)
        x = torch.randn(2, 3, 224, 224)
        out = resnet(x)
        print(f"✓ ResNet output shape: {out.shape}")
        print_model_info(resnet)
        total_params = sum(p.numel() for p in resnet.parameters()) / 1e6
        print(f"✓ Total parameters: {total_params:.2f}M\n")
    except Exception as e:
        print(f"✗ ResNet test failed: {e}\n")
    
    # Test SAM
    print("=" * 60)
    print("Testing SAM with EntityMoE...")
    print("=" * 60)
    try:
        sam = sam_vit_base_entity_moe(pretrained=False)
        x = torch.randn(1, 3, 1024, 1024)
        # SAM 需要 pixel_values 参数
        out = sam(pixel_values=x)
        print(f"✓ SAM vision features shape: {out.vision_outputs.last_hidden_state.shape}")
        print_model_info(sam)
        total_params = sum(p.numel() for p in sam.parameters()) / 1e6
        print(f"✓ Total parameters: {total_params:.2f}M\n")
    except Exception as e:
        print(f"✗ SAM test failed: {e}\n")
    
    print("=" * 60)
    print("Testing completed!")
    print("=" * 60)
