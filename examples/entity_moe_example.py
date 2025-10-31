#!/usr/bin/env python
"""
Vision Models with EntityMoE 使用示例

展示如何使用基于 timm 和 transformers 的 EntityMoE 视觉模型
"""

import sys
sys.path.insert(0, 'src')

import torch
import torch.nn as nn


def example_vit():
    """ViT with EntityMoE 示例"""
    print("\n" + "="*70)
    print("示例 1: ViT with EntityMoE")
    print("="*70)
    
    from models.like.entity_moe.vit_entity_moe import vit_base_entity_moe, print_model_info
    
    # 创建模型（不加载预训练权重以加快演示）
    print("\n创建 ViT-Base 模型，在最后一层注入 EntityMoE...")
    model = vit_base_entity_moe(
        pretrained=False,  # 演示时不加载预训练权重
        num_classes=1000,
        num_experts=4,
        num_experts_shared=2,
        expert_k=1,
        inject_layers='last'  # 只在最后一层注入
    )
    
    # 打印模型配置
    print_model_info(model)
    
    # 前向传播
    print("\n进行前向传播...")
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    
    print(f"✓ 输入形状: {x.shape}")
    print(f"✓ 输出形状: {output.shape}")
    print(f"✓ 模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")


def example_swin():
    """Swin Transformer with EntityMoE 示例"""
    print("\n" + "="*70)
    print("示例 2: Swin Transformer with EntityMoE")
    print("="*70)
    
    from models.like.entity_moe.vit_entity_moe import swin_tiny_entity_moe, print_model_info
    
    # 创建模型
    print("\n创建 Swin-Tiny 模型，在最后一个 stage 注入 EntityMoE...")
    model = swin_tiny_entity_moe(
        pretrained=False,
        num_classes=1000,
        num_experts=4,
        num_experts_shared=2,
        expert_k=1,
        inject_layers='last_stage'
    )
    
    # 打印模型配置
    print_model_info(model)
    
    # 前向传播
    print("\n进行前向传播...")
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    
    print(f"✓ 输入形状: {x.shape}")
    print(f"✓ 输出形状: {output.shape}")
    print(f"✓ 模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")


def example_resnet():
    """ResNet with EntityMoE 示例"""
    print("\n" + "="*70)
    print("示例 3: ResNet with EntityMoE")
    print("="*70)
    
    from models.like.entity_moe.vit_entity_moe import resnet50_entity_moe, print_model_info
    
    # 创建模型
    print("\n创建 ResNet-50 模型，在 layer4 注入 EntityMoE...")
    model = resnet50_entity_moe(
        pretrained=False,
        num_classes=1000,
        num_experts=4,
        num_experts_shared=2,
        expert_k=1,
        inject_layers='layer4'
    )
    
    # 打印模型配置
    print_model_info(model)
    
    # 前向传播
    print("\n进行前向传播...")
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    
    print(f"✓ 输入形状: {x.shape}")
    print(f"✓ 输出形状: {output.shape}")
    print(f"✓ 模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")


def example_sam():
    """SAM with EntityMoE 示例"""
    print("\n" + "="*70)
    print("示例 4: SAM with EntityMoE")
    print("="*70)
    
    from models.like.entity_moe.vit_entity_moe import sam_vit_base_entity_moe, print_model_info
    
    # 创建模型
    print("\n创建 SAM ViT-Base 模型，在后半部分层注入 EntityMoE...")
    model = sam_vit_base_entity_moe(
        pretrained=False,
        num_experts=4,
        num_experts_shared=2,
        expert_k=1,
        inject_layers='last_half'
    )
    
    # 打印模型配置
    print_model_info(model)
    
    # 前向传播
    print("\n进行前向传播...")
    x = torch.randn(1, 3, 1024, 1024)
    with torch.no_grad():
        output = model(pixel_values=x)
    
    print(f"✓ 输入形状: {x.shape}")
    print(f"✓ Vision features 形状: {output.vision_outputs.last_hidden_state.shape}")
    print(f"✓ 模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")


def example_custom_inject():
    """自定义注入层示例"""
    print("\n" + "="*70)
    print("示例 5: 自定义注入层位置")
    print("="*70)
    
    from models.like.entity_moe.vit_entity_moe import create_vit_entity_moe
    
    # 在指定的层注入 EntityMoE
    print("\n创建 ViT 模型，在第 9, 10, 11 层注入 EntityMoE...")
    model = create_vit_entity_moe(
        model_name='vit_base_patch16_224',
        pretrained=False,
        num_classes=1000,
        inject_layers=[9, 10, 11]  # 指定层索引
    )
    
    from models.like.entity_moe.vit_entity_moe import print_model_info
    print_model_info(model)
    
    # 前向传播
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    
    print(f"\n✓ 输出形状: {output.shape}")


def example_training():
    """训练示例"""
    print("\n" + "="*70)
    print("示例 6: 训练示例（伪代码）")
    print("="*70)
    
    from models.like.entity_moe.vit_entity_moe import vit_base_entity_moe
    
    # 创建模型
    print("\n创建模型...")
    model = vit_base_entity_moe(
        pretrained=False,  # 实际训练时设为 True
        num_classes=10,    # 假设是 10 分类任务
        inject_layers='last'
    )
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    print("✓ 模型创建完成")
    print("✓ 损失函数: CrossEntropyLoss")
    print("✓ 优化器: AdamW (lr=1e-4)")
    
    # 模拟一个训练步骤
    print("\n模拟训练步骤...")
    model.train()
    
    # 模拟数据
    images = torch.randn(4, 3, 224, 224)
    labels = torch.randint(0, 10, (4,))
    
    # 前向传播
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"✓ 训练步骤完成")
    print(f"✓ Loss: {loss.item():.4f}")
    
    # 评估模式
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
    
    print(f"✓ 推理完成")
    print(f"✓ 预测结果: {predictions}")


def example_fine_tuning():
    """微调示例"""
    print("\n" + "="*70)
    print("示例 7: 微调技巧")
    print("="*70)
    
    from models.like.entity_moe.vit_entity_moe import vit_base_entity_moe
    
    # 创建模型
    print("\n创建模型...")
    model = vit_base_entity_moe(
        pretrained=False,
        num_classes=100,
        inject_layers='last'
    )
    
    # 方法 1: 冻结预训练的主干网络，只训练 EntityMoE 和分类头
    print("\n方法 1: 冻结主干网络，只训练 EntityMoE 部分")
    frozen_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        if 'entity_moe' not in name and 'head' not in name:
            param.requires_grad = False
            frozen_params += param.numel()
        else:
            trainable_params += param.numel()
    
    print(f"✓ 冻结参数: {frozen_params/1e6:.2f}M")
    print(f"✓ 可训练参数: {trainable_params/1e6:.2f}M")
    
    # 方法 2: 使用不同的学习率
    print("\n方法 2: 使用差异化学习率")
    backbone_params = []
    entitymoe_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'entity_moe' in name:
            entitymoe_params.append(param)
        elif 'head' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},      # 主干网络用小学习率
        {'params': entitymoe_params, 'lr': 1e-4},     # EntityMoE 用中等学习率
        {'params': head_params, 'lr': 1e-3},          # 分类头用大学习率
    ])
    
    print(f"✓ 主干网络学习率: 1e-5")
    print(f"✓ EntityMoE 学习率: 1e-4")
    print(f"✓ 分类头学习率: 1e-3")


def main():
    print("\n" + "="*70)
    print("Vision Models with EntityMoE - 使用示例")
    print("="*70)
    
    try:
        # 运行各个示例
        example_vit()
        example_swin()
        example_resnet()
        
        # SAM 示例（需要 transformers 库）
        try:
            example_sam()
        except Exception as e:
            print(f"\n⚠️ SAM 示例跳过: {e}")
        
        example_custom_inject()
        example_training()
        example_fine_tuning()
        
        print("\n" + "="*70)
        print("✅ 所有示例运行完成！")
        print("="*70)
        print("\n💡 提示:")
        print("  - 详细文档请查看: src/models/like/entity_moe/README.md")
        print("  - 实际使用时建议设置 pretrained=True 加载预训练权重")
        print("  - 根据任务复杂度调整 inject_layers 参数")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

