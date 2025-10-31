"""
Models模块基础使用示例

本示例展示如何使用NeuroTrain的模型模块加载和使用各种模型。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from src.models import get_model
from torchinfo import summary


def example_1_unet_model():
    """示例1: UNet模型"""
    print("=" * 80)
    print("示例1: UNet分割模型")
    print("=" * 80)
    
    config = {
        'n_channels': 3,    # 输入通道数（RGB图像）
        'n_classes': 2,     # 输出类别数（二分类分割）
        'bilinear': False   # 是否使用双线性插值
    }
    
    model = get_model('unet', config)
    print(f"模型类型: {type(model).__name__}")
    
    # 模型摘要
    print("\n模型结构摘要:")
    summary(model, input_size=(1, 3, 512, 512), device='cpu')
    
    # 测试前向传播
    dummy_input = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"\n输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    
    return model


def example_2_torchvision_model():
    """示例2: TorchVision模型"""
    print("\n" + "=" * 80)
    print("示例2: TorchVision ResNet18")
    print("=" * 80)
    
    config = {
        'arch': 'resnet18',     # 模型架构
        'pretrained': True,     # 使用预训练权重
        'n_classes': 10,        # CIFAR-10的10个类别
        'n_channels': 3         # RGB图像
    }
    
    model = get_model('torchvision', config)
    print(f"模型类型: {type(model).__name__}")
    
    # 模型摘要
    print("\n模型结构摘要:")
    summary(model, input_size=(1, 3, 224, 224), device='cpu')
    
    # 测试前向传播
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"\n输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    
    return model


def example_3_timm_model():
    """示例3: TIMM模型"""
    print("\n" + "=" * 80)
    print("示例3: TIMM EfficientNet")
    print("=" * 80)
    
    config = {
        'model_name': 'efficientnet_b0',  # TIMM模型名称
        'pretrained': True,                # 使用预训练权重
        'n_classes': 100,                  # CIFAR-100的100个类别
        'n_channels': 3                    # RGB图像
    }
    
    try:
        model = get_model('timm', config)
        print(f"模型类型: {type(model).__name__}")
        
        # 模型摘要
        print("\n模型结构摘要:")
        summary(model, input_size=(1, 3, 224, 224), device='cpu')
        
        # 测试前向传播
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"\n输入形状: {dummy_input.shape}")
        print(f"输出形状: {output.shape}")
        
        return model
    except ImportError:
        print("TIMM库未安装，跳过此示例")
        print("安装命令: pip install timm")
        return None


def example_4_clip_model():
    """示例4: CLIP多模态模型"""
    print("\n" + "=" * 80)
    print("示例4: CLIP多模态模型")
    print("=" * 80)
    
    config = {
        'model_name': 'openai/clip-vit-base-patch32',
        'cache_dir': 'cache/models/pretrained',
        'device': 'cpu',
        'dtype': torch.float32
    }
    
    try:
        model = get_model('clip', config)
        print(f"模型类型: {type(model).__name__}")
        
        # CLIP模型有图像编码器和文本编码器
        print("\nCLIP模型组件:")
        print(f"  - 图像编码器: {type(model.vision_model).__name__}")
        print(f"  - 文本编码器: {type(model.text_model).__name__}")
        
        # 测试图像编码
        dummy_image = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            image_features = model.get_image_features(dummy_image)
        print(f"\n图像特征形状: {image_features.shape}")
        
        return model
    except Exception as e:
        print(f"加载CLIP模型时出错: {e}")
        print("请确保已安装transformers库")
        return None


def example_5_model_comparison():
    """示例5: 模型对比"""
    print("\n" + "=" * 80)
    print("示例5: 不同模型的参数量和计算量对比")
    print("=" * 80)
    
    models_configs = [
        ('UNet', 'unet', {'n_channels': 3, 'n_classes': 2}),
        ('ResNet18', 'torchvision', {'arch': 'resnet18', 'pretrained': False, 'n_classes': 10}),
        ('ResNet50', 'torchvision', {'arch': 'resnet50', 'pretrained': False, 'n_classes': 10}),
    ]
    
    print(f"\n{'Model':<20} {'Parameters':<15} {'Input Size':<20} {'Output Size':<15}")
    print("-" * 70)
    
    for name, model_type, config in models_configs:
        try:
            model = get_model(model_type, config)
            
            # 计算参数量
            num_params = sum(p.numel() for p in model.parameters())
            
            # 测试输入输出
            if 'unet' in model_type:
                test_input = torch.randn(1, 3, 256, 256)
            else:
                test_input = torch.randn(1, 3, 224, 224)
            
            with torch.no_grad():
                output = model(test_input)
            
            print(f"{name:<20} {num_params:>12,}   {str(tuple(test_input.shape)):<20} {str(tuple(output.shape)):<15}")
            
        except Exception as e:
            print(f"{name:<20} Error: {e}")


def example_6_model_customization():
    """示例6: 模型自定义"""
    print("\n" + "=" * 80)
    print("示例6: 自定义模型修改")
    print("=" * 80)
    
    # 加载基础ResNet模型
    config = {
        'arch': 'resnet18',
        'pretrained': True,
        'n_classes': 1000,  # 先用ImageNet的类别数
    }
    model = get_model('torchvision', config)
    
    print("原始模型最后一层:")
    print(model.fc)
    
    # 修改分类头用于10类分类任务
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 10)
    )
    
    print("\n修改后的分类头:")
    print(model.fc)
    
    # 冻结除分类头外的所有层
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
    
    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"冻结参数量: {total_params - trainable_params:,}")
    
    # 测试修改后的模型
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"\n输出形状: {output.shape}")


def example_7_save_load_model():
    """示例7: 保存和加载模型"""
    print("\n" + "=" * 80)
    print("示例7: 模型保存和加载")
    print("=" * 80)
    
    # 创建模型
    config = {
        'n_channels': 3,
        'n_classes': 2
    }
    model = get_model('unet', config)
    
    # 保存路径
    save_dir = Path('examples/output/models')
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / 'unet_example.pth'
    
    # 保存模型
    print(f"\n保存模型到: {model_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'model_type': 'unet'
    }, model_path)
    
    # 加载模型
    print(f"从文件加载模型...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 重新创建模型
    loaded_model = get_model(checkpoint['model_type'], checkpoint['config'])
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    
    print("模型加载成功！")
    
    # 验证模型一致性
    test_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output1 = model(test_input)
        output2 = loaded_model(test_input)
    
    print(f"\n输出差异: {torch.abs(output1 - output2).max().item():.6e}")


def main():
    """运行所有示例"""
    print("NeuroTrain Models模块使用示例")
    print("=" * 80)
    
    # 创建输出目录
    output_dir = Path('examples/output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 运行各个示例
        example_1_unet_model()
        example_2_torchvision_model()
        example_3_timm_model()
        example_4_clip_model()
        example_5_model_comparison()
        example_6_model_customization()
        example_7_save_load_model()
        
        print("\n" + "=" * 80)
        print("所有示例运行完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

