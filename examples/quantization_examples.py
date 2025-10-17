#!/usr/bin/env python3
"""
量化工具使用示例
展示如何在NeuroTrain框架中使用量化功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加项目根目录到路径
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# 导入量化工具
from src.quantization import (
    QuantizationConfig, 
    QuantizationManager, 
    QuantizationTrainer,
    QuantizationAnalyzer,
    create_quantization_config,
    quantize_model,
    integrate_quantization_with_training,
    analyze_quantization_effectiveness
)
from src.quantization.config import (
    get_quantization_config,
    setup_quantization_from_config,
    create_quantization_config_template,
    get_recommended_quantization_method
)


def create_sample_model():
    """创建示例模型"""
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, num_classes)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x
    
    return SimpleCNN()


def create_sample_data():
    """创建示例数据"""
    # 创建随机数据
    batch_size = 32
    num_samples = 1000
    
    X = torch.randn(num_samples, 3, 32, 32)
    y = torch.randint(0, 10, (num_samples,))
    
    # 分割训练和测试数据
    train_size = int(0.8 * num_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def example_1_basic_quantization():
    """示例1: 基础量化使用"""
    logger.info("=== 示例1: 基础量化使用 ===")
    
    # 创建模型和数据
    model = create_sample_model()
    train_loader, test_loader = create_sample_data()
    
    # 创建量化配置
    quant_config = create_quantization_config(
        method="dynamic",
        dtype="qint8"
    )
    
    # 量化模型
    quantized_model, manager = quantize_model(model, quant_config)
    
    # 获取模型信息
    size_info = manager.get_model_size_info(quantized_model)
    logger.info(f"量化后模型信息: {size_info}")
    
    # 测试推理
    test_input = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        original_output = model(test_input)
        quantized_output = quantized_model(test_input)
    
    logger.info(f"原始模型输出形状: {original_output.shape}")
    logger.info(f"量化模型输出形状: {quantized_output.shape}")


def example_2_quantization_aware_training():
    """示例2: 量化感知训练"""
    logger.info("=== 示例2: 量化感知训练 ===")
    
    # 创建模型和数据
    model = create_sample_model()
    train_loader, test_loader = create_sample_data()
    
    # 创建QAT配置
    qat_config = create_quantization_config(
        method="qat",
        dtype="qint8"
    )
    
    # 创建量化训练器
    trainer = QuantizationTrainer(
        model=model,
        quantization_config=qat_config,
        output_dir="outputs/qat_example"
    )
    
    # 设置量化
    quantized_model = trainer.setup_quantization()
    
    # 设置训练组件
    optimizer = optim.Adam(quantized_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 进行量化感知训练
    trainer.train_with_quantization(
        train_loader=train_loader,
        valid_loader=test_loader,
        num_epochs=3,
        optimizer=optimizer,
        criterion=criterion,
        save_best=True
    )


def example_3_quantization_analysis():
    """示例3: 量化效果分析"""
    logger.info("=== 示例3: 量化效果分析 ===")
    
    # 创建模型和数据
    original_model = create_sample_model()
    train_loader, test_loader = create_sample_data()
    
    # 量化模型
    quant_config = create_quantization_config(method="dynamic")
    quantized_model, manager = quantize_model(original_model, quant_config)
    
    # 创建分析器
    analyzer = QuantizationAnalyzer(original_model, quantized_model)
    
    # 比较模型大小
    size_comparison = analyzer.compare_model_sizes()
    logger.info(f"模型大小比较: {size_comparison}")
    
    # 比较推理速度
    test_input = torch.randn(1, 3, 32, 32)
    speed_comparison = analyzer.compare_inference_speed(test_input, num_runs=10)
    logger.info(f"推理速度比较: {speed_comparison}")
    
    # 比较准确率
    accuracy_comparison = analyzer.compare_accuracy(test_loader)
    logger.info(f"准确率比较: {accuracy_comparison}")
    
    # 生成完整报告
    report = analyzer.generate_report(
        test_loader=test_loader,
        test_input=test_input,
        output_path="outputs/quantization_analysis"
    )
    logger.info(f"量化分析报告已保存")


def example_4_config_based_quantization():
    """示例4: 基于配置的量化"""
    logger.info("=== 示例4: 基于配置的量化 ===")
    
    # 创建配置模板
    config_template = create_quantization_config_template()
    logger.info(f"量化配置模板: {config_template}")
    
    # 获取推荐方法
    recommended_method = get_recommended_quantization_method("cnn", "inference")
    logger.info(f"CNN推理推荐量化方法: {recommended_method}")
    
    # 创建模型
    model = create_sample_model()
    
    # 模拟配置
    config = {
        "quantization": {
            "enabled": True,
            "method": "dynamic",
            "dtype": "qint8"
        }
    }
    
    # 基于配置设置量化
    quantized_model = setup_quantization_from_config(model, config)
    logger.info("基于配置的量化完成")


def example_5_different_quantization_methods():
    """示例5: 不同量化方法对比"""
    logger.info("=== 示例5: 不同量化方法对比 ===")
    
    # 创建模型和数据
    model = create_sample_model()
    train_loader, test_loader = create_sample_data()
    
    # 测试不同量化方法
    methods = ["dynamic", "static"]
    
    for method in methods:
        logger.info(f"测试量化方法: {method}")
        
        try:
            # 创建量化配置
            quant_config = create_quantization_config(method=method)
            
            # 量化模型
            quantized_model, manager = quantize_model(model, quant_config)
            
            # 获取模型信息
            size_info = manager.get_model_size_info(quantized_model)
            logger.info(f"{method}量化 - 模型大小: {size_info['model_size_mb']:.2f}MB")
            
        except Exception as e:
            logger.error(f"{method}量化失败: {e}")


def example_6_integration_with_training():
    """示例6: 与训练流程集成"""
    logger.info("=== 示例6: 与训练流程集成 ===")
    
    # 创建模型和数据
    model = create_sample_model()
    train_loader, test_loader = create_sample_data()
    
    # 模拟配置
    config = {
        "quantization": {
            "enabled": True,
            "method": "dynamic"
        },
        "output_dir": "outputs/integration_example",
        "device": "cpu"  # 使用CPU避免CUDA问题
    }
    
    # 集成量化训练
    trainer = integrate_quantization_with_training(
        model=model,
        config=config,
        quantization_method="dynamic"
    )
    
    # 设置量化
    quantized_model = trainer.setup_quantization()
    
    logger.info("量化训练集成完成")


def main():
    """运行所有示例"""
    logger.info("开始运行量化工具示例...")
    
    try:
        example_1_basic_quantization()
        example_2_quantization_aware_training()
        example_3_quantization_analysis()
        example_4_config_based_quantization()
        example_5_different_quantization_methods()
        example_6_integration_with_training()
        
        logger.info("所有示例运行完成!")
        
    except Exception as e:
        logger.error(f"示例运行出错: {e}")
        raise


if __name__ == "__main__":
    main()
