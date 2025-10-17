#!/usr/bin/env python3
"""
PTQ和QAT量化示例
展示训练后量化和量化感知训练的使用方法
"""

import sys
import os
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.quantization import QuantizationConfig, QuantizationManager, QuantizationTrainer, QuantizationAnalyzer


def create_sample_model():
    """创建示例CNN模型"""
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


def example_ptq_dynamic():
    """示例1: PTQ动态量化"""
    logger.info("=== PTQ动态量化示例 ===")
    
    # 创建模型和数据
    model = create_sample_model()
    train_loader, test_loader = create_sample_data()
    
    # 先训练模型（模拟）
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 简单训练几个epoch
    for epoch in range(3):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    # 训练完成后进行动态量化
    logger.info("开始PTQ动态量化...")
    config = QuantizationConfig(method="dynamic", dtype="qint8")
    manager = QuantizationManager(config)
    quantized_model = manager.quantize_model(model)
    
    # 获取模型信息
    size_info = manager.get_model_size_info(quantized_model)
    logger.info(f"PTQ动态量化完成!")
    logger.info(f"原始模型大小: {sum(p.numel() * p.element_size() for p in model.parameters()) / (1024*1024):.2f}MB")
    logger.info(f"量化模型大小: {size_info['model_size_mb']:.2f}MB")
    
    return model, quantized_model


def example_ptq_static():
    """示例2: PTQ静态量化"""
    logger.info("=== PTQ静态量化示例 ===")
    
    # 创建模型和数据
    model = create_sample_model()
    train_loader, test_loader = create_sample_data()
    
    # 先训练模型
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(2):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    # 准备校准数据
    calibration_data = []
    model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            if i >= 10:  # 只用10个batch作为校准数据
                break
            calibration_data.append(data)
    
    # 静态量化
    logger.info("开始PTQ静态量化...")
    config = QuantizationConfig(
        method="static", 
        dtype="qint8",
        calibration_dataset=calibration_data,
        num_calibration_samples=len(calibration_data) * 32
    )
    manager = QuantizationManager(config)
    quantized_model = manager.quantize_model(model)
    
    # 获取模型信息
    size_info = manager.get_model_size_info(quantized_model)
    logger.info(f"PTQ静态量化完成!")
    logger.info(f"量化模型大小: {size_info['model_size_mb']:.2f}MB")
    
    return model, quantized_model


def example_qat():
    """示例3: QAT量化感知训练"""
    logger.info("=== QAT量化感知训练示例 ===")
    
    # 创建模型和数据
    model = create_sample_model()
    train_loader, test_loader = create_sample_data()
    
    # 创建QAT配置
    qat_config = QuantizationConfig(method="qat", dtype="qint8")
    
    # 创建量化训练器
    trainer = QuantizationTrainer(
        model=model,
        quantization_config=qat_config,
        output_dir="outputs/qat_example"
    )
    
    # 设置量化
    quantized_model = trainer.setup_quantization()
    logger.info("QAT模型设置完成")
    
    # 设置训练组件
    optimizer = optim.Adam(quantized_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 进行量化感知训练
    logger.info("开始QAT训练...")
    trainer.train_with_quantization(
        train_loader=train_loader,
        valid_loader=test_loader,
        num_epochs=3,
        optimizer=optimizer,
        criterion=criterion,
        save_best=True
    )
    
    return model, quantized_model


def example_quantization_analysis():
    """示例4: 量化效果分析"""
    logger.info("=== 量化效果分析示例 ===")
    
    # 创建模型
    original_model = create_sample_model()
    train_loader, test_loader = create_sample_data()
    
    # 量化模型
    config = QuantizationConfig(method="dynamic")
    manager = QuantizationManager(config)
    quantized_model = manager.quantize_model(original_model)
    
    # 创建分析器
    analyzer = QuantizationAnalyzer(original_model, quantized_model)
    
    # 比较模型大小
    size_comparison = analyzer.compare_model_sizes()
    logger.info("模型大小比较:")
    logger.info(f"  原始模型: {size_comparison['original']['model_size_mb']:.2f}MB")
    logger.info(f"  量化模型: {size_comparison['quantized']['model_size_mb']:.2f}MB")
    logger.info(f"  压缩比: {size_comparison['compression_ratio']:.2f}x")
    logger.info(f"  大小减少: {size_comparison['size_reduction_percent']:.1f}%")
    
    # 比较推理速度
    test_input = torch.randn(1, 3, 32, 32)
    speed_comparison = analyzer.compare_inference_speed(test_input, num_runs=10)
    logger.info("推理速度比较:")
    logger.info(f"  原始模型平均时间: {speed_comparison['original_avg_time']:.4f}s")
    logger.info(f"  量化模型平均时间: {speed_comparison['quantized_avg_time']:.4f}s")
    logger.info(f"  加速比: {speed_comparison['speedup']:.2f}x")
    
    # 比较准确率
    accuracy_comparison = analyzer.compare_accuracy(test_loader)
    logger.info("准确率比较:")
    logger.info(f"  原始模型准确率: {accuracy_comparison['original_metrics']['accuracy']:.4f}")
    logger.info(f"  量化模型准确率: {accuracy_comparison['quantized_metrics']['accuracy']:.4f}")
    logger.info(f"  准确率下降: {accuracy_comparison['accuracy_drop']:.4f}")
    
    return analyzer


def main():
    """运行所有PTQ和QAT示例"""
    logger.info("开始PTQ和QAT量化示例...")
    
    try:
        # PTQ动态量化
        original_model1, quantized_model1 = example_ptq_dynamic()
        
        # PTQ静态量化
        original_model2, quantized_model2 = example_ptq_static()
        
        # QAT量化感知训练
        original_model3, quantized_model3 = example_qat()
        
        # 量化效果分析
        analyzer = example_quantization_analysis()
        
        logger.info("🎉 所有PTQ和QAT示例运行完成!")
        
    except Exception as e:
        logger.error(f"示例运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
