#!/usr/bin/env python3
"""
NeuroTrain量化模块
专注于PyTorch、ONNX、TensorRT的量化操作
"""

from .core import QuantizationConfig, QuantizationManager, create_quantization_config
from .trainer import QuantizationTrainer, QuantizationAnalyzer
from .config import (
    get_quantization_config,
    setup_quantization_from_config,
    create_quantization_config_template,
    validate_quantization_config,
    get_quantization_requirements,
    check_quantization_dependencies,
    get_recommended_quantization_method
)

# 预定义的量化配置
from .core import (
    PYTORCH_DYNAMIC_CONFIG,
    PYTORCH_STATIC_CONFIG,
    PYTORCH_QAT_CONFIG,
    ONNX_STATIC_CONFIG,
    TENSORRT_FP16_CONFIG,
    TENSORRT_INT8_CONFIG
)

__all__ = [
    # 核心类
    'QuantizationConfig',
    'QuantizationManager',
    'QuantizationTrainer',
    'QuantizationAnalyzer',
    
    # 便捷函数
    'create_quantization_config',
    'quantize_model',
    'integrate_quantization_with_training',
    'analyze_quantization_effectiveness',
    
    # 配置相关
    'get_quantization_config',
    'setup_quantization_from_config',
    'create_quantization_config_template',
    'validate_quantization_config',
    'get_quantization_requirements',
    'check_quantization_dependencies',
    'get_recommended_quantization_method',
    
    # 预定义配置
    'PYTORCH_DYNAMIC_CONFIG',
    'PYTORCH_STATIC_CONFIG',
    'PYTORCH_QAT_CONFIG',
    'ONNX_STATIC_CONFIG',
    'TENSORRT_FP16_CONFIG',
    'TENSORRT_INT8_CONFIG',
]

# 版本信息
__version__ = "1.0.0"
__author__ = "NeuroTrain Team"
__description__ = "NeuroTrain Quantization Module - PyTorch, ONNX, TensorRT Support"
