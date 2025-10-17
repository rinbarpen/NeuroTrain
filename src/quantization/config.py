#!/usr/bin/env python3
"""
量化配置支持模块
为NeuroTrain框架添加量化配置支持
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch

from .core import QuantizationConfig, create_quantization_config
from ..config import get_config, get_config_value

logger = logging.getLogger(__name__)


def get_quantization_config(config: Optional[Dict] = None) -> Optional[QuantizationConfig]:
    """
    从配置中获取量化配置
    
    Args:
        config: 配置字典，如果为None则使用全局配置
        
    Returns:
        量化配置对象，如果未配置量化则返回None
    """
    if config is None:
        config = get_config()
    
    # 检查是否启用量化
    quantization_enabled = get_config_value("quantization.enabled", default=False)
    if not quantization_enabled:
        return None
    
    # 获取量化方法
    method = get_config_value("quantization.method", default="dynamic")
    
    # 获取量化参数
    quantization_params = {
        "method": method,
        "dtype": get_config_value("quantization.dtype", default="qint8"),
        "device": get_config_value("quantization.device", default="auto"),
        "trust_remote_code": get_config_value("quantization.trust_remote_code", default=False),
    }
    
    # 根据方法添加特定参数
    if method == "static":
        quantization_params.update({
            "num_calibration_samples": get_config_value("quantization.num_calibration_samples", default=100),
        })
    elif method == "gptq":
        quantization_params.update({
            "bits": get_config_value("quantization.bits", default=4),
            "group_size": get_config_value("quantization.group_size", default=128),
            "desc_act": get_config_value("quantization.desc_act", default=False),
        })
    elif method == "awq":
        # AWQ通常使用默认参数
        pass
    elif method in ["bnb_4bit", "bnb_8bit"]:
        quantization_params.update({
            "load_in_4bit": method == "bnb_4bit",
            "load_in_8bit": method == "bnb_8bit",
            "bnb_4bit_compute_dtype": get_config_value("quantization.bnb_4bit_compute_dtype", default="float16"),
            "bnb_4bit_quant_type": get_config_value("quantization.bnb_4bit_quant_type", default="nf4"),
            "bnb_4bit_use_double_quant": get_config_value("quantization.bnb_4bit_use_double_quant", default=True),
        })
    
    # 创建量化配置
    quant_config = create_quantization_config(**quantization_params)
    
    logger.info(f"Loaded quantization config: {method}")
    return quant_config


def setup_quantization_from_config(model: torch.nn.Module, 
                                 config: Optional[Dict] = None,
                                 **kwargs) -> Optional[torch.nn.Module]:
    """
    根据配置设置模型量化
    
    Args:
        model: 要量化的模型
        config: 配置字典
        **kwargs: 额外参数
        
    Returns:
        量化后的模型，如果未启用量化则返回原模型
    """
    quant_config = get_quantization_config(config)
    if quant_config is None:
        logger.info("Quantization not enabled in config")
        return model
    
    # 应用量化
    from .quantization import QuantizationManager
    manager = QuantizationManager(quant_config)
    quantized_model = manager.quantize_model(model, **kwargs)
    
    logger.info(f"Model quantized using {quant_config.method} method")
    return quantized_model


def create_quantization_config_template() -> Dict[str, Any]:
    """
    创建量化配置模板
    
    Returns:
        量化配置模板字典
    """
    template = {
        "quantization": {
            "enabled": False,
            "method": "dynamic",  # dynamic, static, qat, gptq, awq, bnb_4bit, bnb_8bit
            "dtype": "qint8",  # qint8, qint4, float16, bfloat16
            "device": "auto",
            "trust_remote_code": False,
            
            # 静态量化参数
            "num_calibration_samples": 100,
            
            # GPTQ参数
            "bits": 4,
            "group_size": 128,
            "desc_act": False,
            
            # BitsAndBytes参数
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            
            # 输出配置
            "save_quantized_model": True,
            "quantized_model_path": "outputs/quantized_model",
            "analysis_report_path": "outputs/quantization_analysis",
        }
    }
    
    return template


def validate_quantization_config(config: Dict[str, Any]) -> bool:
    """
    验证量化配置
    
    Args:
        config: 配置字典
        
    Returns:
        配置是否有效
    """
    try:
        quantization_config = config.get("quantization", {})
        
        if not quantization_config.get("enabled", False):
            return True  # 未启用量化，配置有效
        
        method = quantization_config.get("method", "dynamic")
        valid_methods = ["dynamic", "static", "qat", "gptq", "awq", "bnb_4bit", "bnb_8bit"]
        
        if method not in valid_methods:
            logger.error(f"Invalid quantization method: {method}")
            return False
        
        # 验证方法特定参数
        if method == "static":
            num_samples = quantization_config.get("num_calibration_samples", 100)
            if num_samples <= 0:
                logger.error("num_calibration_samples must be positive")
                return False
        
        elif method == "gptq":
            bits = quantization_config.get("bits", 4)
            if bits not in [2, 3, 4, 8]:
                logger.error(f"Invalid bits for GPTQ: {bits}")
                return False
            
            group_size = quantization_config.get("group_size", 128)
            if group_size <= 0:
                logger.error("group_size must be positive")
                return False
        
        elif method in ["bnb_4bit", "bnb_8bit"]:
            compute_dtype = quantization_config.get("bnb_4bit_compute_dtype", "float16")
            valid_dtypes = ["float16", "bfloat16", "float32"]
            if compute_dtype not in valid_dtypes:
                logger.error(f"Invalid bnb_4bit_compute_dtype: {compute_dtype}")
                return False
            
            quant_type = quantization_config.get("bnb_4bit_quant_type", "nf4")
            valid_quant_types = ["fp4", "nf4"]
            if quant_type not in valid_quant_types:
                logger.error(f"Invalid bnb_4bit_quant_type: {quant_type}")
                return False
        
        logger.info("Quantization config validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Error validating quantization config: {e}")
        return False


def get_quantization_requirements(method: str) -> Dict[str, Any]:
    """
    获取量化方法的依赖要求
    
    Args:
        method: 量化方法
        
    Returns:
        依赖要求字典
    """
    requirements = {
        "dynamic": {
            "torch": ">=1.8.0",
            "description": "PyTorch内置动态量化，无需额外依赖"
        },
        "static": {
            "torch": ">=1.8.0",
            "description": "PyTorch内置静态量化，需要校准数据集"
        },
        "qat": {
            "torch": ">=1.8.0",
            "description": "PyTorch内置量化感知训练"
        },
        "gptq": {
            "auto-gptq": ">=0.4.0",
            "torch": ">=1.13.0",
            "transformers": ">=4.21.0",
            "description": "GPTQ量化，需要auto-gptq库"
        },
        "awq": {
            "awq": ">=0.1.0",
            "torch": ">=1.13.0",
            "transformers": ">=4.21.0",
            "description": "AWQ量化，需要awq库"
        },
        "bnb_4bit": {
            "bitsandbytes": ">=0.39.0",
            "torch": ">=1.13.0",
            "transformers": ">=4.21.0",
            "description": "BitsAndBytes 4bit量化"
        },
        "bnb_8bit": {
            "bitsandbytes": ">=0.39.0",
            "torch": ">=1.13.0",
            "transformers": ">=4.21.0",
            "description": "BitsAndBytes 8bit量化"
        }
    }
    
    return requirements.get(method, {"description": "Unknown quantization method"})


def check_quantization_dependencies(method: str) -> bool:
    """
    检查量化方法的依赖是否满足
    
    Args:
        method: 量化方法
        
    Returns:
        依赖是否满足
    """
    requirements = get_quantization_requirements(method)
    
    if method in ["dynamic", "static", "qat"]:
        return True  # PyTorch内置方法
    
    elif method == "gptq":
        try:
            import auto_gptq
            return True
        except ImportError:
            logger.warning("auto-gptq not available for GPTQ quantization")
            return False
    
    elif method == "awq":
        try:
            import awq
            return True
        except ImportError:
            logger.warning("awq not available for AWQ quantization")
            return False
    
    elif method in ["bnb_4bit", "bnb_8bit"]:
        try:
            import bitsandbytes
            return True
        except ImportError:
            logger.warning("bitsandbytes not available for BitsAndBytes quantization")
            return False
    
    return False


def get_recommended_quantization_method(model_type: str, 
                                       use_case: str = "inference") -> str:
    """
    根据模型类型和使用场景推荐量化方法
    
    Args:
        model_type: 模型类型 ("transformer", "cnn", "llm", "custom")
        use_case: 使用场景 ("inference", "training", "fine_tuning")
        
    Returns:
        推荐的量化方法
    """
    recommendations = {
        ("transformer", "inference"): "dynamic",
        ("transformer", "training"): "qat",
        ("transformer", "fine_tuning"): "bnb_8bit",
        ("cnn", "inference"): "static",
        ("cnn", "training"): "qat",
        ("cnn", "fine_tuning"): "dynamic",
        ("llm", "inference"): "gptq",
        ("llm", "training"): "bnb_4bit",
        ("llm", "fine_tuning"): "bnb_4bit",
        ("custom", "inference"): "dynamic",
        ("custom", "training"): "qat",
        ("custom", "fine_tuning"): "dynamic",
    }
    
    return recommendations.get((model_type, use_case), "dynamic")
