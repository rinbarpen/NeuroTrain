#!/usr/bin/env python3
"""
NeuroTrain量化工具 - 专注于PyTorch、ONNX、TensorRT
支持PTQ和QAT量化方法
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import torch
import torch.nn as nn

# 兼容不同PyTorch版本的量化API
try:
    from torch.quantization import quantize_dynamic, quantize_static
    from torch.ao.quantization import QConfigMapping, get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
    import torch.fx as fx
    TORCH_QUANTIZATION_AVAILABLE = True
except ImportError:
    try:
        from torch.ao.quantization import quantize_dynamic, quantize_static
        from torch.ao.quantization import QConfigMapping, get_default_qconfig_mapping
        from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
        import torch.fx as fx
        TORCH_QUANTIZATION_AVAILABLE = True
    except ImportError:
        TORCH_QUANTIZATION_AVAILABLE = False
        warnings.warn("PyTorch quantization not available. Some quantization methods will be disabled.")

# ONNX支持
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    warnings.warn("ONNX not available. ONNX quantization will be disabled.")

# TensorRT支持（需要额外安装）
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    warnings.warn("TensorRT not available. TensorRT quantization will be disabled.")

logger = logging.getLogger(__name__)


class QuantizationConfig:
    """量化配置类"""
    
    def __init__(self, 
                 platform: str = "pytorch",
                 method: str = "dynamic",
                 dtype: str = "qint8",
                 # PyTorch specific
                 qconfig: Optional[Dict] = None,
                 calibration_dataset: Optional[Any] = None,
                 num_calibration_samples: int = 100,
                 # ONNX specific
                 onnx_opset_version: int = 11,
                 onnx_quantization_mode: str = "static",
                 # TensorRT specific
                 tensorrt_precision: str = "fp16",
                 tensorrt_workspace_size: int = 1 << 30,  # 1GB
                 # General
                 device: str = "cpu",
                 **kwargs):
        """
        初始化量化配置
        
        Args:
            platform: 量化平台 ("pytorch", "onnx", "tensorrt")
            method: 量化方法 ("dynamic", "static", "qat")
            dtype: 量化数据类型 ("qint8", "qint4", "fp16", "int8")
            qconfig: PyTorch自定义量化配置
            calibration_dataset: 校准数据集
            num_calibration_samples: 校准样本数量
            onnx_opset_version: ONNX opset版本
            onnx_quantization_mode: ONNX量化模式
            tensorrt_precision: TensorRT精度模式
            tensorrt_workspace_size: TensorRT工作空间大小
            device: 设备
        """
        self.platform = platform
        self.method = method
        self.dtype = dtype
        self.qconfig = qconfig
        self.calibration_dataset = calibration_dataset
        self.num_calibration_samples = num_calibration_samples
        self.onnx_opset_version = onnx_opset_version
        self.onnx_quantization_mode = onnx_quantization_mode
        self.tensorrt_precision = tensorrt_precision
        self.tensorrt_workspace_size = tensorrt_workspace_size
        self.device = device
        
        # 存储额外参数
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class QuantizationManager:
    """量化管理器 - 支持PyTorch、ONNX、TensorRT"""
    
    def __init__(self, config: QuantizationConfig):
        """
        初始化量化管理器
        
        Args:
            config: 量化配置
        """
        self.config = config
        self.quantized_model = None
        self.quantization_info = {}
    
    def quantize_model(self, model: nn.Module, **kwargs) -> Union[nn.Module, str, bytes]:
        """
        量化模型
        
        Args:
            model: 要量化的模型
            **kwargs: 额外参数
            
        Returns:
            量化后的模型（PyTorch模型、ONNX文件路径或TensorRT引擎）
        """
        platform = self.config.platform.lower()
        
        if platform == "pytorch":
            return self._quantize_pytorch(model, **kwargs)
        elif platform == "onnx":
            return self._quantize_onnx(model, **kwargs)
        elif platform == "tensorrt":
            return self._quantize_tensorrt(model, **kwargs)
        else:
            raise ValueError(f"Unsupported quantization platform: {platform}")
    
    def _quantize_pytorch(self, model: nn.Module, **kwargs) -> nn.Module:
        """PyTorch量化"""
        logger.info("Applying PyTorch quantization...")
        
        if not TORCH_QUANTIZATION_AVAILABLE:
            logger.warning("PyTorch quantization not available, returning original model")
            self.quantization_info = {
                "platform": "pytorch",
                "method": self.config.method,
                "note": "PyTorch quantization not available"
            }
            return model
        
        method = self.config.method.lower()
        
        if method == "dynamic":
            return self._quantize_pytorch_dynamic(model)
        elif method == "static":
            return self._quantize_pytorch_static(model)
        elif method == "qat":
            return self._quantize_pytorch_qat(model)
        else:
            raise ValueError(f"Unsupported PyTorch quantization method: {method}")
    
    def _quantize_pytorch_dynamic(self, model: nn.Module) -> nn.Module:
        """PyTorch动态量化"""
        logger.info("Applying PyTorch dynamic quantization...")
        
        # 获取量化类型
        dtype_map = {
            "qint8": torch.qint8,
            "qint4": torch.qint4,
        }
        dtype = dtype_map.get(self.config.dtype, torch.qint8)
        
        # 动态量化
        quantized_model = quantize_dynamic(
            model, 
            {nn.Linear, nn.LSTM, nn.GRU}, 
            dtype=dtype
        )
        
        self.quantization_info = {
            "platform": "pytorch",
            "method": "dynamic",
            "dtype": str(dtype),
            "quantized_modules": ["Linear", "LSTM", "GRU"]
        }
        
        logger.info("PyTorch dynamic quantization completed")
        return quantized_model
    
    def _quantize_pytorch_static(self, model: nn.Module) -> nn.Module:
        """PyTorch静态量化"""
        logger.info("Applying PyTorch static quantization...")
        
        if self.config.calibration_dataset is None:
            raise ValueError("Calibration dataset is required for static quantization")
        
        # 设置量化配置
        if self.config.qconfig is None:
            qconfig_mapping = get_default_qconfig_mapping("fbgemm")
        else:
            qconfig_mapping = QConfigMapping.from_dict(self.config.qconfig)
        
        # 准备模型
        model.eval()
        prepared_model = prepare_fx(model, qconfig_mapping)
        
        # 校准
        logger.info(f"Calibrating with {self.config.num_calibration_samples} samples...")
        prepared_model.eval()
        with torch.no_grad():
            for i, sample in enumerate(self.config.calibration_dataset):
                if i >= self.config.num_calibration_samples:
                    break
                if isinstance(sample, (tuple, list)):
                    prepared_model(*sample)
                else:
                    prepared_model(sample)
        
        # 转换
        quantized_model = convert_fx(prepared_model)
        
        self.quantization_info = {
            "platform": "pytorch",
            "method": "static",
            "calibration_samples": self.config.num_calibration_samples,
            "qconfig": self.config.qconfig
        }
        
        logger.info("PyTorch static quantization completed")
        return quantized_model
    
    def _quantize_pytorch_qat(self, model: nn.Module) -> nn.Module:
        """PyTorch量化感知训练"""
        logger.info("Setting up PyTorch quantization-aware training...")
        
        # 设置量化配置
        if self.config.qconfig is None:
            qconfig_mapping = get_default_qconfig_mapping("fbgemm")
        else:
            qconfig_mapping = QConfigMapping.from_dict(self.config.qconfig)
        
        # 准备QAT模型
        model.train()
        qat_model = prepare_fx(model, qconfig_mapping)
        
        self.quantization_info = {
            "platform": "pytorch",
            "method": "qat",
            "qconfig": self.config.qconfig,
            "training_mode": True
        }
        
        logger.info("PyTorch QAT setup completed. Model is ready for training.")
        return qat_model
    
    def _quantize_onnx(self, model: nn.Module, **kwargs) -> str:
        """ONNX量化"""
        logger.info("Applying ONNX quantization...")
        
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX is required for ONNX quantization")
        
        # 导出ONNX模型
        onnx_path = kwargs.get('onnx_path', 'model.onnx')
        input_shape = kwargs.get('input_shape', (1, 3, 224, 224))
        
        # 创建示例输入
        dummy_input = torch.randn(*input_shape)
        
        # 导出ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=self.config.onnx_opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        logger.info(f"ONNX model exported to {onnx_path}")
        
        # 量化ONNX模型
        quantized_onnx_path = onnx_path.replace('.onnx', '_quantized.onnx')
        
        if self.config.onnx_quantization_mode == "static":
            self._quantize_onnx_static(onnx_path, quantized_onnx_path)
        else:
            # 动态量化（ONNX默认）
            quantized_onnx_path = onnx_path
        
        self.quantization_info = {
            "platform": "onnx",
            "method": self.config.onnx_quantization_mode,
            "opset_version": self.config.onnx_opset_version,
            "model_path": quantized_onnx_path
        }
        
        logger.info("ONNX quantization completed")
        return quantized_onnx_path
    
    def _quantize_onnx_static(self, onnx_path: str, output_path: str):
        """ONNX静态量化"""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            quantize_dynamic(
                onnx_path,
                output_path,
                weight_type=QuantType.QUInt8,
                per_channel=False,
                reduce_range=False,
                activation_type=QuantType.QUInt8,
                extra_options={'EnableSubgraph': True}
            )
            logger.info(f"ONNX static quantization completed: {output_path}")
        except ImportError:
            logger.warning("ONNX quantization tools not available, skipping static quantization")
    
    def _quantize_tensorrt(self, model: nn.Module, **kwargs) -> bytes:
        """TensorRT量化"""
        logger.info("Applying TensorRT quantization...")
        
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT is required for TensorRT quantization")
        
        # 先导出ONNX
        onnx_path = kwargs.get('onnx_path', 'model.onnx')
        input_shape = kwargs.get('input_shape', (1, 3, 224, 224))
        
        dummy_input = torch.randn(*input_shape)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output']
        )
        
        # 构建TensorRT引擎
        engine_path = onnx_path.replace('.onnx', '.trt')
        engine = self._build_tensorrt_engine(onnx_path, engine_path)
        
        self.quantization_info = {
            "platform": "tensorrt",
            "precision": self.config.tensorrt_precision,
            "workspace_size": self.config.tensorrt_workspace_size,
            "engine_path": engine_path
        }
        
        logger.info("TensorRT quantization completed")
        return engine
    
    def _build_tensorrt_engine(self, onnx_path: str, engine_path: str) -> bytes:
        """构建TensorRT引擎"""
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        
        # 解析ONNX模型
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    logger.log(trt.Logger.ERROR, parser.get_error(error).desc)
                raise RuntimeError("Failed to parse ONNX model")
        
        # 配置构建器
        config = builder.create_builder_config()
        config.max_workspace_size = self.config.tensorrt_workspace_size
        
        # 设置精度
        if self.config.tensorrt_precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif self.config.tensorrt_precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
        
        # 构建引擎
        engine = builder.build_engine(network, config)
        
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # 保存引擎
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())
        
        return engine
    
    def save_quantized_model(self, model: Union[nn.Module, str, bytes], save_path: Union[str, Path], **kwargs):
        """保存量化模型"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if self.config.platform == "pytorch":
            # 保存PyTorch模型
            if hasattr(model, 'save_pretrained'):
                model.save_pretrained(save_path, **kwargs)
            else:
                torch.save(model.state_dict(), save_path / "model.pt")
        elif self.config.platform == "onnx":
            # ONNX模型已经是文件路径
            import shutil
            shutil.copy2(model, save_path / "model.onnx")
        elif self.config.platform == "tensorrt":
            # TensorRT引擎
            with open(save_path / "model.trt", 'wb') as f:
                f.write(model)
        
        # 保存量化信息
        torch.save(self.quantization_info, save_path / "quantization_info.pt")
        
        logger.info(f"Quantized model saved to {save_path}")
    
    def get_model_size_info(self, model: Union[nn.Module, str, bytes]) -> Dict[str, Any]:
        """获取模型大小信息"""
        if self.config.platform == "pytorch":
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # 估算模型大小（MB）
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            model_size_mb = (param_size + buffer_size) / (1024 * 1024)
            
            return {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "model_size_mb": model_size_mb,
                "platform": self.config.platform,
                "quantization_info": self.quantization_info
            }
        elif self.config.platform == "onnx":
            # ONNX文件大小
            model_path = model if isinstance(model, str) else self.quantization_info.get("model_path", "")
            if model_path and Path(model_path).exists():
                model_size_mb = Path(model_path).stat().st_size / (1024 * 1024)
            else:
                model_size_mb = 0
            
            return {
                "model_size_mb": model_size_mb,
                "platform": self.config.platform,
                "quantization_info": self.quantization_info
            }
        elif self.config.platform == "tensorrt":
            # TensorRT引擎大小
            if isinstance(model, bytes):
                model_size_mb = len(model) / (1024 * 1024)
            else:
                model_size_mb = 0
            
            return {
                "model_size_mb": model_size_mb,
                "platform": self.config.platform,
                "quantization_info": self.quantization_info
            }
        
        return {"platform": self.config.platform, "quantization_info": self.quantization_info}


def create_quantization_config(platform: str = "pytorch", 
                              method: str = "dynamic", 
                              **kwargs) -> QuantizationConfig:
    """
    创建量化配置的便捷函数
    
    Args:
        platform: 量化平台 ("pytorch", "onnx", "tensorrt")
        method: 量化方法
        **kwargs: 其他配置参数
        
    Returns:
        量化配置对象
    """
    return QuantizationConfig(platform=platform, method=method, **kwargs)


def quantize_model(model: nn.Module, config: QuantizationConfig, **kwargs) -> Tuple[Union[nn.Module, str, bytes], QuantizationManager]:
    """
    量化模型的便捷函数
    
    Args:
        model: 要量化的模型
        config: 量化配置
        **kwargs: 额外参数
        
    Returns:
        (量化后的模型, 量化管理器)
    """
    manager = QuantizationManager(config)
    quantized_model = manager.quantize_model(model, **kwargs)
    return quantized_model, manager


# 预定义的量化配置
PYTORCH_DYNAMIC_CONFIG = QuantizationConfig(platform="pytorch", method="dynamic", dtype="qint8")
PYTORCH_STATIC_CONFIG = QuantizationConfig(platform="pytorch", method="static", dtype="qint8")
PYTORCH_QAT_CONFIG = QuantizationConfig(platform="pytorch", method="qat", dtype="qint8")
ONNX_STATIC_CONFIG = QuantizationConfig(platform="onnx", method="static", onnx_quantization_mode="static")
TENSORRT_FP16_CONFIG = QuantizationConfig(platform="tensorrt", tensorrt_precision="fp16")
TENSORRT_INT8_CONFIG = QuantizationConfig(platform="tensorrt", tensorrt_precision="int8")