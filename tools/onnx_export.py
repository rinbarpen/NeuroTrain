# This file is to export the given model to ONNX format

import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, List, Tuple, Optional
import logging


class OnnxExport:
    def __init__(self, model: nn.Module, input_sizes: Union[Tuple, List[Tuple]], 
                 input_names: Optional[List[str]] = None, 
                 output_names: Optional[List[str]] = None,
                 dynamic_axes: Optional[dict] = None):
        """
        初始化ONNX导出器
        
        Args:
            model: 要导出的PyTorch模型
            input_sizes: 输入尺寸，可以是单个元组或元组列表
            input_names: 输入名称列表
            output_names: 输出名称列表
            dynamic_axes: 动态轴配置
        """
        self.model = model
        self.input_sizes = input_sizes if isinstance(input_sizes, list) else [input_sizes]
        self.input_names = input_names or [f"input_{i}" for i in range(len(self.input_sizes))]
        self.output_names = output_names or ["output"]
        self.dynamic_axes = dynamic_axes or {}
        
        # 确保模型处于评估模式
        self.model.eval()

    def save(self, path: Path, opset_version: int = 11, 
             export_params: bool = True, verbose: bool = False) -> bool:
        """
        将模型导出为ONNX格式
        
        Args:
            path: 输出文件路径
            opset_version: ONNX操作集版本
            export_params: 是否导出模型参数
            verbose: 是否显示详细信息
            
        Returns:
            bool: 导出是否成功
        """
        try:
            # 创建虚拟输入
            dummy_inputs = []
            for input_size in self.input_sizes:
                if isinstance(input_size, tuple):
                    dummy_input = torch.randn(1, *input_size)
                else:
                    dummy_input = torch.randn(input_size)
                dummy_inputs.append(dummy_input)
            
            # 如果只有一个输入，直接使用张量
            if len(dummy_inputs) == 1:
                dummy_inputs = dummy_inputs[0]
            
            # 确保输出目录存在
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # 导出模型
            torch.onnx.export(
                self.model,
                dummy_inputs,
                str(path),
                export_params=export_params,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=self.input_names,
                output_names=self.output_names,
                dynamic_axes=self.dynamic_axes,
                verbose=verbose
            )
            
            logging.info(f"模型已成功导出为ONNX格式: {path}")
            return True
            
        except Exception as e:
            logging.error(f"ONNX导出失败: {e}")
            return False
    
    def verify_onnx(self, onnx_path: Path) -> bool:
        """
        验证导出的ONNX模型
        
        Args:
            onnx_path: ONNX文件路径
            
        Returns:
            bool: 验证是否成功
        """
        try:
            import onnx
            
            # 加载ONNX模型
            onnx_model = onnx.load(str(onnx_path))
            
            # 检查模型
            onnx.checker.check_model(onnx_model)
            
            logging.info(f"ONNX模型验证成功: {onnx_path}")
            return True
            
        except ImportError:
            logging.warning("onnx库未安装，跳过模型验证")
            return True
        except Exception as e:
            logging.error(f"ONNX模型验证失败: {e}")
            return False
    
    def get_model_info(self) -> dict:
        """
        获取模型信息
        
        Returns:
            dict: 模型信息字典
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "input_shapes": self.input_sizes,
            "input_names": self.input_names,
            "output_names": self.output_names
        }


def export_model_to_onnx(model: nn.Module, 
                        input_sizes: Union[Tuple, List[Tuple]], 
                        output_path: Path,
                        **kwargs) -> bool:
    """
    便捷函数：将模型导出为ONNX格式
    
    Args:
        model: PyTorch模型
        input_sizes: 输入尺寸
        output_path: 输出路径
        **kwargs: 其他参数
        
    Returns:
        bool: 导出是否成功
    """
    exporter = OnnxExport(model, input_sizes, **kwargs)
    success = exporter.save(output_path)
    
    if success:
        exporter.verify_onnx(output_path)
        info = exporter.get_model_info()
        logging.info(f"模型信息: {info}")
    
    return success
