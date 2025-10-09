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

    def save(self, path: Path):
        dummy_input = torch.randn(self.input_size)
        torch.onnx.export(self.model, dummy_input, path)
