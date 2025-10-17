#!/usr/bin/env python3
"""
量化工具命令行接口
提供简单的命令行接口来使用量化功能
"""

import argparse
import logging
import torch
import torch.nn as nn
from pathlib import Path
import json

# 添加项目根目录到路径
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.quantization import QuantizationConfig, QuantizationManager, QuantizationAnalyzer
from src.quantization.config import (
    get_quantization_config,
    setup_quantization_from_config,
    check_quantization_dependencies,
    get_quantization_requirements
)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_model():
    """创建示例模型"""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(784, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    return SimpleModel()


def quantize_model_cli(model_path: str, 
                      output_path: str, 
                      platform: str = "pytorch",
                      method: str = "dynamic",
                      input_shape: tuple | None = None,
                      onnx_opset: int = 11,
                      onnx_mode: str = "static",
                      tensorrt_precision: str = "fp16",
                      **kwargs):
    """
    命令行量化模型
    
    Args:
        model_path: 模型路径
        output_path: 输出路径
        method: 量化方法
        **kwargs: 其他参数
    """
    logger.info(f"开始量化模型: {model_path}")
    logger.info(f"量化平台: {platform}")
    logger.info(f"量化方法: {method}")
    logger.info(f"输出路径: {output_path}")
    
    try:
        # 加载模型
        if model_path.endswith('.pt') or model_path.endswith('.pth'):
            model = torch.load(model_path, map_location='cpu')
        else:
            # 假设是HuggingFace模型路径
            from transformers import AutoModel
            model = AutoModel.from_pretrained(model_path)
        
        # 创建量化配置（按平台）
        if platform == "pytorch":
            quant_config = QuantizationConfig(platform=platform, method=method, **kwargs)
        elif platform == "onnx":
            quant_config = QuantizationConfig(platform=platform, method=method, onnx_opset_version=onnx_opset, onnx_quantization_mode=onnx_mode, **kwargs)
        elif platform == "tensorrt":
            quant_config = QuantizationConfig(platform=platform, method=method, tensorrt_precision=tensorrt_precision, **kwargs)
        else:
            raise ValueError(f"Unsupported platform: {platform}")
        
        # 量化模型
        manager = QuantizationManager(quant_config)
        if platform == "pytorch":
            quantized_model = manager.quantize_model(model)
        elif platform == "onnx":
            if input_shape is None:
                input_shape = (1, 3, 224, 224)
            quantized_model = manager.quantize_model(model, onnx_path=str(Path(output_path) / "model.onnx"), input_shape=input_shape)
        elif platform == "tensorrt":
            if input_shape is None:
                input_shape = (1, 3, 224, 224)
            quantized_model = manager.quantize_model(model, onnx_path=str(Path(output_path) / "model.onnx"), input_shape=input_shape)
        
        # 保存量化模型
        output_path = Path(output_path)
        manager.save_quantized_model(quantized_model, output_path)
        
        # 获取模型信息
        size_info = manager.get_model_size_info(quantized_model)
        logger.info(f"量化完成! 模型信息:")
        if 'total_parameters' in size_info:
            logger.info(f"  总参数: {size_info['total_parameters']:,}")
        logger.info(f"  模型大小: {size_info['model_size_mb']:.2f}MB")
        logger.info(f"  平台: {size_info.get('platform', platform)}")
        
        return True
        
    except Exception as e:
        logger.error(f"量化失败: {e}")
        return False


def analyze_model_cli(original_model_path: str,
                     quantized_model_path: str,
                     output_path: str,
                     test_data_path: str = None):
    """
    命令行分析量化效果
    
    Args:
        original_model_path: 原始模型路径
        quantized_model_path: 量化模型路径
        output_path: 分析报告输出路径
        test_data_path: 测试数据路径
    """
    logger.info("开始分析量化效果...")
    
    try:
        # 加载模型
        original_model = torch.load(original_model_path, map_location='cpu')
        quantized_model = torch.load(quantized_model_path, map_location='cpu')
        
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
        test_input = torch.randn(1, 784)  # 示例输入
        speed_comparison = analyzer.compare_inference_speed(test_input, num_runs=10)
        logger.info("推理速度比较:")
        logger.info(f"  原始模型平均时间: {speed_comparison['original_avg_time']:.4f}s")
        logger.info(f"  量化模型平均时间: {speed_comparison['quantized_avg_time']:.4f}s")
        logger.info(f"  加速比: {speed_comparison['speedup']:.2f}x")
        
        # 生成报告
        output_path = Path(output_path)
        report = analyzer.generate_report(
            test_loader=None,  # 简化版本不测试准确率
            test_input=test_input,
            output_path=output_path
        )
        
        logger.info(f"分析报告已保存到: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"分析失败: {e}")
        return False


def list_quantization_methods():
    """列出可用的平台与方法"""
    print("可用的量化平台与方法:")
    print("-" * 50)
    print("pytorch: dynamic, static, qat")
    print("onnx: dynamic(导出), static(ORT量化)")
    print("tensorrt: fp16, int8 (需安装TensorRT)")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="NeuroTrain量化工具")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 量化命令
    quantize_parser = subparsers.add_parser('quantize', help='量化模型')
    quantize_parser.add_argument('model_path', help='模型路径')
    quantize_parser.add_argument('output_path', help='输出路径')
    quantize_parser.add_argument('--platform', default='pytorch', choices=['pytorch','onnx','tensorrt'], help='量化平台')
    quantize_parser.add_argument('--method', default='dynamic', choices=['dynamic','static','qat'], help='量化方法 (pytorch)')
    quantize_parser.add_argument('--dtype', default='qint8', help='量化数据类型')
    quantize_parser.add_argument('--input-shape', nargs='+', type=int, default=None, help='输入形状 (onnx/tensorrt)，例如: --input-shape 1 3 224 224')
    quantize_parser.add_argument('--onnx-opset', type=int, default=11, help='ONNX opset 版本')
    quantize_parser.add_argument('--onnx-mode', choices=['dynamic','static'], default='static', help='ONNX 量化模式')
    quantize_parser.add_argument('--trt-precision', choices=['fp16','int8'], default='fp16', help='TensorRT 精度')
    
    # 分析命令
    analyze_parser = subparsers.add_parser('analyze', help='分析量化效果')
    analyze_parser.add_argument('original_model', help='原始模型路径')
    analyze_parser.add_argument('quantized_model', help='量化模型路径')
    analyze_parser.add_argument('output_path', help='分析报告输出路径')
    
    # 列表命令
    list_parser = subparsers.add_parser('list', help='列出可用的量化方法')
    
    # 示例命令
    example_parser = subparsers.add_parser('example', help='运行示例')
    example_parser.add_argument('--method', default='dynamic', help='示例量化方法')
    
    args = parser.parse_args()
    
    if args.command == 'quantize':
        input_shape = tuple(args.input_shape) if args.input_shape is not None else None
        success = quantize_model_cli(
            args.model_path,
            args.output_path,
            platform=args.platform,
            method=args.method,
            dtype=args.dtype,
            input_shape=input_shape,
            onnx_opset=args.onnx_opset,
            onnx_mode=args.onnx_mode,
            tensorrt_precision=args.trt_precision
        )
        exit(0 if success else 1)
        
    elif args.command == 'analyze':
        success = analyze_model_cli(
            args.original_model,
            args.quantized_model,
            args.output_path
        )
        exit(0 if success else 1)
        
    elif args.command == 'list':
        list_quantization_methods()
        
    elif args.command == 'example':
        logger.info("运行量化示例...")
        
        # 创建示例模型
        model = create_sample_model()
        
        # 保存原始模型
        original_model_path = "temp_original_model.pt"
        torch.save(model, original_model_path)
        logger.info(f"原始模型已保存到: {original_model_path}")
        
        # 量化模型
        quantized_model_path = "temp_quantized_model"
        success = quantize_model_cli(
            original_model_path,
            quantized_model_path,
            method=args.method
        )
        
        if success:
            # 分析效果
            analyze_model_cli(
                original_model_path,
                quantized_model_path + "/model.pt",
                "temp_analysis_output"
            )
            
            # 清理临时文件
            Path(original_model_path).unlink(missing_ok=True)
            logger.info("示例完成，临时文件已清理")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
