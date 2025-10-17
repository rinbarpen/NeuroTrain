#!/usr/bin/env python3
"""
量化工具的高级接口和集成功能
提供与NeuroTrain训练框架的集成
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .core import QuantizationManager, QuantizationConfig, create_quantization_config
from ..config import get_config
from ..utils import load_model, save_model

logger = logging.getLogger(__name__)


class QuantizationTrainer:
    """量化训练器，集成到NeuroTrain训练流程中"""
    
    def __init__(self, 
                 model: nn.Module,
                 quantization_config: QuantizationConfig,
                 output_dir: Union[str, Path],
                 device: str = "cuda"):
        """
        初始化量化训练器
        
        Args:
            model: 要量化的模型
            quantization_config: 量化配置
            output_dir: 输出目录
            device: 设备
        """
        self.model = model
        self.quantization_config = quantization_config
        self.output_dir = Path(output_dir)
        self.device = device
        self.quantization_manager = None
        self.quantized_model = None
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_quantization(self, **kwargs) -> nn.Module:
        """设置量化"""
        logger.info(f"Setting up quantization with method: {self.quantization_config.method}")
        
        self.quantization_manager = QuantizationManager(self.quantization_config)
        self.quantized_model = self.quantization_manager.quantize_model(self.model, **kwargs)
        
        # 移动到指定设备
        self.quantized_model = self.quantized_model.to(self.device)
        
        logger.info("Quantization setup completed")
        return self.quantized_model
    
    def train_with_quantization(self,
                               train_loader: DataLoader,
                               valid_loader: Optional[DataLoader] = None,
                               num_epochs: int = 10,
                               optimizer: Optional[torch.optim.Optimizer] = None,
                               criterion: Optional[Callable] = None,
                               lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                               save_best: bool = True,
                               **kwargs):
        """
        使用量化模型进行训练
        
        Args:
            train_loader: 训练数据加载器
            valid_loader: 验证数据加载器
            num_epochs: 训练轮数
            optimizer: 优化器
            criterion: 损失函数
            lr_scheduler: 学习率调度器
            save_best: 是否保存最佳模型
            **kwargs: 其他训练参数
        """
        if self.quantized_model is None:
            raise ValueError("Quantization not set up. Call setup_quantization() first.")
        
        logger.info("Starting quantization-aware training...")
        
        # 设置训练模式
        if self.quantization_config.method == "qat":
            self.quantized_model.train()
        else:
            self.quantized_model.eval()
        
        # 训练循环
        best_loss = float('inf')
        train_losses = []
        valid_losses = []
        
        for epoch in range(num_epochs):
            # 训练阶段
            epoch_train_loss = self._train_epoch(
                self.quantized_model, train_loader, optimizer, criterion, **kwargs
            )
            train_losses.append(epoch_train_loss)
            
            # 验证阶段
            if valid_loader is not None:
                epoch_valid_loss = self._validate_epoch(
                    self.quantized_model, valid_loader, criterion, **kwargs
                )
                valid_losses.append(epoch_valid_loss)
                
                # 保存最佳模型
                if save_best and epoch_valid_loss < best_loss:
                    best_loss = epoch_valid_loss
                    self._save_model(self.output_dir / "best_quantized_model.pt")
                
                logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                          f"Train Loss: {epoch_train_loss:.4f}, "
                          f"Valid Loss: {epoch_valid_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}")
            
            # 学习率调度
            if lr_scheduler is not None:
                lr_scheduler.step()
        
        # 保存最终模型
        self._save_model(self.output_dir / "final_quantized_model.pt")
        
        # 保存训练历史
        self._save_training_history(train_losses, valid_losses)
        
        logger.info("Quantization-aware training completed")
    
    def _train_epoch(self, model: nn.Module, dataloader: DataLoader, 
                    optimizer: torch.optim.Optimizer, criterion: Callable, **kwargs) -> float:
        """训练一个epoch"""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # 准备数据
            if isinstance(batch, (tuple, list)):
                inputs, targets = batch[0], batch[1]
            else:
                inputs, targets = batch, None
            
            inputs = inputs.to(self.device)
            if targets is not None:
                targets = targets.to(self.device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # 计算损失
            if targets is not None:
                loss = criterion(outputs, targets)
            else:
                loss = criterion(outputs)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate_epoch(self, model: nn.Module, dataloader: DataLoader, 
                       criterion: Callable, **kwargs) -> float:
        """验证一个epoch"""
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # 准备数据
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs, targets = batch, None
                
                inputs = inputs.to(self.device)
                if targets is not None:
                    targets = targets.to(self.device)
                
                # 前向传播
                outputs = model(inputs)
                
                # 计算损失
                if targets is not None:
                    loss = criterion(outputs, targets)
                else:
                    loss = criterion(outputs)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _save_model(self, save_path: Path):
        """保存模型"""
        if self.quantization_manager:
            self.quantization_manager.save_quantized_model(self.quantized_model, save_path)
        else:
            torch.save(self.quantized_model.state_dict(), save_path)
    
    def _save_training_history(self, train_losses: List[float], valid_losses: List[float]):
        """保存训练历史"""
        import json
        
        history = {
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "quantization_config": self.quantization_config.to_dict()
        }
        
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)


class QuantizationAnalyzer:
    """量化分析器，用于分析量化效果"""
    
    def __init__(self, original_model: nn.Module, quantized_model: nn.Module):
        """
        初始化量化分析器
        
        Args:
            original_model: 原始模型
            quantized_model: 量化模型
        """
        self.original_model = original_model
        self.quantized_model = quantized_model
    
    def compare_model_sizes(self) -> Dict[str, Any]:
        """比较模型大小"""
        original_info = self._get_model_info(self.original_model)
        quantized_info = self._get_model_info(self.quantized_model)
        
        compression_ratio = original_info["model_size_mb"] / quantized_info["model_size_mb"]
        size_reduction = (original_info["model_size_mb"] - quantized_info["model_size_mb"]) / original_info["model_size_mb"] * 100
        
        return {
            "original": original_info,
            "quantized": quantized_info,
            "compression_ratio": compression_ratio,
            "size_reduction_percent": size_reduction
        }
    
    def compare_inference_speed(self, 
                               test_input: torch.Tensor, 
                               num_runs: int = 100) -> Dict[str, Any]:
        """比较推理速度"""
        import time
        
        # 预热
        with torch.no_grad():
            _ = self.original_model(test_input)
            _ = self.quantized_model(test_input)
        
        # 测试原始模型
        original_times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = self.original_model(test_input)
                end_time = time.time()
                original_times.append(end_time - start_time)
        
        # 测试量化模型
        quantized_times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = self.quantized_model(test_input)
                end_time = time.time()
                quantized_times.append(end_time - start_time)
        
        original_avg = sum(original_times) / len(original_times)
        quantized_avg = sum(quantized_times) / len(quantized_times)
        speedup = original_avg / quantized_avg
        
        return {
            "original_avg_time": original_avg,
            "quantized_avg_time": quantized_avg,
            "speedup": speedup,
            "original_times": original_times,
            "quantized_times": quantized_times
        }
    
    def compare_accuracy(self, 
                        test_loader: DataLoader, 
                        metric_func: Optional[Callable] = None) -> Dict[str, Any]:
        """比较准确率"""
        original_metrics = self._evaluate_model(self.original_model, test_loader, metric_func)
        quantized_metrics = self._evaluate_model(self.quantized_model, test_loader, metric_func)
        
        accuracy_drop = original_metrics.get("accuracy", 0) - quantized_metrics.get("accuracy", 0)
        
        return {
            "original_metrics": original_metrics,
            "quantized_metrics": quantized_metrics,
            "accuracy_drop": accuracy_drop
        }
    
    def _get_model_info(self, model: nn.Module) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 估算模型大小
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        model_size_mb = (param_size + buffer_size) / (1024 * 1024)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": model_size_mb
        }
    
    def _evaluate_model(self, model: nn.Module, test_loader: DataLoader, 
                       metric_func: Optional[Callable] = None) -> Dict[str, Any]:
        """评估模型"""
        model.eval()
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs, targets = batch, None
                
                outputs = model(inputs)
                
                if targets is not None:
                    # 计算准确率
                    if outputs.dim() > 1 and outputs.size(1) > 1:
                        predicted = torch.argmax(outputs, dim=1)
                        total_correct += (predicted == targets).sum().item()
                    total_samples += targets.size(0)
                
                # 如果有自定义评估函数
                if metric_func is not None:
                    metrics = metric_func(outputs, targets)
                    # 这里可以添加更多指标
        
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "total_correct": total_correct,
            "total_samples": total_samples
        }
    
    def generate_report(self, 
                       test_loader: DataLoader,
                       test_input: torch.Tensor,
                       output_path: Union[str, Path],
                       metric_func: Optional[Callable] = None) -> Dict[str, Any]:
        """生成量化分析报告"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 收集分析数据
        size_comparison = self.compare_model_sizes()
        speed_comparison = self.compare_inference_speed(test_input)
        accuracy_comparison = self.compare_accuracy(test_loader, metric_func)
        
        # 生成报告
        report = {
            "size_comparison": size_comparison,
            "speed_comparison": speed_comparison,
            "accuracy_comparison": accuracy_comparison,
            "summary": {
                "compression_ratio": size_comparison["compression_ratio"],
                "size_reduction_percent": size_comparison["size_reduction_percent"],
                "speedup": speed_comparison["speedup"],
                "accuracy_drop": accuracy_comparison["accuracy_drop"]
            }
        }
        
        # 保存报告
        import json
        with open(output_path / "quantization_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Quantization analysis report saved to {output_path}")
        return report


def integrate_quantization_with_training(model: nn.Module,
                                       config: Optional[Dict] = None,
                                       quantization_method: str = "dynamic",
                                       **kwargs) -> QuantizationTrainer:
    """
    将量化集成到训练流程中的便捷函数
    
    Args:
        model: 要量化的模型
        config: 配置字典
        quantization_method: 量化方法
        **kwargs: 其他参数
        
    Returns:
        量化训练器
    """
    if config is None:
        config = get_config()
    
    # 创建量化配置
    quant_config = create_quantization_config(method=quantization_method, **kwargs)
    
    # 获取输出目录
    output_dir = config.get("output_dir", "outputs/quantization")
    
    # 创建量化训练器
    trainer = QuantizationTrainer(
        model=model,
        quantization_config=quant_config,
        output_dir=output_dir,
        device=config.get("device", "cuda")
    )
    
    return trainer


def analyze_quantization_effectiveness(original_model: nn.Module,
                                     quantized_model: nn.Module,
                                     test_loader: DataLoader,
                                     test_input: torch.Tensor,
                                     output_path: Union[str, Path],
                                     metric_func: Optional[Callable] = None) -> Dict[str, Any]:
    """
    分析量化效果的便捷函数
    
    Args:
        original_model: 原始模型
        quantized_model: 量化模型
        test_loader: 测试数据加载器
        test_input: 测试输入
        output_path: 输出路径
        metric_func: 自定义评估函数
        
    Returns:
        分析报告
    """
    analyzer = QuantizationAnalyzer(original_model, quantized_model)
    report = analyzer.generate_report(test_loader, test_input, output_path, metric_func)
    return report
