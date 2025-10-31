"""
训练监控器 - 核心训练监控功能

提供训练过程中的实时监控、进度跟踪、性能分析和异常检测功能。
"""

import time
import threading
import queue
import logging
from typing import Dict, List, Optional, Callable, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import psutil
import torch
import numpy as np

from src.recorder.meter import Meter, MiniMeter
from src.utils.ndict import NDict


@dataclass
class MonitorConfig:
    """监控配置"""
    # 基础配置
    log_interval: float = 1.0  # 日志记录间隔（秒）
    save_interval: float = 60.0  # 数据保存间隔（秒）
    max_history: int = 1000  # 最大历史记录数
    
    # 性能监控
    enable_gpu_monitor: bool = True
    enable_cpu_monitor: bool = True
    enable_memory_monitor: bool = True
    
    # 告警配置
    enable_alerts: bool = True
    loss_threshold: float = 10.0  # 损失值告警阈值
    memory_threshold: float = 0.9  # 内存使用率告警阈值
    gpu_memory_threshold: float = 0.9  # GPU内存使用率告警阈值
    
    # 输出配置
    log_dir: Optional[Path] = None
    enable_console_output: bool = True
    enable_file_output: bool = True


@dataclass
class SystemMetrics:
    """系统指标"""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    gpu_utilization: float = 0.0


@dataclass
class TrainingMetrics:
    """训练指标"""
    timestamp: datetime = field(default_factory=datetime.now)
    epoch: int = 0
    step: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    batch_size: int = 0
    throughput: float = 0.0  # samples/second
    eta: Optional[timedelta] = None


class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self, config: Optional[MonitorConfig] = None):
        self.config = config or MonitorConfig()
        self.logger = self._setup_logger()
        
        # 监控状态
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # 指标存储
        self.system_metrics_history: List[SystemMetrics] = []
        self.training_metrics_history: List[TrainingMetrics] = []
        
        # 性能指标
        self.system_meters: Dict[str, Meter] = {}
        self.training_meters: Dict[str, Meter] = {}
        
        # 告警回调
        self.alert_callbacks: List[Callable[[str, Dict], None]] = []
        
        # 初始化指标
        self._init_meters()
        
        # 创建输出目录
        if self.config.log_dir:
            self.config.log_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志器"""
        logger = logging.getLogger(f"TrainingMonitor_{id(self)}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # 控制台输出
            if self.config.enable_console_output:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)
            
            # 文件输出
            if self.config.enable_file_output and self.config.log_dir:
                file_handler = logging.FileHandler(
                    self.config.log_dir / "monitor.log"
                )
                file_handler.setLevel(logging.INFO)
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
        
        return logger
    
    def _init_meters(self):
        """初始化指标计量器"""
        # 系统指标
        if self.config.enable_cpu_monitor:
            self.system_meters['cpu_percent'] = Meter('cpu_percent', ':2f')
        if self.config.enable_memory_monitor:
            self.system_meters['memory_percent'] = Meter('memory_percent', ':2f')
            self.system_meters['memory_used_gb'] = Meter('memory_used_gb', ':2f')
        if self.config.enable_gpu_monitor and torch.cuda.is_available():
            self.system_meters['gpu_memory_percent'] = Meter('gpu_memory_percent', ':2f')
            self.system_meters['gpu_utilization'] = Meter('gpu_utilization', ':2f')
        
        # 训练指标
        self.training_meters['loss'] = Meter('loss', ':4f')
        self.training_meters['learning_rate'] = Meter('learning_rate', ':6f')
        self.training_meters['throughput'] = Meter('throughput', ':2f')
    
    def start_monitoring(self):
        """开始监控"""
        if self.is_monitoring:
            self.logger.warning("监控已在运行中")
            return
        
        self.is_monitoring = True
        self.stop_event.clear()
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info("训练监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.stop_event.set()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        self.logger.info("训练监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        last_log_time = time.time()
        last_save_time = time.time()
        
        while not self.stop_event.is_set():
            try:
                current_time = time.time()
                
                # 收集系统指标
                if current_time - last_log_time >= self.config.log_interval:
                    self._collect_system_metrics()
                    last_log_time = current_time
                
                # 保存数据
                if current_time - last_save_time >= self.config.save_interval:
                    self._save_metrics()
                    last_save_time = current_time
                
                # 检查告警
                if self.config.enable_alerts:
                    self._check_alerts()
                
                time.sleep(0.1)  # 避免过度占用CPU
                
            except Exception as e:
                self.logger.error(f"监控循环出错: {e}")
                time.sleep(1.0)
    
    def _collect_system_metrics(self):
        """收集系统指标"""
        metrics = SystemMetrics()
        
        # CPU使用率
        if self.config.enable_cpu_monitor:
            metrics.cpu_percent = psutil.cpu_percent()
            self.system_meters['cpu_percent'].update(metrics.cpu_percent)
        
        # 内存使用情况
        if self.config.enable_memory_monitor:
            memory = psutil.virtual_memory()
            metrics.memory_percent = memory.percent
            metrics.memory_used_gb = memory.used / (1024**3)
            metrics.memory_total_gb = memory.total / (1024**3)
            
            self.system_meters['memory_percent'].update(metrics.memory_percent)
            self.system_meters['memory_used_gb'].update(metrics.memory_used_gb)
        
        # GPU使用情况
        if self.config.enable_gpu_monitor and torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                metrics.gpu_memory_used_gb = gpu_memory
                metrics.gpu_memory_total_gb = gpu_memory_total
                metrics.gpu_utilization = (gpu_memory / gpu_memory_total) * 100
                
                self.system_meters['gpu_memory_percent'].update(metrics.gpu_utilization)
                self.system_meters['gpu_utilization'].update(metrics.gpu_utilization)
            except Exception as e:
                self.logger.warning(f"GPU监控出错: {e}")
        
        # 添加到历史记录
        self.system_metrics_history.append(metrics)
        
        # 限制历史记录长度
        if len(self.system_metrics_history) > self.config.max_history:
            self.system_metrics_history = self.system_metrics_history[-self.config.max_history:]
    
    def update_training_metrics(self, 
                              epoch: int = 0,
                              step: int = 0,
                              loss: float = 0.0,
                              learning_rate: float = 0.0,
                              batch_size: int = 0,
                              throughput: float = 0.0,
                              eta: Optional[timedelta] = None):
        """更新训练指标"""
        metrics = TrainingMetrics(
            epoch=epoch,
            step=step,
            loss=loss,
            learning_rate=learning_rate,
            batch_size=batch_size,
            throughput=throughput,
            eta=eta
        )
        
        # 更新计量器
        self.training_meters['loss'].update(loss)
        self.training_meters['learning_rate'].update(learning_rate)
        self.training_meters['throughput'].update(throughput)
        
        # 添加到历史记录
        self.training_metrics_history.append(metrics)
        
        # 限制历史记录长度
        if len(self.training_metrics_history) > self.config.max_history:
            self.training_metrics_history = self.training_metrics_history[-self.config.max_history:]
    
    def _check_alerts(self):
        """检查告警条件"""
        if not self.system_metrics_history:
            return
        
        latest_system = self.system_metrics_history[-1]
        latest_training = self.training_metrics_history[-1] if self.training_metrics_history else None
        
        # 内存使用率告警
        if latest_system.memory_percent > self.config.memory_threshold * 100:
            self._trigger_alert("high_memory_usage", {
                "memory_percent": latest_system.memory_percent,
                "threshold": self.config.memory_threshold * 100
            })
        
        # GPU内存使用率告警
        if latest_system.gpu_utilization > self.config.gpu_memory_threshold * 100:
            self._trigger_alert("high_gpu_memory_usage", {
                "gpu_utilization": latest_system.gpu_utilization,
                "threshold": self.config.gpu_memory_threshold * 100
            })
        
        # 损失值告警
        if latest_training and latest_training.loss > self.config.loss_threshold:
            self._trigger_alert("high_loss", {
                "loss": latest_training.loss,
                "threshold": self.config.loss_threshold
            })
    
    def _trigger_alert(self, alert_type: str, data: Dict[str, Any]):
        """触发告警"""
        alert_message = f"告警: {alert_type} - {data}"
        self.logger.warning(alert_message)
        
        # 调用注册的回调函数
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, data)
            except Exception as e:
                self.logger.error(f"告警回调出错: {e}")
    
    def add_alert_callback(self, callback: Callable[[str, Dict], None]):
        """添加告警回调函数"""
        self.alert_callbacks.append(callback)
    
    def _save_metrics(self):
        """保存指标数据"""
        if not self.config.log_dir:
            return
        
        try:
            # 保存系统指标
            system_data = []
            for metrics in self.system_metrics_history:
                system_data.append({
                    'timestamp': metrics.timestamp.isoformat(),
                    'cpu_percent': metrics.cpu_percent,
                    'memory_percent': metrics.memory_percent,
                    'memory_used_gb': metrics.memory_used_gb,
                    'memory_total_gb': metrics.memory_total_gb,
                    'gpu_memory_used_gb': metrics.gpu_memory_used_gb,
                    'gpu_memory_total_gb': metrics.gpu_memory_total_gb,
                    'gpu_utilization': metrics.gpu_utilization
                })
            
            with open(self.config.log_dir / "system_metrics.json", 'w') as f:
                json.dump(system_data, f, indent=2)
            
            # 保存训练指标
            training_data = []
            for metrics in self.training_metrics_history:
                training_data.append({
                    'timestamp': metrics.timestamp.isoformat(),
                    'epoch': metrics.epoch,
                    'step': metrics.step,
                    'loss': metrics.loss,
                    'learning_rate': metrics.learning_rate,
                    'batch_size': metrics.batch_size,
                    'throughput': metrics.throughput,
                    'eta': str(metrics.eta) if metrics.eta else None
                })
            
            with open(self.config.log_dir / "training_metrics.json", 'w') as f:
                json.dump(training_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"保存指标数据出错: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        summary = {
            'is_monitoring': self.is_monitoring,
            'system_metrics_count': len(self.system_metrics_history),
            'training_metrics_count': len(self.training_metrics_history),
            'meters': {}
        }
        
        # 添加计量器摘要
        for name, meter in {**self.system_meters, **self.training_meters}.items():
            summary['meters'][name] = {
                'avg': meter.avg,
                'sum': meter.sum,
                'count': meter.count
            }
        
        return summary
    
    def reset(self):
        """重置监控器"""
        self.stop_monitoring()
        
        # 清空历史记录
        self.system_metrics_history.clear()
        self.training_metrics_history.clear()
        
        # 重置计量器
        for meter in {**self.system_meters, **self.training_meters}.values():
            meter.reset()
        
        self.logger.info("监控器已重置")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_monitoring()
