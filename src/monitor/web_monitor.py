"""
Web监控器 - 集成所有监控功能的Web界面

提供类似TensorBoard的Web监控体验，支持实时查看训练状态、图表和报告。
"""

import threading
import time
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime

from .training_monitor import TrainingMonitor, MonitorConfig
from .progress_tracker import ProgressTracker, ProgressConfig
from .alert_system import AlertSystem, AlertConfig, AlertLevel
from .web_server import WebMonitorServer
from queue import Queue


class WebMonitor:
    """Web监控器 - 集成所有监控功能"""
    
    def __init__(self, 
                 host: str = "0.0.0.0",
                 port: int = 5000,
                 debug: bool = False,
                 monitor_config: Optional[MonitorConfig] = None,
                 progress_config: Optional[ProgressConfig] = None,
                 alert_config: Optional[AlertConfig] = None):
        
        # 创建监控组件
        self.monitor = TrainingMonitor(monitor_config)
        self.progress_tracker = ProgressTracker(progress_config)
        self.alert_system = AlertSystem(alert_config)
        
        # 创建Web服务器
        self.web_server = WebMonitorServer(
            monitor=self.monitor,
            progress_tracker=self.progress_tracker,
            alert_system=self.alert_system,
            host=host,
            port=port,
            debug=debug
        )
        
        # 状态
        self.is_running = False
        self.server_thread = None
        
        # 设置默认告警规则
        self._setup_default_alerts()
    
    def _setup_default_alerts(self):
        """设置默认告警规则"""
        # 损失值告警
        loss_rule = self.alert_system.create_threshold_rule(
            name="high_loss",
            threshold_value=1.0,
            operator=">",
            level=AlertLevel.WARNING,
            cooldown=60.0
        )
        self.alert_system.add_rule(loss_rule)
        
        # 内存使用率告警
        memory_rule = self.alert_system.create_threshold_rule(
            name="high_memory",
            threshold_value=90.0,
            operator=">",
            level=AlertLevel.ERROR,
            cooldown=30.0
        )
        self.alert_system.add_rule(memory_rule)
        
        # CPU使用率告警
        cpu_rule = self.alert_system.create_threshold_rule(
            name="high_cpu",
            threshold_value=95.0,
            operator=">",
            level=AlertLevel.WARNING,
            cooldown=30.0
        )
        self.alert_system.add_rule(cpu_rule)
        
        # 损失值异常检测
        anomaly_rule = self.alert_system.create_anomaly_rule(
            name="loss_anomaly",
            anomaly_method="zscore",
            anomaly_threshold=3.0,
            level=AlertLevel.ERROR,
            cooldown=60.0
        )
        self.alert_system.add_rule(anomaly_rule)
    
    def start(self, block: bool = True):
        """启动Web监控器"""
        if self.is_running:
            print("Web monitor is already running")
            return
        
        print(f"🚀 Starting Web Monitor at http://{self.web_server.host}:{self.web_server.port}")
        print("📊 Features:")
        print("  - Real-time training metrics visualization")
        print("  - System resource monitoring")
        print("  - Progress tracking with ETA")
        print("  - Alert system with notifications")
        print("  - Performance statistics")
        print("  - Data export capabilities")
        print("\n💡 Open your browser and navigate to the URL above to view the dashboard")
        
        self.is_running = True
        
        if block:
            try:
                self.web_server.start()
            except KeyboardInterrupt:
                print("\n🛑 Shutting down Web Monitor...")
                self.stop()
        else:
            # 在后台线程中启动服务器
            self.server_thread = threading.Thread(
                target=self.web_server.start,
                daemon=True
            )
            self.server_thread.start()
    
    def stop(self):
        """停止Web监控器"""
        if not self.is_running:
            return
        
        print("🛑 Stopping Web Monitor...")
        self.is_running = False
        self.web_server.stop()
        
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=5.0)
    
    def start_training(self, total_epochs: int, steps_per_epoch: int):
        """开始训练跟踪"""
        self.progress_tracker.start_training(total_epochs, steps_per_epoch)
        self.monitor.start_monitoring()
        print(f"🎯 Started training tracking: {total_epochs} epochs, {steps_per_epoch} steps/epoch")
    
    def start_epoch(self, epoch: int):
        """开始新轮次"""
        self.progress_tracker.start_epoch(epoch)
        print(f"📈 Started epoch {epoch}")
    
    def start_step(self, step: int, batch_size: int = 1):
        """开始新步骤"""
        self.progress_tracker.start_step(step, batch_size)
    
    def end_step(self, batch_size: int = 1):
        """结束步骤"""
        self.progress_tracker.end_step(batch_size)
    
    def end_epoch(self):
        """结束轮次"""
        self.progress_tracker.end_epoch()
        print(f"✅ Completed epoch {self.progress_tracker.current_epoch}")
    
    def end_training(self):
        """结束训练"""
        self.progress_tracker.end_training()
        self.monitor.stop_monitoring()
        print("🏁 Training completed")
    
    def update_metrics(self, 
                      epoch: int = 0,
                      step: int = 0,
                      loss: float = 0.0,
                      learning_rate: float = 0.0,
                      batch_size: int = 0,
                      throughput: float = 0.0):
        """更新训练指标"""
        # 更新训练监控器
        self.monitor.update_training_metrics(
            epoch=epoch,
            step=step,
            loss=loss,
            learning_rate=learning_rate,
            batch_size=batch_size,
            throughput=throughput
        )
        
        # 更新告警系统
        self.alert_system.update_data('loss', loss)
        self.alert_system.update_data('throughput', throughput)
        
        # 更新进度跟踪器
        if self.progress_tracker.is_epoch_active:
            self.progress_tracker.end_step(batch_size)
            self.progress_tracker.start_step(step + 1, batch_size)
    
    def get_url(self) -> str:
        """获取Web界面URL"""
        return f"http://{self.web_server.host}:{self.web_server.port}"
    
    def get_status(self) -> Dict[str, Any]:
        """获取监控状态"""
        return {
            'is_running': self.is_running,
            'monitor_active': self.monitor.is_monitoring,
            'progress_active': self.progress_tracker.is_training,
            'alert_rules': len(self.alert_system.rules),
            'url': self.get_url()
        }
    
    def add_custom_alert_rule(self, 
                             name: str,
                             alert_type: str,
                             threshold_value: float = None,
                             operator: str = ">",
                             level: str = "WARNING",
                             cooldown: float = 60.0):
        """添加自定义告警规则"""
        alert_level = AlertLevel(level.upper())
        
        if alert_type == "threshold":
            rule = self.alert_system.create_threshold_rule(
                name=name,
                threshold_value=threshold_value,
                operator=operator,
                level=alert_level,
                cooldown=cooldown
            )
        elif alert_type == "trend":
            rule = self.alert_system.create_trend_rule(
                name=name,
                level=alert_level,
                cooldown=cooldown
            )
        elif alert_type == "anomaly":
            rule = self.alert_system.create_anomaly_rule(
                name=name,
                level=alert_level,
                cooldown=cooldown
            )
        else:
            raise ValueError(f"Unsupported alert type: {alert_type}")
        
        self.alert_system.add_rule(rule)
        print(f"✅ Added alert rule: {name} ({alert_type})")
    
    def save_config(self, filepath: Path):
        """保存配置"""
        config = {
            'monitor_config': {
                'log_interval': self.monitor.config.log_interval,
                'save_interval': self.monitor.config.save_interval,
                'enable_gpu_monitor': self.monitor.config.enable_gpu_monitor,
                'memory_threshold': self.monitor.config.memory_threshold,
                'loss_threshold': self.monitor.config.loss_threshold
            },
            'progress_config': {
                'window_size': self.progress_tracker.config.window_size,
                'min_samples': self.progress_tracker.config.min_samples,
                'enable_console_output': self.progress_tracker.config.enable_console_output
            },
            'alert_config': {
                'enable_console_output': self.alert_system.config.enable_console_output,
                'enable_file_output': self.alert_system.config.enable_file_output,
                'aggregation_window': self.alert_system.config.aggregation_window
            },
            'server_config': {
                'host': self.web_server.host,
                'port': self.web_server.port,
                'debug': self.web_server.debug
            }
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"💾 Configuration saved to {filepath}")
    
    def load_config(self, filepath: Path):
        """加载配置"""
        import json
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        # 更新配置
        if 'monitor_config' in config:
            for key, value in config['monitor_config'].items():
                if hasattr(self.monitor.config, key):
                    setattr(self.monitor.config, key, value)
        
        if 'progress_config' in config:
            for key, value in config['progress_config'].items():
                if hasattr(self.progress_tracker.config, key):
                    setattr(self.progress_tracker.config, key, value)
        
        if 'alert_config' in config:
            for key, value in config['alert_config'].items():
                if hasattr(self.alert_system.config, key):
                    setattr(self.alert_system.config, key, value)
        
        print(f"📂 Configuration loaded from {filepath}")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start(block=False)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()


def create_web_monitor(host: str = "0.0.0.0",
                      port: int = 5000,
                      debug: bool = False,
                      **kwargs) -> WebMonitor:
    """创建Web监控器"""
    return WebMonitor(host=host, port=port, debug=debug, **kwargs)
