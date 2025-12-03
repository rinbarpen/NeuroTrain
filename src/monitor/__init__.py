"""
训练监控模块 - 提供训练过程中的实时监控、进度跟踪、性能分析和异常检测功能

主要组件:
- TrainingMonitor: 核心训练监控器，提供系统资源监控和训练指标跟踪
- ProgressTracker: 进度跟踪器，提供训练进度跟踪和ETA计算
- AlertSystem: 告警系统，提供多种告警机制和异常检测
- MonitorUtils: 监控工具函数，提供数据可视化、分析和导出功能

使用示例:
    from src.monitor import TrainingMonitor, ProgressTracker, AlertSystem
    
    # 创建监控器
    monitor = TrainingMonitor()
    progress_tracker = ProgressTracker()
    alert_system = AlertSystem()
    
    # 开始监控
    with monitor:
        # 开始训练跟踪
        progress_tracker.start_training(total_epochs=100, steps_per_epoch=1000)
        
        for epoch in range(100):
            progress_tracker.start_epoch(epoch)
            
            for step in range(1000):
                progress_tracker.start_step(step)
                
                # 训练代码...
                loss = model.train_step(batch)
                
                # 更新监控数据
                monitor.update_training_metrics(
                    epoch=epoch, step=step, loss=loss
                )
                alert_system.update_data('loss', loss)
                
                progress_tracker.end_step()
            
            progress_tracker.end_epoch()
        
        progress_tracker.end_training()
"""

# 核心监控组件
from .training_monitor import (
    TrainingMonitor,
    MonitorConfig,
    SystemMetrics,
    TrainingMetrics
)

from .progress_tracker import (
    ProgressTracker,
    ProgressConfig,
    ProgressSnapshot
)

from .alert_system import (
    AlertSystem,
    AlertConfig,
    AlertRule,
    Alert,
    AlertLevel,
    AlertType
)

# 工具函数
from .monitor_utils import (
    plot_training_metrics,
    plot_system_metrics,
    plot_progress_tracker,
    export_monitor_data,
    export_progress_data,
    export_alert_data,
    generate_monitor_report,
    analyze_performance_trends,
    detect_performance_anomalies
)

from .experiment_manager import (
    discover_experiments,
    load_experiment_snapshot,
)

# Web监控（可选，需要flask_socketio）
try:
    from .web_monitor import WebMonitor, create_web_monitor
    from .web_server import WebMonitorServer
    WEB_MONITOR_AVAILABLE = True
except ImportError:
    WEB_MONITOR_AVAILABLE = False
    WebMonitor = None
    WebMonitorServer = None
    create_web_monitor = None

# 导出所有主要类和函数
__all__ = [
    # 核心监控组件
    'TrainingMonitor',
    'MonitorConfig',
    'SystemMetrics',
    'TrainingMetrics',
    
    # 进度跟踪
    'ProgressTracker',
    'ProgressConfig',
    'ProgressSnapshot',
    
    # 告警系统
    'AlertSystem',
    'AlertConfig',
    'AlertRule',
    'Alert',
    'AlertLevel',
    'AlertType',
    
    # 工具函数
    'plot_training_metrics',
    'plot_system_metrics',
    'plot_progress_tracker',
    'export_monitor_data',
    'export_progress_data',
    'export_alert_data',
    'generate_monitor_report',
    'analyze_performance_trends',
    'detect_performance_anomalies',
    'discover_experiments',
    'load_experiment_snapshot',
    
    # Web监控
    'WebMonitor',
    'WebMonitorServer',
    'create_web_monitor',
]
