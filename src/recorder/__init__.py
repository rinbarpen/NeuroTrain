"""
记录器模块 - 用于训练过程中的指标记录和数据保存

主要组件:
- DataSaver: 数据保存器，用于异步保存训练数据
- MiniMeter: 轻量级指标计量器
- Meter: 完整功能的指标计量器，支持可视化和持久化
- MetricManager: 指标管理器，统一管理多个指标
- MetricRecorder: 指标记录器，用于训练过程中的完整指标记录
- ScoreAggregator: 分数聚合器，用于统计和聚合评估分数
"""

# 数据保存相关
from .data_saver import DataSaver

# 指标计量相关
from .meter import MiniMeter, Meter

# 指标管理相关
from .metric_manager import MetricManager

# 指标记录相关
from .meter_recorder import MeterRecorder

# 导出所有主要类
__all__ = [
    # 数据保存
    'DataSaver',
    
    # 指标计量
    'MiniMeter',
    'Meter',
    
    # 指标管理
    'MetricManager',
    
    # 指标记录
    'MeterRecorder',
]