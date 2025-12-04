"""
进度跟踪器 - 训练进度跟踪和预估功能

提供训练过程中的进度跟踪、ETA计算、性能分析等功能。
"""

import time
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path

from src.recorder.meter import Meter
from src.utils.ndict import NDict


@dataclass
class ProgressConfig:
    """进度跟踪配置"""
    # 时间窗口配置
    window_size: int = 100  # 滑动窗口大小
    min_samples: int = 10   # 最小样本数用于计算ETA
    
    # 更新频率
    update_interval: float = 1.0  # 更新间隔（秒）
    
    # 输出配置
    enable_console_output: bool = True
    enable_file_output: bool = False
    output_file: Optional[Path] = None


@dataclass
class ProgressSnapshot:
    """进度快照"""
    timestamp: datetime = field(default_factory=datetime.now)
    epoch: int = 0
    step: int = 0
    total_steps: int = 0
    total_epochs: int = 0
    
    # 性能指标
    step_time: float = 0.0  # 单步时间（秒）
    epoch_time: float = 0.0  # 单轮时间（秒）
    throughput: float = 0.0  # 吞吐量（samples/second）
    
    # 进度百分比
    epoch_progress: float = 0.0  # 当前轮进度百分比
    total_progress: float = 0.0  # 总体进度百分比
    
    # 时间预估
    eta_epoch: Optional[timedelta] = None  # 当前轮剩余时间
    eta_total: Optional[timedelta] = None  # 总体剩余时间


class ProgressTracker:
    """进度跟踪器"""
    
    def __init__(self, config: Optional[ProgressConfig] = None):
        self.config = config or ProgressConfig()
        
        # 进度状态
        self.current_epoch = 0
        self.current_step = 0
        self.total_epochs = 0
        self.total_steps = 0
        self.steps_per_epoch = 0
        
        # 时间跟踪
        self.epoch_start_time: Optional[float] = None
        self.step_start_time: Optional[float] = None
        self.training_start_time: Optional[float] = None
        
        # 历史记录
        self.progress_history: List[ProgressSnapshot] = []
        self.step_times: List[float] = []
        self.epoch_times: List[float] = []
        
        # 性能计量器
        self.step_time_meter = Meter('step_time', ':3f')
        self.epoch_time_meter = Meter('epoch_time', ':3f')
        self.throughput_meter = Meter('throughput', ':2f')
        
        # 状态标志
        self.is_training = False
        self.is_epoch_active = False
    
    def start_training(self, total_epochs: int, steps_per_epoch: int):
        """开始训练跟踪"""
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        
        self.current_epoch = 0
        self.current_step = 0
        self.training_start_time = time.time()
        self.is_training = True
        
        # 清空历史记录
        self.progress_history.clear()
        self.step_times.clear()
        self.epoch_times.clear()
        
        # 重置计量器
        self.step_time_meter.reset()
        self.epoch_time_meter.reset()
        self.throughput_meter.reset()
    
    def start_epoch(self, epoch: int):
        """开始新轮次"""
        if not self.is_training:
            raise RuntimeError("训练未开始，请先调用start_training()")
        
        self.current_epoch = epoch
        self.current_step = 0
        self.epoch_start_time = time.time()
        self.is_epoch_active = True
    
    def start_step(self, step: int, batch_size: int = 1):
        """开始新步骤"""
        if not self.is_epoch_active:
            raise RuntimeError("轮次未开始，请先调用start_epoch()")
        
        self.current_step = step
        self.step_start_time = time.time()
    
    def end_step(self, batch_size: int = 1):
        """结束步骤"""
        if self.step_start_time is None:
            return
        
        step_time = time.time() - self.step_start_time
        self.step_times.append(step_time)
        self.step_time_meter.update(step_time)
        
        # 计算吞吐量
        throughput = batch_size / step_time if step_time > 0 else 0
        self.throughput_meter.update(throughput)
        
        # 更新进度
        self._update_progress(batch_size)
        
        self.step_start_time = None
    
    def end_epoch(self):
        """结束轮次"""
        if not self.is_epoch_active or self.epoch_start_time is None:
            return
        
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        self.epoch_time_meter.update(epoch_time)
        
        self.is_epoch_active = False
        self.epoch_start_time = None
    
    def end_training(self):
        """结束训练"""
        self.is_training = False
        self.is_epoch_active = False
    
    def _update_progress(self, batch_size: int):
        """更新进度信息"""
        if not self.is_training:
            return
        
        # 计算当前轮进度
        epoch_progress = (self.current_step / self.steps_per_epoch) * 100
        
        # 计算总体进度
        completed_steps = (self.current_epoch * self.steps_per_epoch) + self.current_step
        total_progress = (completed_steps / self.total_steps) * 100
        
        # 计算ETA
        eta_epoch = self._calculate_epoch_eta()
        eta_total = self._calculate_total_eta()
        
        # 创建进度快照
        snapshot = ProgressSnapshot(
            epoch=self.current_epoch,
            step=self.current_step,
            total_steps=self.total_steps,
            total_epochs=self.total_epochs,
            step_time=self.step_times[-1] if self.step_times else 0.0,
            epoch_time=self.epoch_times[-1] if self.epoch_times else 0.0,
            throughput=self.throughput_meter.val,
            epoch_progress=epoch_progress,
            total_progress=total_progress,
            eta_epoch=eta_epoch,
            eta_total=eta_total
        )
        
        self.progress_history.append(snapshot)
        
        # 限制历史记录长度
        if len(self.progress_history) > self.config.window_size:
            self.progress_history = self.progress_history[-self.config.window_size:]
    
    def _calculate_epoch_eta(self) -> Optional[timedelta]:
        """计算当前轮剩余时间"""
        if not self.step_times or len(self.step_times) < self.config.min_samples:
            return None
        
        # 使用最近的步骤时间计算平均时间
        recent_times = self.step_times[-self.config.min_samples:]
        avg_step_time = np.mean(recent_times)
        
        # 计算剩余步骤
        remaining_steps = self.steps_per_epoch - self.current_step
        
        # 计算剩余时间
        remaining_time = remaining_steps * avg_step_time
        
        return timedelta(seconds=float(remaining_time))
    
    def _calculate_total_eta(self) -> Optional[timedelta]:
        """计算总体剩余时间"""
        if not self.step_times or len(self.step_times) < self.config.min_samples:
            return None
        
        # 使用最近的步骤时间计算平均时间
        recent_times = self.step_times[-self.config.min_samples:]
        avg_step_time = np.mean(recent_times)
        
        # 计算剩余步骤
        completed_steps = (self.current_epoch * self.steps_per_epoch) + self.current_step
        remaining_steps = self.total_steps - completed_steps
        
        # 计算剩余时间
        remaining_time = remaining_steps * avg_step_time
        
        return timedelta(seconds=float(remaining_time))
    
    def get_current_progress(self) -> Optional[ProgressSnapshot]:
        """获取当前进度"""
        return self.progress_history[-1] if self.progress_history else None
    
    def get_progress_summary(self) -> Dict[str, Union[float, int, str, None]]:
        """获取进度摘要"""
        current = self.get_current_progress()
        if not current:
            return {}
        
        return {
            'current_epoch': current.epoch,
            'current_step': current.step,
            'total_epochs': current.total_epochs,
            'total_steps': current.total_steps,
            'epoch_progress': current.epoch_progress,
            'total_progress': current.total_progress,
            'avg_step_time': self.step_time_meter.avg,
            'avg_epoch_time': self.epoch_time_meter.avg,
            'avg_throughput': self.throughput_meter.avg,
            'eta_epoch': str(current.eta_epoch) if current.eta_epoch else None,
            'eta_total': str(current.eta_total) if current.eta_total else None
        }
    
    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计"""
        def _safe_float(value: float) -> float:
            try:
                value = float(value)
            except (TypeError, ValueError):
                return 0.0
            if not math.isfinite(value):
                return 0.0
            return value

        def _clean_values(values) -> List[float]:
            cleaned: List[float] = []
            for item in values:
                try:
                    val = float(item)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(val):
                    cleaned.append(val)
            return cleaned

        def _calc_avg(values: List[float]) -> float:
            if not values:
                return 0.0
            return sum(values) / len(values)

        def _calc_std(values: List[float]) -> float:
            if len(values) <= 1:
                return 0.0
            return float(np.std(values))

        step_times = _clean_values(self.step_times)
        epoch_times = _clean_values(self.epoch_times)
        throughput_values = _clean_values(self.throughput_meter.vals.tolist())

        stats = {
            'avg_step_time': _calc_avg(step_times),
            'min_step_time': min(step_times) if step_times else 0.0,
            'max_step_time': max(step_times) if step_times else 0.0,
            'std_step_time': _calc_std(step_times),
            'avg_epoch_time': _calc_avg(epoch_times),
            'min_epoch_time': min(epoch_times) if epoch_times else 0.0,
            'max_epoch_time': max(epoch_times) if epoch_times else 0.0,
            'std_epoch_time': _calc_std(epoch_times),
            'avg_throughput': _calc_avg(throughput_values),
            'min_throughput': min(throughput_values) if throughput_values else 0.0,
            'max_throughput': max(throughput_values) if throughput_values else 0.0,
            'std_throughput': _calc_std(throughput_values)
        }
        
        return {key: _safe_float(value) for key, value in stats.items()}

    def record_remote_progress(self, data: Dict[str, Any]):
        """从远程推送的数据记录快照"""
        timestamp_str = data.get('timestamp')
        try:
            timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()
        except (TypeError, ValueError):
            timestamp = datetime.now()

        epoch = int(data.get('epoch', data.get('current_epoch', self.current_epoch)))
        step = int(data.get('step', data.get('current_step', self.current_step)))
        total_epochs = int(data.get('total_epochs', self.total_epochs))
        total_steps = int(data.get('total_steps', self.total_steps))
        step_time = float(data.get('step_time', 0.0))
        epoch_time = float(data.get('epoch_time', 0.0))
        throughput = float(data.get('throughput', 0.0))
        epoch_progress = float(data.get('epoch_progress', 0.0))
        total_progress = float(data.get('total_progress', 0.0))

        eta_epoch_seconds = data.get('eta_epoch_seconds')
        eta_total_seconds = data.get('eta_total_seconds')
        eta_epoch = timedelta(seconds=float(eta_epoch_seconds)) if eta_epoch_seconds is not None else None
        eta_total = timedelta(seconds=float(eta_total_seconds)) if eta_total_seconds is not None else None

        snapshot = ProgressSnapshot(
            timestamp=timestamp,
            epoch=epoch,
            step=step,
            total_steps=total_steps,
            total_epochs=total_epochs,
            step_time=step_time,
            epoch_time=epoch_time,
            throughput=throughput,
            epoch_progress=epoch_progress,
            total_progress=total_progress,
            eta_epoch=eta_epoch,
            eta_total=eta_total
        )

        self.current_epoch = epoch
        self.current_step = step
        self.total_epochs = total_epochs
        self.total_steps = total_steps
        self.steps_per_epoch = total_steps // total_epochs if total_epochs else self.steps_per_epoch

        self.progress_history.append(snapshot)
        if len(self.progress_history) > self.config.window_size:
            self.progress_history = self.progress_history[-self.config.window_size:]

        if step_time:
            self.step_times.append(step_time)
            self.step_time_meter.update(step_time)
        if epoch_time:
            self.epoch_times.append(epoch_time)
            self.epoch_time_meter.update(epoch_time)
        if throughput:
            self.throughput_meter.update(throughput)
    
    def format_progress_bar(self, width: int = 50, show_eta: bool = True) -> str:
        """格式化进度条"""
        current = self.get_current_progress()
        if not current:
            return "进度未知"
        
        # 计算进度条
        filled = int(width * current.total_progress / 100)
        bar = "█" * filled + "░" * (width - filled)
        
        # 基本信息
        info = f"Epoch {current.epoch}/{current.total_epochs} | Step {current.step}/{self.steps_per_epoch}"
        
        # ETA信息
        eta_info = ""
        if show_eta and current.eta_total:
            eta_info = f" | ETA: {current.eta_total}"
        
        # 性能信息
        perf_info = f" | {current.throughput:.1f} samples/s"
        
        return f"{info} | [{bar}] {current.total_progress:.1f}%{eta_info}{perf_info}"
    
    def save_progress(self, filepath: Optional[Path] = None):
        """保存进度数据"""
        if filepath is None:
            filepath = self.config.output_file
        
        if filepath is None:
            return
        
        # 准备数据
        data = {
            'config': {
                'total_epochs': self.total_epochs,
                'steps_per_epoch': self.steps_per_epoch,
                'total_steps': self.total_steps
            },
            'current_state': {
                'current_epoch': self.current_epoch,
                'current_step': self.current_step,
                'is_training': self.is_training,
                'is_epoch_active': self.is_epoch_active
            },
            'progress_history': [
                {
                    'timestamp': snapshot.timestamp.isoformat(),
                    'epoch': snapshot.epoch,
                    'step': snapshot.step,
                    'epoch_progress': snapshot.epoch_progress,
                    'total_progress': snapshot.total_progress,
                    'step_time': snapshot.step_time,
                    'epoch_time': snapshot.epoch_time,
                    'throughput': snapshot.throughput,
                    'eta_epoch': str(snapshot.eta_epoch) if snapshot.eta_epoch else None,
                    'eta_total': str(snapshot.eta_total) if snapshot.eta_total else None
                }
                for snapshot in self.progress_history
            ],
            'performance_stats': self.get_performance_stats()
        }
        
        # 保存到文件
        import json
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_progress(self, filepath: Path):
        """加载进度数据"""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # 恢复配置
        config = data['config']
        self.total_epochs = config['total_epochs']
        self.steps_per_epoch = config['steps_per_epoch']
        self.total_steps = config['total_steps']
        
        # 恢复状态
        state = data['current_state']
        self.current_epoch = state['current_epoch']
        self.current_step = state['current_step']
        self.is_training = state['is_training']
        self.is_epoch_active = state['is_epoch_active']
        
        # 恢复历史记录
        self.progress_history.clear()
        for item in data['progress_history']:
            snapshot = ProgressSnapshot(
                timestamp=datetime.fromisoformat(item['timestamp']),
                epoch=item['epoch'],
                step=item['step'],
                total_steps=self.total_steps,
                total_epochs=self.total_epochs,
                step_time=item['step_time'],
                epoch_time=item['epoch_time'],
                throughput=item['throughput'],
                epoch_progress=item['epoch_progress'],
                total_progress=item['total_progress'],
                eta_epoch=timedelta.fromisoformat(item['eta_epoch']) if item['eta_epoch'] else None,
                eta_total=timedelta.fromisoformat(item['eta_total']) if item['eta_total'] else None
            )
            self.progress_history.append(snapshot)
    
    def reset(self):
        """重置进度跟踪器"""
        self.current_epoch = 0
        self.current_step = 0
        self.total_epochs = 0
        self.total_steps = 0
        self.steps_per_epoch = 0
        
        self.epoch_start_time = None
        self.step_start_time = None
        self.training_start_time = None
        
        self.progress_history.clear()
        self.step_times.clear()
        self.epoch_times.clear()
        
        self.step_time_meter.reset()
        self.epoch_time_meter.reset()
        self.throughput_meter.reset()
        
        self.is_training = False
        self.is_epoch_active = False
