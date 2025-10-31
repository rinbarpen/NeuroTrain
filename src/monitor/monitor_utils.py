"""
监控工具函数 - 训练监控相关的实用工具和辅助函数

提供监控数据的可视化、分析、导出等功能。
"""

import json
import csv
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import seaborn as sns

from src.recorder.meter import Meter
from .training_monitor import TrainingMonitor, SystemMetrics, TrainingMetrics
from .progress_tracker import ProgressTracker, ProgressSnapshot
from .alert_system import AlertSystem, Alert


def plot_training_metrics(monitor: TrainingMonitor, 
                         output_dir: Path,
                         metrics: Optional[List[str]] = None,
                         figsize: Tuple[int, int] = (12, 8)) -> List[Path]:
    """
    绘制训练指标图表
    
    Args:
        monitor: 训练监控器
        output_dir: 输出目录
        metrics: 要绘制的指标列表，None表示绘制所有指标
        figsize: 图表大小
    
    Returns:
        生成的图片文件路径列表
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []
    
    # 设置matplotlib中文字体为英语
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # 获取训练指标历史
    training_history = monitor.training_metrics_history
    if not training_history:
        print("No training metrics history available")
        return generated_files
    
    # 准备数据
    timestamps = [m.timestamp for m in training_history]
    data = {
        'loss': [m.loss for m in training_history],
        'learning_rate': [m.learning_rate for m in training_history],
        'throughput': [m.throughput for m in training_history]
    }
    
    # 确定要绘制的指标
    if metrics is None:
        metrics = list(data.keys())
    
    # 创建子图
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        if metric not in data:
            continue
            
        ax = axes[i]
        ax.plot(timestamps, data[metric], linewidth=2)
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        
        # 设置x轴格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.xlabel('Time')
    plt.tight_layout()
    
    # 保存图片
    output_file = output_dir / 'training_metrics.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append(output_file)
    
    return generated_files


def plot_system_metrics(monitor: TrainingMonitor,
                       output_dir: Path,
                       metrics: List[str] = None,
                       figsize: Tuple[int, int] = (12, 8)) -> List[Path]:
    """
    绘制系统指标图表
    
    Args:
        monitor: 训练监控器
        output_dir: 输出目录
        metrics: 要绘制的指标列表
        figsize: 图表大小
    
    Returns:
        生成的图片文件路径列表
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []
    
    # 设置matplotlib中文字体为英语
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # 获取系统指标历史
    system_history = monitor.system_metrics_history
    if not system_history:
        print("No system metrics history available")
        return generated_files
    
    # 准备数据
    timestamps = [m.timestamp for m in system_history]
    data = {
        'cpu_percent': [m.cpu_percent for m in system_history],
        'memory_percent': [m.memory_percent for m in system_history],
        'memory_used_gb': [m.memory_used_gb for m in system_history],
        'gpu_utilization': [m.gpu_utilization for m in system_history],
        'gpu_memory_used_gb': [m.gpu_memory_used_gb for m in system_history]
    }
    
    # 确定要绘制的指标
    if metrics is None:
        metrics = list(data.keys())
    
    # 创建子图
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        if metric not in data:
            continue
            
        ax = axes[i]
        ax.plot(timestamps, data[metric], linewidth=2)
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        
        # 设置x轴格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.xlabel('Time')
    plt.tight_layout()
    
    # 保存图片
    output_file = output_dir / 'system_metrics.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append(output_file)
    
    return generated_files


def plot_progress_tracker(progress_tracker: ProgressTracker,
                         output_dir: Path,
                         figsize: Tuple[int, int] = (15, 10)) -> List[Path]:
    """
    绘制进度跟踪器图表
    
    Args:
        progress_tracker: 进度跟踪器
        output_dir: 输出目录
        figsize: 图表大小
    
    Returns:
        生成的图片文件路径列表
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []
    
    # 设置matplotlib中文字体为英语
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # 获取进度历史
    progress_history = progress_tracker.progress_history
    if not progress_history:
        print("No progress history available")
        return generated_files
    
    # 准备数据
    timestamps = [p.timestamp for p in progress_history]
    epochs = [p.epoch for p in progress_history]
    steps = [p.step for p in progress_history]
    epoch_progress = [p.epoch_progress for p in progress_history]
    total_progress = [p.total_progress for p in progress_history]
    step_times = [p.step_time for p in progress_history]
    throughput = [p.throughput for p in progress_history]
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 进度图
    ax1 = axes[0, 0]
    ax1.plot(timestamps, epoch_progress, label='Epoch Progress', linewidth=2)
    ax1.plot(timestamps, total_progress, label='Total Progress', linewidth=2)
    ax1.set_ylabel('Progress (%)')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 步骤时间图
    ax2 = axes[0, 1]
    ax2.plot(timestamps, step_times, linewidth=2, color='orange')
    ax2.set_ylabel('Step Time (s)')
    ax2.set_title('Step Time Over Time')
    ax2.grid(True, alpha=0.3)
    
    # 吞吐量图
    ax3 = axes[1, 0]
    ax3.plot(timestamps, throughput, linewidth=2, color='green')
    ax3.set_ylabel('Throughput (samples/s)')
    ax3.set_title('Training Throughput')
    ax3.grid(True, alpha=0.3)
    
    # ETA图
    ax4 = axes[1, 1]
    eta_total_hours = []
    for p in progress_history:
        if p.eta_total:
            eta_total_hours.append(p.eta_total.total_seconds() / 3600)
        else:
            eta_total_hours.append(None)
    
    # 过滤None值
    valid_indices = [i for i, eta in enumerate(eta_total_hours) if eta is not None]
    if valid_indices:
        valid_timestamps = [timestamps[i] for i in valid_indices]
        valid_eta = [eta_total_hours[i] for i in valid_indices]
        ax4.plot(valid_timestamps, valid_eta, linewidth=2, color='red')
        ax4.set_ylabel('ETA (hours)')
        ax4.set_title('Estimated Time to Completion')
        ax4.grid(True, alpha=0.3)
    
    # 设置x轴格式
    for ax in axes.flat:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # 保存图片
    output_file = output_dir / 'progress_tracker.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append(output_file)
    
    return generated_files


def export_monitor_data(monitor: TrainingMonitor,
                       output_dir: Path,
                       formats: List[str] = ['json', 'csv', 'parquet']) -> List[Path]:
    """
    导出监控数据
    
    Args:
        monitor: 训练监控器
        output_dir: 输出目录
        formats: 导出格式列表
    
    Returns:
        生成的文件路径列表
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []
    
    # 导出训练指标
    if monitor.training_metrics_history:
        training_data = []
        for metrics in monitor.training_metrics_history:
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
        
        df_training = pd.DataFrame(training_data)
        
        for fmt in formats:
            if fmt == 'json':
                file_path = output_dir / 'training_metrics.json'
                df_training.to_json(file_path, orient='records', indent=2)
                generated_files.append(file_path)
            elif fmt == 'csv':
                file_path = output_dir / 'training_metrics.csv'
                df_training.to_csv(file_path, index=False)
                generated_files.append(file_path)
            elif fmt == 'parquet':
                file_path = output_dir / 'training_metrics.parquet'
                df_training.to_parquet(file_path, index=False)
                generated_files.append(file_path)
    
    # 导出系统指标
    if monitor.system_metrics_history:
        system_data = []
        for metrics in monitor.system_metrics_history:
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
        
        df_system = pd.DataFrame(system_data)
        
        for fmt in formats:
            if fmt == 'json':
                file_path = output_dir / 'system_metrics.json'
                df_system.to_json(file_path, orient='records', indent=2)
                generated_files.append(file_path)
            elif fmt == 'csv':
                file_path = output_dir / 'system_metrics.csv'
                df_system.to_csv(file_path, index=False)
                generated_files.append(file_path)
            elif fmt == 'parquet':
                file_path = output_dir / 'system_metrics.parquet'
                df_system.to_parquet(file_path, index=False)
                generated_files.append(file_path)
    
    return generated_files


def export_progress_data(progress_tracker: ProgressTracker,
                        output_dir: Path,
                        formats: List[str] = ['json', 'csv', 'parquet']) -> List[Path]:
    """
    导出进度跟踪数据
    
    Args:
        progress_tracker: 进度跟踪器
        output_dir: 输出目录
        formats: 导出格式列表
    
    Returns:
        生成的文件路径列表
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []
    
    if not progress_tracker.progress_history:
        print("No progress history available")
        return generated_files
    
    # 准备数据
    progress_data = []
    for snapshot in progress_tracker.progress_history:
        progress_data.append({
            'timestamp': snapshot.timestamp.isoformat(),
            'epoch': snapshot.epoch,
            'step': snapshot.step,
            'total_steps': snapshot.total_steps,
            'total_epochs': snapshot.total_epochs,
            'step_time': snapshot.step_time,
            'epoch_time': snapshot.epoch_time,
            'throughput': snapshot.throughput,
            'epoch_progress': snapshot.epoch_progress,
            'total_progress': snapshot.total_progress,
            'eta_epoch': str(snapshot.eta_epoch) if snapshot.eta_epoch else None,
            'eta_total': str(snapshot.eta_total) if snapshot.eta_total else None
        })
    
    df_progress = pd.DataFrame(progress_data)
    
    for fmt in formats:
        if fmt == 'json':
            file_path = output_dir / 'progress_data.json'
            df_progress.to_json(file_path, orient='records', indent=2)
            generated_files.append(file_path)
        elif fmt == 'csv':
            file_path = output_dir / 'progress_data.csv'
            df_progress.to_csv(file_path, index=False)
            generated_files.append(file_path)
        elif fmt == 'parquet':
            file_path = output_dir / 'progress_data.parquet'
            df_progress.to_parquet(file_path, index=False)
            generated_files.append(file_path)
    
    return generated_files


def export_alert_data(alert_system: AlertSystem,
                     output_dir: Path,
                     formats: List[str] = ['json', 'csv']) -> List[Path]:
    """
    导出告警数据
    
    Args:
        alert_system: 告警系统
        output_dir: 输出目录
        formats: 导出格式列表
    
    Returns:
        生成的文件路径列表
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []
    
    if not alert_system.alert_history:
        print("No alert history available")
        return generated_files
    
    # 准备告警数据
    alert_data = []
    for alert in alert_system.alert_history:
        alert_data.append({
            'timestamp': alert.timestamp.isoformat(),
            'rule_name': alert.rule_name,
            'alert_type': alert.alert_type.value,
            'level': alert.level.value,
            'message': alert.message,
            'value': alert.value,
            'threshold': alert.threshold,
            'context': json.dumps(alert.context)
        })
    
    df_alerts = pd.DataFrame(alert_data)
    
    for fmt in formats:
        if fmt == 'json':
            file_path = output_dir / 'alerts.json'
            df_alerts.to_json(file_path, orient='records', indent=2)
            generated_files.append(file_path)
        elif fmt == 'csv':
            file_path = output_dir / 'alerts.csv'
            df_alerts.to_csv(file_path, index=False)
            generated_files.append(file_path)
    
    return generated_files


def generate_monitor_report(monitor: TrainingMonitor,
                           progress_tracker: Optional[ProgressTracker] = None,
                           alert_system: Optional[AlertSystem] = None,
                           output_dir: Path = None) -> Path:
    """
    生成监控报告
    
    Args:
        monitor: 训练监控器
        progress_tracker: 进度跟踪器（可选）
        alert_system: 告警系统（可选）
        output_dir: 输出目录
    
    Returns:
        生成的报告文件路径
    """
    if output_dir is None:
        output_dir = Path("monitor_reports")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 生成图表
    plot_files = []
    plot_files.extend(plot_training_metrics(monitor, output_dir))
    plot_files.extend(plot_system_metrics(monitor, output_dir))
    
    if progress_tracker:
        plot_files.extend(plot_progress_tracker(progress_tracker, output_dir))
    
    # 导出数据
    data_files = []
    data_files.extend(export_monitor_data(monitor, output_dir))
    
    if progress_tracker:
        data_files.extend(export_progress_data(progress_tracker, output_dir))
    
    if alert_system:
        data_files.extend(export_alert_data(alert_system, output_dir))
    
    # 生成HTML报告
    report_file = output_dir / f"monitor_report_{timestamp}.html"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Training Monitor Report - {timestamp}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2 {{ color: #333; }}
            .summary {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; }}
            .metrics {{ margin: 20px 0; }}
            .chart {{ text-align: center; margin: 20px 0; }}
            .chart img {{ max-width: 100%; height: auto; }}
            .files {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; }}
            ul {{ list-style-type: none; padding: 0; }}
            li {{ margin: 5px 0; }}
            a {{ color: #0066cc; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <h1>Training Monitor Report</h1>
        <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <div class="summary">
            <h2>Summary</h2>
            <p><strong>Training Duration:</strong> {len(monitor.training_metrics_history)} data points</p>
            <p><strong>System Metrics:</strong> {len(monitor.system_metrics_history)} data points</p>
            {f'<p><strong>Progress Tracking:</strong> {len(progress_tracker.progress_history)} snapshots</p>' if progress_tracker else ''}
            {f'<p><strong>Alerts:</strong> {len(alert_system.alert_history)} alerts</p>' if alert_system else ''}
        </div>
        
        <div class="metrics">
            <h2>Performance Metrics</h2>
            <p><strong>Average Loss:</strong> {monitor.training_meters['loss'].avg:.4f}</p>
            <p><strong>Average Throughput:</strong> {monitor.training_meters['throughput'].avg:.2f} samples/s</p>
            <p><strong>Average CPU Usage:</strong> {monitor.system_meters.get('cpu_percent', Meter('dummy')).avg:.1f}%</p>
            <p><strong>Average Memory Usage:</strong> {monitor.system_meters.get('memory_percent', Meter('dummy')).avg:.1f}%</p>
        </div>
        
        <div class="chart">
            <h2>Training Metrics</h2>
            <img src="training_metrics.png" alt="Training Metrics">
        </div>
        
        <div class="chart">
            <h2>System Metrics</h2>
            <img src="system_metrics.png" alt="System Metrics">
        </div>
        
        {f'''
        <div class="chart">
            <h2>Progress Tracking</h2>
            <img src="progress_tracker.png" alt="Progress Tracker">
        </div>
        ''' if progress_tracker else ''}
        
        <div class="files">
            <h2>Data Files</h2>
            <ul>
                {''.join([f'<li><a href="{f.name}">{f.name}</a></li>' for f in data_files])}
            </ul>
        </div>
        
        <div class="files">
            <h2>Chart Files</h2>
            <ul>
                {''.join([f'<li><a href="{f.name}">{f.name}</a></li>' for f in plot_files])}
            </ul>
        </div>
    </body>
    </html>
    """
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return report_file


def analyze_performance_trends(monitor: TrainingMonitor,
                              window_size: int = 100) -> Dict[str, Any]:
    """
    分析性能趋势
    
    Args:
        monitor: 训练监控器
        window_size: 分析窗口大小
    
    Returns:
        趋势分析结果
    """
    results = {}
    
    # 分析训练指标趋势
    if monitor.training_metrics_history:
        losses = [m.loss for m in monitor.training_metrics_history]
        throughputs = [m.throughput for m in monitor.training_metrics_history]
        
        if len(losses) >= window_size:
            recent_losses = losses[-window_size:]
            recent_throughputs = throughputs[-window_size:]
            
            # 计算趋势
            loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
            throughput_trend = np.polyfit(range(len(recent_throughputs)), recent_throughputs, 1)[0]
            
            results['loss_trend'] = {
                'slope': loss_trend,
                'direction': 'decreasing' if loss_trend < 0 else 'increasing',
                'magnitude': abs(loss_trend)
            }
            
            results['throughput_trend'] = {
                'slope': throughput_trend,
                'direction': 'increasing' if throughput_trend > 0 else 'decreasing',
                'magnitude': abs(throughput_trend)
            }
    
    # 分析系统指标趋势
    if monitor.system_metrics_history:
        cpu_usage = [m.cpu_percent for m in monitor.system_metrics_history]
        memory_usage = [m.memory_percent for m in monitor.system_metrics_history]
        
        if len(cpu_usage) >= window_size:
            recent_cpu = cpu_usage[-window_size:]
            recent_memory = memory_usage[-window_size:]
            
            cpu_trend = np.polyfit(range(len(recent_cpu)), recent_cpu, 1)[0]
            memory_trend = np.polyfit(range(len(recent_memory)), recent_memory, 1)[0]
            
            results['cpu_trend'] = {
                'slope': cpu_trend,
                'direction': 'increasing' if cpu_trend > 0 else 'decreasing',
                'magnitude': abs(cpu_trend)
            }
            
            results['memory_trend'] = {
                'slope': memory_trend,
                'direction': 'increasing' if memory_trend > 0 else 'decreasing',
                'magnitude': abs(memory_trend)
            }
    
    return results


def detect_performance_anomalies(monitor: TrainingMonitor,
                                method: str = 'zscore',
                                threshold: float = 3.0) -> Dict[str, List[int]]:
    """
    检测性能异常
    
    Args:
        monitor: 训练监控器
        method: 异常检测方法 ('zscore', 'iqr')
        threshold: 异常阈值
    
    Returns:
        异常点索引字典
    """
    anomalies = {}
    
    if not monitor.training_metrics_history:
        return anomalies
    
    # 提取指标数据
    losses = [m.loss for m in monitor.training_metrics_history]
    throughputs = [m.throughput for m in monitor.training_metrics_history]
    
    # 检测损失异常
    if method == 'zscore':
        loss_mean = np.mean(losses)
        loss_std = np.std(losses)
        if loss_std > 0:
            loss_zscore = np.abs(np.array(losses) - loss_mean) / loss_std
            anomalies['loss'] = np.where(loss_zscore > threshold)[0].tolist()
    
    elif method == 'iqr':
        loss_q1 = np.percentile(losses, 25)
        loss_q3 = np.percentile(losses, 75)
        loss_iqr = loss_q3 - loss_q1
        loss_lower = loss_q1 - 1.5 * loss_iqr
        loss_upper = loss_q3 + 1.5 * loss_iqr
        anomalies['loss'] = np.where((np.array(losses) < loss_lower) | 
                                    (np.array(losses) > loss_upper))[0].tolist()
    
    # 检测吞吐量异常
    if method == 'zscore':
        throughput_mean = np.mean(throughputs)
        throughput_std = np.std(throughputs)
        if throughput_std > 0:
            throughput_zscore = np.abs(np.array(throughputs) - throughput_mean) / throughput_std
            anomalies['throughput'] = np.where(throughput_zscore > threshold)[0].tolist()
    
    elif method == 'iqr':
        throughput_q1 = np.percentile(throughputs, 25)
        throughput_q3 = np.percentile(throughputs, 75)
        throughput_iqr = throughput_q3 - throughput_q1
        throughput_lower = throughput_q1 - 1.5 * throughput_iqr
        throughput_upper = throughput_q3 + 1.5 * throughput_iqr
        anomalies['throughput'] = np.where((np.array(throughputs) < throughput_lower) | 
                                         (np.array(throughputs) > throughput_upper))[0].tolist()
    
    return anomalies
