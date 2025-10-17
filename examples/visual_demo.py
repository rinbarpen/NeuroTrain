#!/usr/bin/env python3
"""
Monitor模块可视化演示
展示monitor模块生成的图表和报告效果
"""

import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目路径
sys.path.append('/home/rczx/workspace/sxy/lab/NeuroTrain')

def create_sample_data():
    """创建示例监控数据"""
    print("📊 创建示例监控数据...")
    
    # 模拟训练数据
    epochs = 10
    steps_per_epoch = 50
    
    timestamps = []
    losses = []
    learning_rates = []
    throughputs = []
    cpu_usage = []
    memory_usage = []
    
    base_time = datetime.now()
    
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            timestamp = base_time + timedelta(seconds=(epoch * steps_per_epoch + step) * 2)
            timestamps.append(timestamp)
            
            # 模拟损失值下降
            loss = 1.0 - (epoch * steps_per_epoch + step) * 0.001 + random.uniform(-0.05, 0.05)
            losses.append(max(0, loss))
            
            # 模拟学习率衰减
            lr = 0.001 * (0.95 ** epoch)
            learning_rates.append(lr)
            
            # 模拟吞吐量
            throughput = random.uniform(80, 120)
            throughputs.append(throughput)
            
            # 模拟系统资源
            cpu_usage.append(random.uniform(20, 80))
            memory_usage.append(random.uniform(30, 70))
    
    return {
        'timestamps': timestamps,
        'losses': losses,
        'learning_rates': learning_rates,
        'throughputs': throughputs,
        'cpu_usage': cpu_usage,
        'memory_usage': memory_usage
    }

def plot_training_metrics_demo(data, output_dir):
    """绘制训练指标演示图表"""
    print("📈 生成训练指标图表...")
    
    # 设置matplotlib中文字体为英语
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # 损失值图表
    axes[0].plot(data['timestamps'], data['losses'], linewidth=2, color='red')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Over Time')
    axes[0].grid(True, alpha=0.3)
    
    # 学习率图表
    axes[1].plot(data['timestamps'], data['learning_rates'], linewidth=2, color='blue')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].grid(True, alpha=0.3)
    
    # 吞吐量图表
    axes[2].plot(data['timestamps'], data['throughputs'], linewidth=2, color='green')
    axes[2].set_ylabel('Throughput (samples/s)')
    axes[2].set_title('Training Throughput')
    axes[2].set_xlabel('Time')
    axes[2].grid(True, alpha=0.3)
    
    # 设置x轴格式
    import matplotlib.dates as mdates
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # 保存图表
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'training_metrics_demo.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 训练指标图表已保存: {output_file}")
    return output_file

def plot_system_metrics_demo(data, output_dir):
    """绘制系统指标演示图表"""
    print("💻 生成系统指标图表...")
    
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # CPU使用率图表
    axes[0].plot(data['timestamps'], data['cpu_usage'], linewidth=2, color='orange')
    axes[0].set_ylabel('CPU Usage (%)')
    axes[0].set_title('System CPU Usage')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 100)
    
    # 内存使用率图表
    axes[1].plot(data['timestamps'], data['memory_usage'], linewidth=2, color='purple')
    axes[1].set_ylabel('Memory Usage (%)')
    axes[1].set_title('System Memory Usage')
    axes[1].set_xlabel('Time')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 100)
    
    # 设置x轴格式
    import matplotlib.dates as mdates
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # 保存图表
    output_file = output_dir / 'system_metrics_demo.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 系统指标图表已保存: {output_file}")
    return output_file

def plot_progress_demo(output_dir):
    """绘制进度跟踪演示图表"""
    print("⏱️ 生成进度跟踪图表...")
    
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # 模拟进度数据
    epochs = 10
    steps_per_epoch = 50
    total_steps = epochs * steps_per_epoch
    
    timestamps = []
    epoch_progress = []
    total_progress = []
    step_times = []
    throughputs = []
    
    base_time = datetime.now()
    
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            timestamp = base_time + timedelta(seconds=(epoch * steps_per_epoch + step) * 2)
            timestamps.append(timestamp)
            
            # 计算进度
            current_step = epoch * steps_per_epoch + step
            epoch_progress.append((step / steps_per_epoch) * 100)
            total_progress.append((current_step / total_steps) * 100)
            
            # 模拟步骤时间和吞吐量
            step_times.append(random.uniform(0.1, 0.3))
            throughputs.append(random.uniform(80, 120))
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 进度图
    axes[0, 0].plot(timestamps, epoch_progress, label='Epoch Progress', linewidth=2)
    axes[0, 0].plot(timestamps, total_progress, label='Total Progress', linewidth=2)
    axes[0, 0].set_ylabel('Progress (%)')
    axes[0, 0].set_title('Training Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 步骤时间图
    axes[0, 1].plot(timestamps, step_times, linewidth=2, color='orange')
    axes[0, 1].set_ylabel('Step Time (s)')
    axes[0, 1].set_title('Step Time Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 吞吐量图
    axes[1, 0].plot(timestamps, throughputs, linewidth=2, color='green')
    axes[1, 0].set_ylabel('Throughput (samples/s)')
    axes[1, 0].set_title('Training Throughput')
    axes[1, 0].grid(True, alpha=0.3)
    
    # ETA图
    eta_hours = []
    for i, progress in enumerate(total_progress):
        if progress > 0:
            remaining_progress = 100 - progress
            avg_step_time = np.mean(step_times[:i+1])
            remaining_steps = (remaining_progress / 100) * total_steps
            eta_seconds = remaining_steps * avg_step_time
            eta_hours.append(eta_seconds / 3600)
        else:
            eta_hours.append(None)
    
    valid_eta = [eta for eta in eta_hours if eta is not None]
    valid_timestamps = timestamps[:len(valid_eta)]
    
    if valid_eta:
        axes[1, 1].plot(valid_timestamps, valid_eta, linewidth=2, color='red')
        axes[1, 1].set_ylabel('ETA (hours)')
        axes[1, 1].set_title('Estimated Time to Completion')
        axes[1, 1].grid(True, alpha=0.3)
    
    # 设置x轴格式
    import matplotlib.dates as mdates
    for ax in axes.flat:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # 保存图表
    output_file = output_dir / 'progress_tracker_demo.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 进度跟踪图表已保存: {output_file}")
    return output_file

def create_monitor_report_demo(data, output_dir):
    """创建监控报告演示"""
    print("📋 生成监控报告...")
    
    # 计算统计信息
    avg_loss = np.mean(data['losses'])
    min_loss = np.min(data['losses'])
    max_loss = np.max(data['losses'])
    
    avg_throughput = np.mean(data['throughputs'])
    min_throughput = np.min(data['throughputs'])
    max_throughput = np.max(data['throughputs'])
    
    avg_cpu = np.mean(data['cpu_usage'])
    avg_memory = np.mean(data['memory_usage'])
    
    # 生成HTML报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"monitor_report_demo_{timestamp}.html"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Training Monitor Report Demo - {timestamp}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
            h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            .summary {{ background-color: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
            .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
            .metric-card {{ background-color: #fff; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
            .metric-label {{ color: #7f8c8d; margin-top: 5px; }}
            .chart {{ text-align: center; margin: 30px 0; }}
            .chart img {{ max-width: 100%; height: auto; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .alert {{ background-color: #f39c12; color: white; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            .success {{ background-color: #27ae60; color: white; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            .info {{ background-color: #3498db; color: white; padding: 10px; border-radius: 5px; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Training Monitor Report Demo</h1>
            <p style="text-align: center; color: #7f8c8d;">Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="summary">
                <h2>📊 Training Summary</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-value">{len(data['losses'])}</div>
                        <div class="metric-label">Total Steps</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{avg_loss:.4f}</div>
                        <div class="metric-label">Average Loss</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{min_loss:.4f}</div>
                        <div class="metric-label">Minimum Loss</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{max_loss:.4f}</div>
                        <div class="metric-label">Maximum Loss</div>
                    </div>
                </div>
            </div>
            
            <div class="summary">
                <h2>⚡ Performance Metrics</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-value">{avg_throughput:.1f}</div>
                        <div class="metric-label">Avg Throughput (samples/s)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{min_throughput:.1f}</div>
                        <div class="metric-label">Min Throughput</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{max_throughput:.1f}</div>
                        <div class="metric-label">Max Throughput</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{avg_cpu:.1f}%</div>
                        <div class="metric-label">Average CPU Usage</div>
                    </div>
                </div>
            </div>
            
            <div class="summary">
                <h2>💻 System Resources</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-value">{avg_cpu:.1f}%</div>
                        <div class="metric-label">Average CPU Usage</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{avg_memory:.1f}%</div>
                        <div class="metric-label">Average Memory Usage</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{np.max(data['cpu_usage']):.1f}%</div>
                        <div class="metric-label">Peak CPU Usage</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{np.max(data['memory_usage']):.1f}%</div>
                        <div class="metric-label">Peak Memory Usage</div>
                    </div>
                </div>
            </div>
            
            <div class="chart">
                <h2>📈 Training Metrics</h2>
                <img src="training_metrics_demo.png" alt="Training Metrics">
            </div>
            
            <div class="chart">
                <h2>💻 System Metrics</h2>
                <img src="system_metrics_demo.png" alt="System Metrics">
            </div>
            
            <div class="chart">
                <h2>⏱️ Progress Tracking</h2>
                <img src="progress_tracker_demo.png" alt="Progress Tracker">
            </div>
            
            <div class="summary">
                <h2>🎯 Key Insights</h2>
                <div class="success">
                    <strong>✅ Training Progress:</strong> Loss decreased from {max_loss:.4f} to {min_loss:.4f}, showing good convergence.
                </div>
                <div class="info">
                    <strong>ℹ️ Performance:</strong> Average throughput of {avg_throughput:.1f} samples/s indicates efficient training.
                </div>
                <div class="alert">
                    <strong>⚠️ Resource Usage:</strong> Peak CPU usage of {np.max(data['cpu_usage']):.1f}% and memory usage of {np.max(data['memory_usage']):.1f}% are within normal ranges.
                </div>
            </div>
            
            <div class="summary">
                <h2>🔧 Monitor Module Features</h2>
                <ul>
                    <li><strong>Real-time Monitoring:</strong> Continuous tracking of system resources and training metrics</li>
                    <li><strong>Progress Tracking:</strong> Accurate ETA calculation and progress visualization</li>
                    <li><strong>Alert System:</strong> Configurable alerts for anomalies and thresholds</li>
                    <li><strong>Data Visualization:</strong> Automatic generation of charts and reports</li>
                    <li><strong>Performance Analysis:</strong> Trend analysis and anomaly detection</li>
                    <li><strong>Easy Integration:</strong> Simple API for seamless integration with training code</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ 监控报告已保存: {report_file}")
    return report_file

def main():
    """主函数"""
    print("🎨 Monitor模块可视化演示")
    print("=" * 60)
    
    # 创建输出目录
    output_dir = Path("monitor_demo_output")
    output_dir.mkdir(exist_ok=True)
    
    # 创建示例数据
    data = create_sample_data()
    
    # 生成各种图表
    plot_training_metrics_demo(data, output_dir)
    plot_system_metrics_demo(data, output_dir)
    plot_progress_demo(output_dir)
    
    # 生成监控报告
    report_file = create_monitor_report_demo(data, output_dir)
    
    print("\n🎉 演示完成！")
    print("=" * 60)
    print(f"📁 输出文件保存在: {output_dir.absolute()}")
    print(f"📋 监控报告: {report_file.name}")
    print("\n💡 这些图表展示了Monitor模块的实际效果:")
    print("  📊 训练指标图表 - 显示损失值、学习率、吞吐量变化")
    print("  💻 系统指标图表 - 显示CPU和内存使用情况")
    print("  ⏱️ 进度跟踪图表 - 显示训练进度和ETA")
    print("  📋 HTML报告 - 完整的监控报告和分析")
    
    print("\n🚀 在实际使用中，Monitor模块会:")
    print("  ✅ 自动收集这些数据")
    print("  ✅ 实时生成图表和报告")
    print("  ✅ 提供告警和异常检测")
    print("  ✅ 支持分布式训练监控")

if __name__ == "__main__":
    main()
