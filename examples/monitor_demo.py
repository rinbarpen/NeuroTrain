"""
训练监控模块使用示例

展示如何使用monitor模块进行训练监控、进度跟踪和告警。
"""

import time
import random
from pathlib import Path
import sys
sys.path.append('/home/rczx/workspace/sxy/lab/NeuroTrain')

from src.monitor import (
    TrainingMonitor, MonitorConfig,
    ProgressTracker, ProgressConfig,
    AlertSystem, AlertConfig, AlertLevel, AlertType
)


def demo_training_monitor():
    """演示训练监控器"""
    print("=== 训练监控器演示 ===")
    
    # 创建监控配置
    config = MonitorConfig(
        log_interval=0.5,  # 每0.5秒记录一次
        save_interval=5.0,  # 每5秒保存一次
        enable_console_output=True,
        log_dir=Path("monitor_logs")
    )
    
    # 创建监控器
    monitor = TrainingMonitor(config)
    
    # 使用上下文管理器自动启动/停止监控
    with monitor:
        print("开始模拟训练...")
        
        # 模拟训练过程
        for epoch in range(3):
            for step in range(10):
                # 模拟训练指标
                loss = 1.0 - (epoch * 10 + step) * 0.01 + random.uniform(-0.1, 0.1)
                lr = 0.001 * (0.9 ** epoch)
                throughput = random.uniform(50, 100)
                
                # 更新训练指标
                monitor.update_training_metrics(
                    epoch=epoch,
                    step=step,
                    loss=max(0, loss),
                    learning_rate=lr,
                    batch_size=32,
                    throughput=throughput
                )
                
                time.sleep(0.1)  # 模拟训练时间
        
        print("训练完成")
    
    # 获取监控摘要
    summary = monitor.get_summary()
    print(f"监控摘要: {summary}")


def demo_progress_tracker():
    """演示进度跟踪器"""
    print("\n=== 进度跟踪器演示 ===")
    
    # 创建进度跟踪器
    progress_tracker = ProgressTracker()
    
    # 开始训练跟踪
    total_epochs = 5
    steps_per_epoch = 20
    progress_tracker.start_training(total_epochs, steps_per_epoch)
    
    print("开始训练跟踪...")
    
    for epoch in range(total_epochs):
        progress_tracker.start_epoch(epoch)
        
        for step in range(steps_per_epoch):
            # 开始步骤
            progress_tracker.start_step(step, batch_size=32)
            
            # 模拟训练时间
            time.sleep(0.05)
            
            # 结束步骤
            progress_tracker.end_step()
            
            # 每5步显示一次进度
            if step % 5 == 0:
                progress_bar = progress_tracker.format_progress_bar(width=30)
                print(f"Epoch {epoch}: {progress_bar}")
        
        progress_tracker.end_epoch()
    
    progress_tracker.end_training()
    
    # 获取性能统计
    stats = progress_tracker.get_performance_stats()
    print(f"\n性能统计: {stats}")


def demo_alert_system():
    """演示告警系统"""
    print("\n=== 告警系统演示 ===")
    
    # 创建告警配置
    alert_config = AlertConfig(
        enable_console_output=True,
        log_file=Path("alert_logs/alerts.log")
    )
    
    # 创建告警系统
    alert_system = AlertSystem(alert_config)
    
    # 添加告警规则
    # 1. 损失值过高告警
    loss_rule = alert_system.create_threshold_rule(
        name="high_loss",
        threshold_value=0.8,
        operator=">",
        level=AlertLevel.WARNING,
        cooldown=10.0
    )
    alert_system.add_rule(loss_rule)
    
    # 2. 损失值异常告警
    anomaly_rule = alert_system.create_anomaly_rule(
        name="loss_anomaly",
        anomaly_method="zscore",
        anomaly_threshold=2.0,
        level=AlertLevel.ERROR,
        cooldown=5.0
    )
    alert_system.add_rule(anomaly_rule)
    
    # 3. 损失值下降趋势告警
    trend_rule = alert_system.create_trend_rule(
        name="loss_trend",
        trend_window=5,
        trend_threshold=0.05,
        level=AlertLevel.INFO,
        cooldown=15.0
    )
    alert_system.add_rule(trend_rule)
    
    print("开始模拟训练数据...")
    
    # 模拟训练数据
    base_loss = 1.0
    for step in range(50):
        # 模拟损失值变化
        if step < 20:
            loss = base_loss - step * 0.02 + random.uniform(-0.1, 0.1)
        elif step < 35:
            # 模拟异常
            loss = base_loss - 0.4 + random.uniform(-0.2, 0.2)
        else:
            loss = base_loss - 0.4 - (step - 35) * 0.01 + random.uniform(-0.05, 0.05)
        
        # 更新告警系统
        alert_system.update_data('loss', max(0, loss))
        
        time.sleep(0.1)
    
    # 获取告警摘要
    alert_summary = alert_system.get_alert_summary()
    print(f"\n告警摘要: {alert_summary}")


def demo_integrated_monitoring():
    """演示集成监控"""
    print("\n=== 集成监控演示 ===")
    
    # 创建所有监控组件
    monitor_config = MonitorConfig(
        log_interval=1.0,
        save_interval=10.0,
        log_dir=Path("integrated_logs")
    )
    monitor = TrainingMonitor(monitor_config)
    
    progress_tracker = ProgressTracker()
    
    alert_config = AlertConfig(
        enable_console_output=True,
        log_file=Path("integrated_logs/alerts.log")
    )
    alert_system = AlertSystem(alert_config)
    
    # 添加告警规则
    loss_rule = alert_system.create_threshold_rule(
        name="high_loss",
        threshold_value=0.5,
        operator=">",
        level=AlertLevel.WARNING
    )
    alert_system.add_rule(loss_rule)
    
    # 开始集成监控
    with monitor:
        # 开始训练跟踪
        total_epochs = 3
        steps_per_epoch = 15
        progress_tracker.start_training(total_epochs, steps_per_epoch)
        
        print("开始集成训练监控...")
        
        for epoch in range(total_epochs):
            progress_tracker.start_epoch(epoch)
            
            for step in range(steps_per_epoch):
                # 开始步骤
                progress_tracker.start_step(step, batch_size=32)
                
                # 模拟训练
                time.sleep(0.1)
                
                # 模拟训练指标
                loss = 1.0 - (epoch * steps_per_epoch + step) * 0.01 + random.uniform(-0.05, 0.05)
                lr = 0.001 * (0.9 ** epoch)
                throughput = random.uniform(60, 120)
                
                # 更新所有监控组件
                monitor.update_training_metrics(
                    epoch=epoch,
                    step=step,
                    loss=max(0, loss),
                    learning_rate=lr,
                    batch_size=32,
                    throughput=throughput
                )
                
                alert_system.update_data('loss', max(0, loss))
                
                # 结束步骤
                progress_tracker.end_step()
                
                # 每5步显示进度
                if step % 5 == 0:
                    progress_bar = progress_tracker.format_progress_bar(width=25)
                    print(f"Epoch {epoch}: {progress_bar}")
            
            progress_tracker.end_epoch()
        
        progress_tracker.end_training()
    
    # 生成监控报告
    from src.monitor.monitor_utils import generate_monitor_report
    report_file = generate_monitor_report(
        monitor=monitor,
        progress_tracker=progress_tracker,
        alert_system=alert_system,
        output_dir=Path("monitor_reports")
    )
    
    print(f"\n监控报告已生成: {report_file}")
    
    # 显示摘要
    monitor_summary = monitor.get_summary()
    progress_summary = progress_tracker.get_progress_summary()
    alert_summary = alert_system.get_alert_summary()
    
    print(f"\n监控摘要: {monitor_summary}")
    print(f"进度摘要: {progress_summary}")
    print(f"告警摘要: {alert_summary}")


if __name__ == "__main__":
    print("训练监控模块演示")
    print("=" * 50)
    
    # 运行各个演示
    demo_training_monitor()
    demo_progress_tracker()
    demo_alert_system()
    demo_integrated_monitoring()
    
    print("\n演示完成！")
