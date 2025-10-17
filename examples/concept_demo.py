#!/usr/bin/env python3
"""
Monitor模块核心功能演示
展示monitor模块的基本概念和功能
"""

import sys
import time
import random
from pathlib import Path

# 添加项目路径
sys.path.append('/home/rczx/workspace/sxy/lab/NeuroTrain')

def demo_concepts():
    """演示monitor模块的核心概念"""
    print("🚀 Monitor模块功能演示")
    print("=" * 60)
    
    print("\n📊 1. 训练监控器 (TrainingMonitor)")
    print("-" * 40)
    print("功能:")
    print("  ✅ 实时监控系统资源 (CPU、内存、GPU)")
    print("  ✅ 跟踪训练指标 (损失值、学习率、吞吐量)")
    print("  ✅ 自动数据收集和存储")
    print("  ✅ 阈值告警和异常检测")
    print("  ✅ 支持分布式训练")
    
    print("\n使用方式:")
    print("""
    from src.monitor import TrainingMonitor
    
    # 创建监控器
    monitor = TrainingMonitor()
    
    # 使用上下文管理器自动启动/停止
    with monitor:
        for epoch in range(100):
            for step in range(1000):
                # 训练代码
                loss = model.train_step(batch)
                
                # 更新监控数据
                monitor.update_training_metrics(
                    epoch=epoch, step=step, loss=loss
                )
    """)
    
    print("\n⏱️  2. 进度跟踪器 (ProgressTracker)")
    print("-" * 40)
    print("功能:")
    print("  ✅ 精确的ETA计算")
    print("  ✅ 训练进度可视化")
    print("  ✅ 性能统计分析")
    print("  ✅ 进度数据持久化")
    
    print("\n使用方式:")
    print("""
    from src.monitor import ProgressTracker
    
    # 创建进度跟踪器
    progress_tracker = ProgressTracker()
    
    # 开始训练跟踪
    progress_tracker.start_training(total_epochs=100, steps_per_epoch=1000)
    
    for epoch in range(100):
        progress_tracker.start_epoch(epoch)
        
        for step in range(1000):
            progress_tracker.start_step(step)
            
            # 训练代码
            time.sleep(0.01)  # 模拟训练时间
            
            progress_tracker.end_step()
            
            # 显示进度条
            if step % 100 == 0:
                print(progress_tracker.format_progress_bar())
        
        progress_tracker.end_epoch()
    
    progress_tracker.end_training()
    """)
    
    print("\n🚨 3. 告警系统 (AlertSystem)")
    print("-" * 40)
    print("功能:")
    print("  ✅ 多种告警类型 (阈值、趋势、异常)")
    print("  ✅ 灵活的告警规则配置")
    print("  ✅ 多渠道通知 (控制台、文件、邮件)")
    print("  ✅ 告警聚合防止风暴")
    
    print("\n使用方式:")
    print("""
    from src.monitor import AlertSystem, AlertLevel
    
    # 创建告警系统
    alert_system = AlertSystem()
    
    # 添加告警规则
    loss_rule = alert_system.create_threshold_rule(
        name="high_loss",
        threshold_value=1.0,
        operator=">",
        level=AlertLevel.WARNING,
        cooldown=60.0
    )
    alert_system.add_rule(loss_rule)
    
    # 更新数据触发告警检查
    alert_system.update_data('loss', 0.5)
    alert_system.update_data('loss', 1.2)  # 触发告警
    """)
    
    print("\n📈 4. 监控工具 (MonitorUtils)")
    print("-" * 40)
    print("功能:")
    print("  ✅ 自动生成监控图表")
    print("  ✅ 数据导出 (JSON、CSV、Parquet)")
    print("  ✅ HTML监控报告生成")
    print("  ✅ 性能趋势分析")
    print("  ✅ 异常检测")
    
    print("\n使用方式:")
    print("""
    from src.monitor.monitor_utils import (
        plot_training_metrics, generate_monitor_report,
        analyze_performance_trends
    )
    
    # 绘制训练指标图表
    plot_files = plot_training_metrics(
        monitor=monitor,
        output_dir=Path("charts"),
        metrics=['loss', 'learning_rate', 'throughput']
    )
    
    # 生成完整监控报告
    report_file = generate_monitor_report(
        monitor=monitor,
        progress_tracker=progress_tracker,
        alert_system=alert_system,
        output_dir=Path("reports")
    )
    
    # 分析性能趋势
    trends = analyze_performance_trends(monitor=monitor)
    """)
    
    print("\n🎯 5. 实际应用场景")
    print("-" * 40)
    print("✅ 深度学习训练监控")
    print("  - 实时监控训练过程中的各种指标")
    print("  - 自动检测训练异常和性能问题")
    print("  - 提供详细的训练报告和分析")
    
    print("\n✅ 分布式训练支持")
    print("  - 支持多GPU和多节点训练监控")
    print("  - 自动同步分布式训练指标")
    print("  - 提供全局训练状态视图")
    
    print("\n✅ 长期训练管理")
    print("  - 支持长时间训练的进度跟踪")
    print("  - 自动保存训练检查点")
    print("  - 提供训练恢复和继续功能")
    
    print("\n✅ 实验管理")
    print("  - 完整的训练数据记录")
    print("  - 自动生成实验报告")
    print("  - 支持实验对比和分析")
    
    print("\n📋 6. 配置选项")
    print("-" * 40)
    print("MonitorConfig:")
    print("  - log_interval: 日志记录间隔")
    print("  - save_interval: 数据保存间隔")
    print("  - enable_gpu_monitor: 启用GPU监控")
    print("  - memory_threshold: 内存告警阈值")
    print("  - loss_threshold: 损失值告警阈值")
    
    print("\nProgressConfig:")
    print("  - window_size: 滑动窗口大小")
    print("  - min_samples: 最小样本数")
    print("  - enable_console_output: 控制台输出")
    
    print("\nAlertConfig:")
    print("  - enable_email_output: 邮件通知")
    print("  - aggregation_window: 告警聚合窗口")
    print("  - max_alerts_per_window: 最大告警数")
    
    print("\n🎉 总结")
    print("=" * 60)
    print("Monitor模块提供了完整的训练监控解决方案:")
    print("  🔍 实时监控 - 系统资源和训练指标")
    print("  📊 进度跟踪 - 精确的ETA和性能分析")
    print("  🚨 智能告警 - 多种告警机制和异常检测")
    print("  📈 数据可视化 - 自动生成图表和报告")
    print("  🔧 易于集成 - 简单的API设计")
    print("  ⚡ 高性能 - 优化的数据结构和算法")
    
    print("\n💡 使用建议:")
    print("  1. 在训练开始时创建监控器")
    print("  2. 定期更新训练指标")
    print("  3. 配置合适的告警规则")
    print("  4. 定期查看监控报告")
    print("  5. 根据监控数据优化训练")

def simulate_monitoring():
    """模拟监控过程"""
    print("\n🔄 模拟监控过程")
    print("-" * 40)
    
    # 模拟训练数据
    print("模拟训练过程...")
    
    base_loss = 1.0
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}:")
        for step in range(5):
            # 模拟损失值下降
            loss = base_loss - (epoch * 5 + step) * 0.1 + random.uniform(-0.05, 0.05)
            throughput = random.uniform(80, 120)
            
            print(f"  Step {step + 1}: Loss={loss:.3f}, Throughput={throughput:.1f} samples/s")
            
            # 模拟系统资源
            cpu_usage = random.uniform(20, 80)
            memory_usage = random.uniform(30, 70)
            print(f"    System: CPU={cpu_usage:.1f}%, Memory={memory_usage:.1f}%")
            
            time.sleep(0.1)
    
    print("\n✅ 模拟完成")
    print("在实际使用中，这些数据会被自动收集和分析")

if __name__ == "__main__":
    demo_concepts()
    simulate_monitoring()
    
    print("\n" + "=" * 60)
    print("📚 更多信息请查看:")
    print("  - docs/monitor_module.md (详细文档)")
    print("  - examples/monitor_demo.py (完整示例)")
    print("  - src/monitor/ (源代码)")
    print("=" * 60)
