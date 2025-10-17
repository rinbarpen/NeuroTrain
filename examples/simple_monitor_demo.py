#!/usr/bin/env python3
"""
简单的Monitor模块演示
展示monitor模块的基本功能和使用效果
"""

import sys
import time
import random
from pathlib import Path

# 添加项目路径
sys.path.append('/home/rczx/workspace/sxy/lab/NeuroTrain')

def simple_demo():
    """简单演示"""
    print("🚀 Monitor模块演示开始")
    print("=" * 50)
    
    try:
        # 导入monitor模块
        from src.monitor import TrainingMonitor, ProgressTracker, AlertSystem, AlertLevel
        
        print("✅ 成功导入monitor模块")
        
        # 1. 训练监控器演示
        print("\n📊 1. 训练监控器演示")
        print("-" * 30)
        
        monitor = TrainingMonitor()
        print("✅ 创建训练监控器")
        
        # 使用上下文管理器
        with monitor:
            print("✅ 开始监控系统资源")
            
            # 模拟训练过程
            for epoch in range(3):
                for step in range(5):
                    # 模拟训练指标
                    loss = 1.0 - (epoch * 5 + step) * 0.05 + random.uniform(-0.1, 0.1)
                    throughput = random.uniform(50, 100)
                    
                    # 更新监控数据
                    monitor.update_training_metrics(
                        epoch=epoch,
                        step=step,
                        loss=max(0, loss),
                        learning_rate=0.001,
                        batch_size=32,
                        throughput=throughput
                    )
                    
                    print(f"  Epoch {epoch}, Step {step}: Loss={loss:.3f}, Throughput={throughput:.1f}")
                    time.sleep(0.2)  # 模拟训练时间
        
        print("✅ 监控完成")
        
        # 获取监控摘要
        summary = monitor.get_summary()
        print(f"📈 监控摘要: {summary['training_metrics_count']} 个训练指标点")
        
        # 2. 进度跟踪器演示
        print("\n⏱️  2. 进度跟踪器演示")
        print("-" * 30)
        
        progress_tracker = ProgressTracker()
        print("✅ 创建进度跟踪器")
        
        # 开始训练跟踪
        total_epochs = 3
        steps_per_epoch = 10
        progress_tracker.start_training(total_epochs, steps_per_epoch)
        print(f"✅ 开始跟踪训练: {total_epochs} epochs, {steps_per_epoch} steps/epoch")
        
        for epoch in range(total_epochs):
            progress_tracker.start_epoch(epoch)
            
            for step in range(steps_per_epoch):
                progress_tracker.start_step(step, batch_size=32)
                
                # 模拟训练时间
                time.sleep(0.05)
                
                progress_tracker.end_step()
                
                # 每3步显示进度
                if step % 3 == 0:
                    progress_bar = progress_tracker.format_progress_bar(width=25)
                    print(f"  {progress_bar}")
            
            progress_tracker.end_epoch()
        
        progress_tracker.end_training()
        print("✅ 进度跟踪完成")
        
        # 获取性能统计
        stats = progress_tracker.get_performance_stats()
        print(f"📊 性能统计:")
        print(f"  - 平均步骤时间: {stats['avg_step_time']:.3f}s")
        print(f"  - 平均吞吐量: {stats['avg_throughput']:.1f} samples/s")
        
        # 3. 告警系统演示
        print("\n🚨 3. 告警系统演示")
        print("-" * 30)
        
        alert_system = AlertSystem()
        print("✅ 创建告警系统")
        
        # 添加告警规则
        loss_rule = alert_system.create_threshold_rule(
            name="high_loss",
            threshold_value=0.8,
            operator=">",
            level=AlertLevel.WARNING,
            cooldown=5.0
        )
        alert_system.add_rule(loss_rule)
        print("✅ 添加损失值告警规则")
        
        # 模拟训练数据
        print("📊 模拟训练数据...")
        base_loss = 1.0
        for step in range(20):
            # 模拟损失值变化
            if step < 10:
                loss = base_loss - step * 0.05 + random.uniform(-0.1, 0.1)
            else:
                # 模拟异常高损失
                loss = base_loss - 0.5 + random.uniform(-0.2, 0.2)
            
            alert_system.update_data('loss', max(0, loss))
            print(f"  Step {step}: Loss={loss:.3f}")
            time.sleep(0.1)
        
        # 获取告警摘要
        alert_summary = alert_system.get_alert_summary()
        print(f"🚨 告警摘要: {alert_summary['total_alerts']} 个告警")
        
        print("\n🎉 演示完成！")
        print("=" * 50)
        
        # 4. 展示生成的文件
        print("\n📁 生成的文件:")
        monitor_logs = Path("monitor_logs")
        if monitor_logs.exists():
            for file in monitor_logs.glob("*"):
                print(f"  - {file}")
        
        alert_logs = Path("alert_logs")
        if alert_logs.exists():
            for file in alert_logs.glob("*"):
                print(f"  - {file}")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保所有依赖已安装")
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_demo()
