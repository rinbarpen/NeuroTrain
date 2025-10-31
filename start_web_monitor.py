#!/usr/bin/env python3
"""
启动Web监控服务器

快速启动Web监控界面，类似TensorBoard的体验。
"""

import sys
import time
import random
import threading
from pathlib import Path

# 添加项目路径
sys.path.append('/home/rczx/workspace/sxy/lab/NeuroTrain')

def main():
    """主函数"""
    print("🌐 NeuroTrain Web Monitor")
    print("=" * 50)
    
    try:
        from src.monitor import WebMonitor, MonitorConfig, ProgressConfig, AlertConfig
        
        # 创建配置
        monitor_config = MonitorConfig(
            log_interval=1.0,
            save_interval=30.0,
            enable_gpu_monitor=True,
            memory_threshold=0.85,
            loss_threshold=2.0,
            log_dir=Path("web_monitor_logs")
        )
        
        progress_config = ProgressConfig(
            window_size=100,
            min_samples=10,
            enable_console_output=True
        )
        
        alert_config = AlertConfig(
            enable_console_output=True,
            enable_file_output=True,
            log_file=Path("web_monitor_logs/alerts.log")
        )
        
        # 创建Web监控器
        web_monitor = WebMonitor(
            host="0.0.0.0",
            port=5000,
            debug=False,
            monitor_config=monitor_config,
            progress_config=progress_config,
            alert_config=alert_config
        )
        
        print("🚀 启动Web监控服务器...")
        print(f"📱 Web界面地址: {web_monitor.get_url()}")
        print("💡 在浏览器中打开上述地址查看监控界面")
        print("🎯 开始模拟训练数据...")
        print("🛑 按 Ctrl+C 停止服务器")
        
        # 启动Web服务器
        web_monitor.start(block=False)
        
        # 等待服务器启动
        time.sleep(2)
        
        # 开始模拟训练
        simulate_training_data(web_monitor)
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保安装了所有依赖: pip install flask flask-socketio")
    except KeyboardInterrupt:
        print("\n🛑 停止Web监控服务器...")
        if 'web_monitor' in locals():
            web_monitor.stop()
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


def simulate_training_data(web_monitor):
    """模拟训练数据"""
    try:
        # 开始训练跟踪
        total_epochs = 10
        steps_per_epoch = 50
        web_monitor.start_training(total_epochs, steps_per_epoch)
        
        base_loss = 2.0
        
        for epoch in range(total_epochs):
            web_monitor.start_epoch(epoch)
            
            for step in range(steps_per_epoch):
                web_monitor.start_step(step, batch_size=32)
                
                # 模拟训练指标
                loss = base_loss - (epoch * steps_per_epoch + step) * 0.01 + random.uniform(-0.1, 0.1)
                learning_rate = 0.001 * (0.95 ** epoch)
                throughput = random.uniform(80, 150)
                
                # 更新指标
                web_monitor.update_metrics(
                    epoch=epoch,
                    step=step,
                    loss=max(0, loss),
                    learning_rate=learning_rate,
                    batch_size=32,
                    throughput=throughput
                )
                
                # 每10步显示一次进度
                if step % 10 == 0:
                    progress = web_monitor.progress_tracker.get_current_progress()
                    if progress:
                        print(f"  Epoch {epoch}, Step {step}: Loss={loss:.3f}, "
                              f"Progress={progress.total_progress:.1f}%, "
                              f"ETA={progress.eta_total}")
                
                time.sleep(0.1)  # 模拟训练时间
            
            web_monitor.end_epoch()
            print(f"✅ 完成 Epoch {epoch}")
        
        web_monitor.end_training()
        print("🏁 模拟训练完成")
        
        # 保持服务器运行
        print("🌐 Web服务器继续运行，按 Ctrl+C 停止")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 停止模拟训练...")
        if web_monitor.progress_tracker.is_training:
            web_monitor.end_training()


if __name__ == "__main__":
    main()


