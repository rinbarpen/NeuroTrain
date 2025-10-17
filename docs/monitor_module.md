# 训练监控模块 (Monitor)

训练监控模块提供了完整的训练过程监控解决方案，包括实时系统监控、进度跟踪、性能分析和异常检测功能。

## 主要功能

### 1. 训练监控器 (TrainingMonitor)
- **系统资源监控**: CPU使用率、内存使用率、GPU使用率
- **训练指标跟踪**: 损失值、学习率、吞吐量等
- **实时数据收集**: 自动收集和存储监控数据
- **告警机制**: 支持阈值告警和异常检测

### 2. 进度跟踪器 (ProgressTracker)
- **训练进度跟踪**: 实时跟踪训练进度和ETA
- **性能分析**: 步骤时间、轮次时间、吞吐量统计
- **进度可视化**: 提供进度条和性能图表
- **数据持久化**: 支持进度数据的保存和加载

### 3. 告警系统 (AlertSystem)
- **多种告警类型**: 阈值告警、趋势告警、异常检测
- **灵活配置**: 支持自定义告警规则和冷却时间
- **多渠道通知**: 控制台、文件、邮件通知
- **告警聚合**: 防止告警风暴

### 4. 监控工具 (MonitorUtils)
- **数据可视化**: 自动生成监控图表
- **数据导出**: 支持JSON、CSV、Parquet格式
- **报告生成**: 自动生成HTML监控报告
- **性能分析**: 趋势分析和异常检测

## 快速开始

### 基础使用

```python
from src.monitor import TrainingMonitor, ProgressTracker, AlertSystem

# 创建监控组件
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
            
            # 训练代码
            loss = model.train_step(batch)
            
            # 更新监控数据
            monitor.update_training_metrics(
                epoch=epoch, step=step, loss=loss
            )
            alert_system.update_data('loss', loss)
            
            progress_tracker.end_step()
        
        progress_tracker.end_epoch()
    
    progress_tracker.end_training()
```

### 高级配置

```python
from src.monitor import (
    TrainingMonitor, MonitorConfig,
    ProgressTracker, ProgressConfig,
    AlertSystem, AlertConfig, AlertLevel
)

# 监控配置
monitor_config = MonitorConfig(
    log_interval=1.0,  # 日志记录间隔
    save_interval=60.0,  # 数据保存间隔
    enable_gpu_monitor=True,
    enable_cpu_monitor=True,
    memory_threshold=0.9,  # 内存告警阈值
    log_dir=Path("monitor_logs")
)

# 进度跟踪配置
progress_config = ProgressConfig(
    window_size=100,
    min_samples=10,
    enable_console_output=True
)

# 告警配置
alert_config = AlertConfig(
    enable_console_output=True,
    enable_email_output=True,
    email_recipients=["admin@example.com"],
    log_file=Path("alert_logs/alerts.log")
)

# 创建组件
monitor = TrainingMonitor(monitor_config)
progress_tracker = ProgressTracker(progress_config)
alert_system = AlertSystem(alert_config)
```

## 详细使用指南

### 1. 训练监控器

#### 基本配置
```python
from src.monitor import TrainingMonitor, MonitorConfig

config = MonitorConfig(
    log_interval=1.0,  # 系统指标记录间隔（秒）
    save_interval=60.0,  # 数据保存间隔（秒）
    max_history=1000,  # 最大历史记录数
    
    # 性能监控开关
    enable_gpu_monitor=True,
    enable_cpu_monitor=True,
    enable_memory_monitor=True,
    
    # 告警配置
    enable_alerts=True,
    loss_threshold=10.0,
    memory_threshold=0.9,
    gpu_memory_threshold=0.9,
    
    # 输出配置
    log_dir=Path("monitor_logs"),
    enable_console_output=True,
    enable_file_output=True
)

monitor = TrainingMonitor(config)
```

#### 使用方式
```python
# 方式1: 上下文管理器（推荐）
with monitor:
    # 训练代码
    monitor.update_training_metrics(epoch=0, step=0, loss=0.5)

# 方式2: 手动控制
monitor.start_monitoring()
try:
    # 训练代码
    monitor.update_training_metrics(epoch=0, step=0, loss=0.5)
finally:
    monitor.stop_monitoring()
```

### 2. 进度跟踪器

#### 基本使用
```python
from src.monitor import ProgressTracker

progress_tracker = ProgressTracker()

# 开始训练跟踪
progress_tracker.start_training(total_epochs=100, steps_per_epoch=1000)

for epoch in range(100):
    progress_tracker.start_epoch(epoch)
    
    for step in range(1000):
        progress_tracker.start_step(step, batch_size=32)
        
        # 训练代码
        time.sleep(0.01)  # 模拟训练时间
        
        progress_tracker.end_step()
        
        # 显示进度
        if step % 100 == 0:
            print(progress_tracker.format_progress_bar())
    
    progress_tracker.end_epoch()

progress_tracker.end_training()
```

#### 进度信息获取
```python
# 获取当前进度
current_progress = progress_tracker.get_current_progress()

# 获取进度摘要
summary = progress_tracker.get_progress_summary()
print(f"当前轮次: {summary['current_epoch']}")
print(f"总体进度: {summary['total_progress']:.1f}%")
print(f"平均吞吐量: {summary['avg_throughput']:.1f} samples/s")

# 获取性能统计
stats = progress_tracker.get_performance_stats()
print(f"平均步骤时间: {stats['avg_step_time']:.3f}s")
print(f"平均轮次时间: {stats['avg_epoch_time']:.3f}s")
```

### 3. 告警系统

#### 创建告警规则
```python
from src.monitor import AlertSystem, AlertLevel, AlertType

alert_system = AlertSystem()

# 阈值告警
loss_rule = alert_system.create_threshold_rule(
    name="high_loss",
    threshold_value=1.0,
    operator=">",
    level=AlertLevel.WARNING,
    cooldown=60.0  # 冷却时间60秒
)
alert_system.add_rule(loss_rule)

# 趋势告警
trend_rule = alert_system.create_trend_rule(
    name="loss_increasing",
    trend_window=10,
    trend_threshold=0.05,
    level=AlertLevel.ERROR
)
alert_system.add_rule(trend_rule)

# 异常检测告警
anomaly_rule = alert_system.create_anomaly_rule(
    name="loss_anomaly",
    anomaly_method="zscore",
    anomaly_threshold=3.0,
    level=AlertLevel.CRITICAL
)
alert_system.add_rule(anomaly_rule)
```

#### 使用告警系统
```python
# 更新数据触发告警检查
alert_system.update_data('loss', 0.5)
alert_system.update_data('loss', 1.2)  # 可能触发阈值告警

# 添加自定义告警回调
def custom_alert_handler(alert):
    print(f"收到告警: {alert.message}")
    # 发送到Slack、钉钉等

alert_system.add_alert_callback(custom_alert_handler)
```

### 4. 监控工具

#### 数据可视化
```python
from src.monitor.monitor_utils import (
    plot_training_metrics, plot_system_metrics,
    plot_progress_tracker, generate_monitor_report
)

# 绘制训练指标图表
plot_files = plot_training_metrics(
    monitor=monitor,
    output_dir=Path("charts"),
    metrics=['loss', 'learning_rate', 'throughput']
)

# 绘制系统指标图表
plot_files = plot_system_metrics(
    monitor=monitor,
    output_dir=Path("charts"),
    metrics=['cpu_percent', 'memory_percent', 'gpu_utilization']
)

# 绘制进度跟踪图表
plot_files = plot_progress_tracker(
    progress_tracker=progress_tracker,
    output_dir=Path("charts")
)
```

#### 数据导出
```python
from src.monitor.monitor_utils import (
    export_monitor_data, export_progress_data,
    export_alert_data
)

# 导出监控数据
data_files = export_monitor_data(
    monitor=monitor,
    output_dir=Path("exports"),
    formats=['json', 'csv', 'parquet']
)

# 导出进度数据
data_files = export_progress_data(
    progress_tracker=progress_tracker,
    output_dir=Path("exports"),
    formats=['json', 'csv']
)

# 导出告警数据
data_files = export_alert_data(
    alert_system=alert_system,
    output_dir=Path("exports"),
    formats=['json', 'csv']
)
```

#### 生成监控报告
```python
from src.monitor.monitor_utils import generate_monitor_report

# 生成完整的HTML监控报告
report_file = generate_monitor_report(
    monitor=monitor,
    progress_tracker=progress_tracker,
    alert_system=alert_system,
    output_dir=Path("reports")
)

print(f"监控报告已生成: {report_file}")
```

#### 性能分析
```python
from src.monitor.monitor_utils import (
    analyze_performance_trends, detect_performance_anomalies
)

# 分析性能趋势
trends = analyze_performance_trends(
    monitor=monitor,
    window_size=100
)

print(f"损失趋势: {trends['loss_trend']['direction']}")
print(f"吞吐量趋势: {trends['throughput_trend']['direction']}")

# 检测性能异常
anomalies = detect_performance_anomalies(
    monitor=monitor,
    method='zscore',
    threshold=3.0
)

print(f"损失异常点: {anomalies['loss']}")
print(f"吞吐量异常点: {anomalies['throughput']}")
```

## 配置选项

### MonitorConfig
- `log_interval`: 日志记录间隔（秒）
- `save_interval`: 数据保存间隔（秒）
- `max_history`: 最大历史记录数
- `enable_gpu_monitor`: 是否启用GPU监控
- `enable_cpu_monitor`: 是否启用CPU监控
- `enable_memory_monitor`: 是否启用内存监控
- `enable_alerts`: 是否启用告警
- `loss_threshold`: 损失值告警阈值
- `memory_threshold`: 内存使用率告警阈值
- `gpu_memory_threshold`: GPU内存使用率告警阈值
- `log_dir`: 日志输出目录
- `enable_console_output`: 是否启用控制台输出
- `enable_file_output`: 是否启用文件输出

### ProgressConfig
- `window_size`: 滑动窗口大小
- `min_samples`: 最小样本数用于计算ETA
- `update_interval`: 更新间隔（秒）
- `enable_console_output`: 是否启用控制台输出
- `enable_file_output`: 是否启用文件输出
- `output_file`: 输出文件路径

### AlertConfig
- `enable_console_output`: 是否启用控制台输出
- `enable_file_output`: 是否启用文件输出
- `enable_email_output`: 是否启用邮件输出
- `log_file`: 日志文件路径
- `smtp_server`: SMTP服务器地址
- `smtp_port`: SMTP端口
- `email_username`: 邮件用户名
- `email_password`: 邮件密码
- `email_recipients`: 邮件接收者列表
- `enable_aggregation`: 是否启用告警聚合
- `aggregation_window`: 聚合时间窗口（秒）
- `max_alerts_per_window`: 每个窗口最大告警数

## 最佳实践

### 1. 资源管理
- 使用上下文管理器确保监控器正确启动和停止
- 定期清理历史数据避免内存泄漏
- 合理设置监控间隔平衡性能和准确性

### 2. 告警配置
- 设置合适的告警阈值避免误报
- 使用冷却时间防止告警风暴
- 配置多渠道通知确保告警及时送达

### 3. 数据管理
- 定期导出监控数据用于后续分析
- 使用压缩格式存储大量历史数据
- 建立数据备份和恢复机制

### 4. 性能优化
- 在分布式训练中合理使用监控功能
- 避免在高频更新时进行复杂计算
- 使用异步方式处理告警通知

## 故障排除

### 常见问题

1. **监控器无法启动**
   - 检查系统权限
   - 确认依赖包已安装
   - 查看日志文件获取详细错误信息

2. **GPU监控不工作**
   - 确认CUDA环境正确安装
   - 检查GPU设备是否可用
   - 验证PyTorch CUDA支持

3. **告警不触发**
   - 检查告警规则配置
   - 确认数据更新频率
   - 验证告警阈值设置

4. **内存使用过高**
   - 减少历史数据保存量
   - 增加数据清理频率
   - 优化监控间隔

### 调试技巧

1. **启用详细日志**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **检查监控状态**
```python
summary = monitor.get_summary()
print(f"监控状态: {summary}")
```

3. **验证告警规则**
```python
alert_summary = alert_system.get_alert_summary()
print(f"告警状态: {alert_summary}")
```

## 示例代码

完整的使用示例请参考 `examples/monitor_demo.py` 文件。

## 依赖要求

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn
- psutil
- pathlib

## 许可证

本模块遵循项目主许可证。
