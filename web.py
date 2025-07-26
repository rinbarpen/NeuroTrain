import gradio as gr
import subprocess
import threading
import queue
import json
import time
import os
import psutil
import GPUtil
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
# 配置matplotlib支持中文字体
import matplotlib.font_manager as fm
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import numpy as np
from datetime import datetime
import torch
import logging
from typing import Optional, Tuple, List
import io
import base64
from PIL import Image

class SystemMonitor:
    """系统资源监控器"""
    
    def __init__(self):
        self.cpu_history = []
        self.memory_history = []
        self.gpu_history = []
        self.max_history = 100
        
    def get_system_info(self):
        """获取系统信息"""
        info = {
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        return info
    
    def get_current_usage(self):
        """获取当前资源使用情况"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 内存使用情况
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used = memory.used / (1024**3)  # GB
        memory_total = memory.total / (1024**3)  # GB
        
        # GPU使用情况
        gpu_info = []
        if torch.cuda.is_available():
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    gpu_info.append({
                        'id': gpu.id,
                        'name': gpu.name,
                        'load': gpu.load * 100,
                        'memory_used': gpu.memoryUsed,
                        'memory_total': gpu.memoryTotal,
                        'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                        'temperature': gpu.temperature
                    })
            except:
                # 如果GPUtil失败，使用torch获取基本信息
                for i in range(torch.cuda.device_count()):
                    memory_used = torch.cuda.memory_allocated(i) / (1024**3)
                    memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    gpu_info.append({
                        'id': i,
                        'name': torch.cuda.get_device_name(i),
                        'load': 0,  # torch无法直接获取GPU使用率
                        'memory_used': memory_used,
                        'memory_total': memory_total,
                        'memory_percent': (memory_used / memory_total) * 100,
                        'temperature': 0
                    })
        
        # 更新历史记录
        self.cpu_history.append(cpu_percent)
        self.memory_history.append(memory_percent)
        if gpu_info:
            self.gpu_history.append(gpu_info[0]['memory_percent'])
        
        # 限制历史记录长度
        if len(self.cpu_history) > self.max_history:
            self.cpu_history.pop(0)
            self.memory_history.pop(0)
            if self.gpu_history:
                self.gpu_history.pop(0)
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'memory_used': memory_used,
            'memory_total': memory_total,
            'gpu_info': gpu_info,
            'cpu_history': self.cpu_history.copy(),
            'memory_history': self.memory_history.copy(),
            'gpu_history': self.gpu_history.copy()
        }

class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self):
        self.process = None
        self.output_dir = None
        self.is_running = False
        self.mode = 'train'  # train, test, predict
        self.stdout_lines = []
        self.stderr_lines = []
        self.max_lines = 1000
        
    def start_process(self, mode: str, config_file: str, output_dir: str, extra_args: List[str] = None) -> Tuple[bool, str]:
        """启动训练/测试/预测进程"""
        if self.is_running:
            return False, f"{self.mode}进程正在运行中"
            
        self.mode = mode
        self.output_dir = Path(output_dir)
        self.stdout_lines = []
        self.stderr_lines = []
        
        # 构建命令
        if mode == 'train':
            cmd = ["python", "main.py", "-c", config_file]
        elif mode == 'test':
            cmd = ["python", "main.py", "-c", config_file, "--test"]
        elif mode == 'predict':
            cmd = ["python", "main.py", "-c", config_file, "--predict"]
        else:
            return False, f"不支持的模式: {mode}"
            
        if extra_args:
            cmd.extend(extra_args)
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.is_running = True
            
            # 启动输出监控线程
            threading.Thread(target=self._monitor_output, daemon=True).start()
            
            return True, f"{mode}进程启动成功"
        except Exception as e:
            return False, f"启动失败: {str(e)}"
    
    def stop_process(self) -> Tuple[bool, str]:
        """停止进程"""
        if self.process and self.is_running:
            self.process.terminate()
            self.is_running = False
            return True, f"{self.mode}进程已停止"
        return False, "没有运行中的进程"
    
    def _monitor_output(self):
        """监控进程输出"""
        def read_stdout():
            while self.is_running and self.process:
                try:
                    line = self.process.stdout.readline()
                    if line:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        self.stdout_lines.append(f"[{timestamp}] {line.strip()}")
                        if len(self.stdout_lines) > self.max_lines:
                            self.stdout_lines.pop(0)
                    elif self.process.poll() is not None:
                        break
                except:
                    break
        
        def read_stderr():
            while self.is_running and self.process:
                try:
                    line = self.process.stderr.readline()
                    if line:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        self.stderr_lines.append(f"[{timestamp}] {line.strip()}")
                        if len(self.stderr_lines) > self.max_lines:
                            self.stderr_lines.pop(0)
                    elif self.process.poll() is not None:
                        break
                except:
                    break
        
        # 启动读取线程
        threading.Thread(target=read_stdout, daemon=True).start()
        threading.Thread(target=read_stderr, daemon=True).start()
        
        # 等待进程结束
        if self.process:
            self.process.wait()
            self.is_running = False
    
    def get_output_logs(self) -> Tuple[str, str]:
        """获取输出日志"""
        stdout_text = "\n".join(self.stdout_lines[-100:])  # 最近100行
        stderr_text = "\n".join(self.stderr_lines[-100:])  # 最近100行
        return stdout_text, stderr_text
    
    def get_training_data(self):
        """获取训练数据"""
        if not self.output_dir or not self.output_dir.exists():
            return None, None, None
        
        # 读取损失数据
        loss_data = {}
        train_loss_file = self.output_dir / "train_loss.csv"
        valid_loss_file = self.output_dir / "valid_loss.csv"
        
        if train_loss_file.exists():
            try:
                df = pd.read_csv(train_loss_file)
                loss_data['train_loss'] = df['loss'].tolist()
            except:
                pass
        
        if valid_loss_file.exists():
            try:
                df = pd.read_csv(valid_loss_file)
                loss_data['valid_loss'] = df['loss'].tolist()
            except:
                pass
        
        # 读取指标数据
        metrics_data = {}
        mean_metric_file = self.output_dir / "mean_metric.csv"
        if mean_metric_file.exists():
            try:
                df = pd.read_csv(mean_metric_file)
                if not df.empty:
                    metrics_data = df.iloc[-1].to_dict()
            except:
                pass
        
        # 读取图片
        loss_image = None
        loss_image_file = self.output_dir / "train_epoch_loss.png"
        if loss_image_file.exists():
            try:
                loss_image = str(loss_image_file)
            except:
                pass
        
        return loss_data, metrics_data, loss_image

# 全局实例
system_monitor = SystemMonitor()
training_monitor = TrainingMonitor()

def create_system_info_display():
    """创建系统信息显示"""
    info = system_monitor.get_system_info()
    
    info_text = f"""
    🔧 **系统信息**
    - PyTorch版本: {info['torch_version']}
    - CUDA可用: {'是' if info['cuda_available'] else '否'}
    - CUDA版本: {info['cuda_version']}
    - GPU数量: {info['gpu_count']}
    """
    
    return info_text

def create_resource_monitor():
    """创建资源监控图表"""
    usage = system_monitor.get_current_usage()
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('系统资源监控', fontsize=16)
    
    # CPU使用率历史
    if usage['cpu_history']:
        ax1.plot(usage['cpu_history'], 'b-', linewidth=2)
        ax1.set_title(f'CPU使用率: {usage["cpu_percent"]:.1f}%')
        ax1.set_ylabel('使用率 (%)')
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)
    
    # 内存使用率历史
    if usage['memory_history']:
        ax2.plot(usage['memory_history'], 'g-', linewidth=2)
        ax2.set_title(f'内存使用率: {usage["memory_percent"]:.1f}% ({usage["memory_used"]:.1f}/{usage["memory_total"]:.1f} GB)')
        ax2.set_ylabel('使用率 (%)')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
    
    # GPU使用率历史
    if usage['gpu_history']:
        ax3.plot(usage['gpu_history'], 'r-', linewidth=2)
        gpu_current = usage['gpu_info'][0]['memory_percent'] if usage['gpu_info'] else 0
        ax3.set_title(f'GPU内存使用率: {gpu_current:.1f}%')
        ax3.set_ylabel('使用率 (%)')
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'GPU不可用', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('GPU监控')
    
    # GPU详细信息
    if usage['gpu_info']:
        gpu_names = []
        gpu_loads = []
        gpu_temps = []
        
        for gpu in usage['gpu_info']:
            gpu_names.append(f"GPU{gpu['id']}")
            gpu_loads.append(gpu['memory_percent'])
            gpu_temps.append(gpu['temperature'] if gpu['temperature'] > 0 else 0)
        
        x_pos = np.arange(len(gpu_names))
        bars = ax4.bar(x_pos, gpu_loads, alpha=0.7, color='orange')
        ax4.set_title('GPU内存使用详情')
        ax4.set_ylabel('内存使用率 (%)')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(gpu_names)
        ax4.set_ylim(0, 100)
        
        # 添加数值标签
        for i, (bar, load, temp) in enumerate(zip(bars, gpu_loads, gpu_temps)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{load:.1f}%\n{temp:.0f}°C' if temp > 0 else f'{load:.1f}%',
                    ha='center', va='bottom', fontsize=8)
    else:
        ax4.text(0.5, 0.5, 'GPU不可用', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('GPU详细信息')
    
    plt.tight_layout()
    
    # 保存图片到内存
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf.getvalue()

def create_loss_chart(loss_data):
    """创建损失曲线图"""
    if not loss_data:
        return None
    
    plt.figure(figsize=(10, 6))
    
    if 'train_loss' in loss_data and loss_data['train_loss']:
        epochs = range(1, len(loss_data['train_loss']) + 1)
        plt.plot(epochs, loss_data['train_loss'], 'b-', label='训练损失', linewidth=2)
    
    if 'valid_loss' in loss_data and loss_data['valid_loss']:
        epochs = range(1, len(loss_data['valid_loss']) + 1)
        plt.plot(epochs, loss_data['valid_loss'], 'r-', label='验证损失', linewidth=2)
    
    plt.title('训练损失曲线', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图片到内存
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf.getvalue()

def format_metrics_display(metrics_data):
    """格式化指标显示"""
    if not metrics_data:
        return "暂无指标数据"
    
    formatted_text = "📊 **最新训练指标**\n\n"
    
    for key, value in metrics_data.items():
        if key != 'Unnamed: 0' and isinstance(value, (int, float)):
            formatted_text += f"- **{key}**: {value:.4f}\n"
    
    return formatted_text

def start_training(config_file, output_dir, extra_args):
    """启动训练"""
    extra_list = extra_args.split() if extra_args.strip() else []
    success, message = training_monitor.start_process('train', config_file, output_dir, extra_list)
    return message

def start_testing(config_file, output_dir, extra_args):
    """启动测试"""
    extra_list = extra_args.split() if extra_args.strip() else []
    success, message = training_monitor.start_process('test', config_file, output_dir, extra_list)
    return message

def start_prediction(config_file, output_dir, extra_args):
    """启动预测"""
    extra_list = extra_args.split() if extra_args.strip() else []
    success, message = training_monitor.start_process('predict', config_file, output_dir, extra_list)
    return message

def stop_process():
    """停止进程"""
    success, message = training_monitor.stop_process()
    return message

def get_status():
    """获取状态"""
    if training_monitor.is_running:
        return f"🟢 {training_monitor.mode}进程运行中"
    else:
        return "🔴 进程已停止"

def update_logs():
    """更新日志显示"""
    stdout, stderr = training_monitor.get_output_logs()
    return stdout, stderr

def update_training_data():
    """更新训练数据显示"""
    loss_data, metrics_data, loss_image = training_monitor.get_training_data()
    
    # 创建损失图表
    loss_chart_bytes = create_loss_chart(loss_data)
    loss_chart_image = None
    if loss_chart_bytes:
        loss_chart_image = Image.open(io.BytesIO(loss_chart_bytes))
    
    # 格式化指标
    metrics_text = format_metrics_display(metrics_data)
    
    # 原始损失图片
    original_loss_image = None
    if loss_image and Path(loss_image).exists():
        original_loss_image = loss_image
    
    return loss_chart_image, metrics_text, original_loss_image

def update_system_monitor():
    """更新系统监控"""
    resource_chart_bytes = create_resource_monitor()
    resource_chart_image = Image.open(io.BytesIO(resource_chart_bytes))
    
    system_info = create_system_info_display()
    
    return resource_chart_image, system_info

# 创建Gradio界面
with gr.Blocks(title="NeuroTrain Monitor", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🧠 NeuroTrain Monitor
    
    实时监控神经网络训练、测试和预测过程
    """)
    
    with gr.Tab("控制面板"):
        with gr.Row():
            with gr.Column(scale=2):
                config_file = gr.Textbox(
                    label="配置文件路径",
                    value="configs/single/train.template.toml",
                    placeholder="输入配置文件路径"
                )
                output_dir = gr.Textbox(
                    label="输出目录",
                    value="outputs/web_training",
                    placeholder="输入输出目录路径"
                )
                extra_args = gr.Textbox(
                    label="额外参数",
                    placeholder="输入额外的命令行参数",
                    value=""
                )
            
            with gr.Column(scale=1):
                status_display = gr.Textbox(
                    label="状态",
                    value="🔴 进程已停止",
                    interactive=False
                )
                
                with gr.Row():
                    train_btn = gr.Button("🚀 开始训练", variant="primary")
                    test_btn = gr.Button("🧪 开始测试", variant="secondary")
                    predict_btn = gr.Button("🔮 开始预测", variant="secondary")
                    stop_btn = gr.Button("⏹️ 停止", variant="stop")
                
                message_display = gr.Textbox(
                    label="消息",
                    interactive=False,
                    lines=3
                )
    
    with gr.Tab("输出日志"):
        with gr.Row():
            with gr.Column():
                stdout_display = gr.Textbox(
                    label="📤 标准输出 (stdout)",
                    lines=20,
                    max_lines=20,
                    interactive=False,
                    show_copy_button=True
                )
            
            with gr.Column():
                stderr_display = gr.Textbox(
                    label="❌ 错误输出 (stderr)",
                    lines=20,
                    max_lines=20,
                    interactive=False,
                    show_copy_button=True
                )
    
    with gr.Tab("训练数据"):
        with gr.Row():
            with gr.Column():
                loss_chart_display = gr.Image(
                    label="📈 损失曲线",
                    type="pil"
                )
                
                original_loss_display = gr.Image(
                    label="📊 原始训练图表",
                    type="filepath"
                )
            
            with gr.Column():
                metrics_display = gr.Markdown(
                    label="📊 训练指标",
                    value="暂无指标数据"
                )
    
    with gr.Tab("系统监控"):
        with gr.Row():
            with gr.Column():
                system_info_display = gr.Markdown(
                    value=create_system_info_display()
                )
            
            with gr.Column():
                resource_chart_display = gr.Image(
                    label="📊 系统资源监控",
                    type="pil"
                )
    
    # 事件绑定
    train_btn.click(
        fn=start_training,
        inputs=[config_file, output_dir, extra_args],
        outputs=[message_display]
    )
    
    test_btn.click(
        fn=start_testing,
        inputs=[config_file, output_dir, extra_args],
        outputs=[message_display]
    )
    
    predict_btn.click(
        fn=start_prediction,
        inputs=[config_file, output_dir, extra_args],
        outputs=[message_display]
    )
    
    stop_btn.click(
        fn=stop_process,
        outputs=[message_display]
    )

# 定时更新函数
def auto_update():
    """自动更新函数"""
    while True:
        try:
            # 更新日志
            stdout, stderr = update_logs()
            
            # 更新状态
            status = get_status()
            
            # 更新训练数据
            loss_chart, metrics_text, original_loss = update_training_data()
            
            # 更新系统监控
            resource_chart, system_info = update_system_monitor()
            
            # 这里可以通过其他方式更新界面，比如使用全局变量或队列
            
        except Exception as e:
            print(f"Auto update error: {e}")
        
        time.sleep(2)  # 每2秒更新一次

if __name__ == "__main__":
    print("NeuroTrain Gradio Monitor starting...")
    print("系统信息:")
    print(create_system_info_display())
    
    # 启动自动更新线程
    threading.Thread(target=auto_update, daemon=True).start()
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )
