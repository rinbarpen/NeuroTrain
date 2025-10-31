"""
Web监控服务器 - 提供类似TensorBoard的Web界面

支持实时查看训练状态、图表、报告和告警信息。
"""

import json
import time
import threading
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
import asyncio
from dataclasses import asdict

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
import numpy as np

from .training_monitor import TrainingMonitor, SystemMetrics, TrainingMetrics
from .progress_tracker import ProgressTracker, ProgressSnapshot
from .alert_system import AlertSystem, Alert
from .monitor_utils import (
    plot_training_metrics, plot_system_metrics, plot_progress_tracker,
    analyze_performance_trends, detect_performance_anomalies
)


class WebMonitorServer:
    """Web监控服务器"""
    
    def __init__(self, 
                 monitor: Optional[TrainingMonitor] = None,
                 progress_tracker: Optional[ProgressTracker] = None,
                 alert_system: Optional[AlertSystem] = None,
                 host: str = "0.0.0.0",
                 port: int = 5000,
                 debug: bool = False):
        
        self.monitor = monitor
        self.progress_tracker = progress_tracker
        self.alert_system = alert_system
        
        # Flask应用配置
        self.app = Flask(__name__, 
                        template_folder='templates',
                        static_folder='static')
        self.app.config['SECRET_KEY'] = 'monitor_secret_key'
        
        # SocketIO配置
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # 服务器配置
        self.host = host
        self.port = port
        self.debug = debug
        
        # 数据更新线程
        self.update_thread = None
        self.running = False
        
        # 设置路由
        self._setup_routes()
        self._setup_socket_events()
    
    def _setup_routes(self):
        """设置路由"""
        
        @self.app.route('/')
        def index():
            """主页"""
            return render_template('index.html')
        
        @self.app.route('/api/status')
        def get_status():
            """获取监控状态"""
            status = {
                'monitor_active': self.monitor.is_monitoring if self.monitor else False,
                'progress_active': self.progress_tracker.is_training if self.progress_tracker else False,
                'alert_active': len(self.alert_system.rules) > 0 if self.alert_system else False,
                'timestamp': datetime.now().isoformat()
            }
            return jsonify(status)
        
        @self.app.route('/api/training_metrics')
        def get_training_metrics():
            """获取训练指标"""
            if not self.monitor or not self.monitor.training_metrics_history:
                return jsonify([])
            
            metrics = []
            for m in self.monitor.training_metrics_history[-100:]:  # 最近100个点
                metrics.append({
                    'timestamp': m.timestamp.isoformat(),
                    'epoch': m.epoch,
                    'step': m.step,
                    'loss': m.loss,
                    'learning_rate': m.learning_rate,
                    'throughput': m.throughput,
                    'batch_size': m.batch_size
                })
            
            return jsonify(metrics)
        
        @self.app.route('/api/system_metrics')
        def get_system_metrics():
            """获取系统指标"""
            if not self.monitor or not self.monitor.system_metrics_history:
                return jsonify([])
            
            metrics = []
            for m in self.monitor.system_metrics_history[-100:]:  # 最近100个点
                metrics.append({
                    'timestamp': m.timestamp.isoformat(),
                    'cpu_percent': m.cpu_percent,
                    'memory_percent': m.memory_percent,
                    'memory_used_gb': m.memory_used_gb,
                    'gpu_utilization': m.gpu_utilization,
                    'gpu_memory_used_gb': m.gpu_memory_used_gb
                })
            
            return jsonify(metrics)
        
        @self.app.route('/api/progress')
        def get_progress():
            """获取进度信息"""
            if not self.progress_tracker:
                return jsonify({})
            
            current = self.progress_tracker.get_current_progress()
            if not current:
                return jsonify({})
            
            return jsonify({
                'current_epoch': current.epoch,
                'current_step': current.step,
                'total_epochs': current.total_epochs,
                'total_steps': current.total_steps,
                'epoch_progress': current.epoch_progress,
                'total_progress': current.total_progress,
                'eta_epoch': str(current.eta_epoch) if current.eta_epoch else None,
                'eta_total': str(current.eta_total) if current.eta_total else None,
                'throughput': current.throughput
            })
        
        @self.app.route('/api/progress_summary')
        def get_progress_summary():
            """获取进度摘要"""
            if not self.progress_tracker:
                return jsonify({})
            
            return jsonify(self.progress_tracker.get_progress_summary())
        
        @self.app.route('/api/alerts')
        def get_alerts():
            """获取告警信息"""
            if not self.alert_system:
                return jsonify([])
            
            alerts = []
            for alert in self.alert_system.alert_history[-50:]:  # 最近50个告警
                alerts.append({
                    'timestamp': alert.timestamp.isoformat(),
                    'rule_name': alert.rule_name,
                    'level': alert.level.value,
                    'message': alert.message,
                    'value': alert.value,
                    'threshold': alert.threshold
                })
            
            return jsonify(alerts)
        
        @self.app.route('/api/alert_summary')
        def get_alert_summary():
            """获取告警摘要"""
            if not self.alert_system:
                return jsonify({})
            
            return jsonify(self.alert_system.get_alert_summary())
        
        @self.app.route('/api/performance_stats')
        def get_performance_stats():
            """获取性能统计"""
            if not self.progress_tracker:
                return jsonify({})
            
            return jsonify(self.progress_tracker.get_performance_stats())
        
        @self.app.route('/api/trends')
        def get_trends():
            """获取性能趋势"""
            if not self.monitor:
                return jsonify({})
            
            trends = analyze_performance_trends(self.monitor)
            return jsonify(trends)
        
        @self.app.route('/api/anomalies')
        def get_anomalies():
            """获取异常检测结果"""
            if not self.monitor:
                return jsonify({})
            
            anomalies = detect_performance_anomalies(self.monitor)
            return jsonify(anomalies)
        
        @self.app.route('/api/charts/training')
        def get_training_chart():
            """获取训练指标图表"""
            if not self.monitor:
                return jsonify({'error': 'Monitor not available'})
            
            try:
                chart_dir = Path("web_charts")
                chart_dir.mkdir(exist_ok=True)
                
                plot_files = plot_training_metrics(
                    self.monitor, 
                    chart_dir,
                    metrics=['loss', 'learning_rate', 'throughput']
                )
                
                return jsonify({
                    'success': True,
                    'files': [str(f) for f in plot_files]
                })
            except Exception as e:
                return jsonify({'error': str(e)})
        
        @self.app.route('/api/charts/system')
        def get_system_chart():
            """获取系统指标图表"""
            if not self.monitor:
                return jsonify({'error': 'Monitor not available'})
            
            try:
                chart_dir = Path("web_charts")
                chart_dir.mkdir(exist_ok=True)
                
                plot_files = plot_system_metrics(
                    self.monitor, 
                    chart_dir,
                    metrics=['cpu_percent', 'memory_percent', 'gpu_utilization']
                )
                
                return jsonify({
                    'success': True,
                    'files': [str(f) for f in plot_files]
                })
            except Exception as e:
                return jsonify({'error': str(e)})
        
        @self.app.route('/api/charts/progress')
        def get_progress_chart():
            """获取进度跟踪图表"""
            if not self.progress_tracker:
                return jsonify({'error': 'Progress tracker not available'})
            
            try:
                chart_dir = Path("web_charts")
                chart_dir.mkdir(exist_ok=True)
                
                plot_files = plot_progress_tracker(
                    self.progress_tracker, 
                    chart_dir
                )
                
                return jsonify({
                    'success': True,
                    'files': [str(f) for f in plot_files]
                })
            except Exception as e:
                return jsonify({'error': str(e)})
        
        @self.app.route('/api/export/<format>')
        def export_data(format):
            """导出数据"""
            if format not in ['json', 'csv', 'parquet']:
                return jsonify({'error': 'Unsupported format'})
            
            try:
                from .monitor_utils import export_monitor_data, export_progress_data, export_alert_data
                
                export_dir = Path("web_exports")
                export_dir.mkdir(exist_ok=True)
                
                files = []
                if self.monitor:
                    files.extend(export_monitor_data(self.monitor, export_dir, [format]))
                if self.progress_tracker:
                    files.extend(export_progress_data(self.progress_tracker, export_dir, [format]))
                if self.alert_system:
                    files.extend(export_alert_data(self.alert_system, export_dir, [format]))
                
                return jsonify({
                    'success': True,
                    'files': [str(f) for f in files]
                })
            except Exception as e:
                return jsonify({'error': str(e)})
        
        @self.app.route('/api/control/start')
        def start_monitoring():
            """启动监控"""
            if self.monitor:
                self.monitor.start_monitoring()
            return jsonify({'success': True})
        
        @self.app.route('/api/control/stop')
        def stop_monitoring():
            """停止监控"""
            if self.monitor:
                self.monitor.stop_monitoring()
            return jsonify({'success': True})
        
        @self.app.route('/api/control/reset')
        def reset_monitoring():
            """重置监控"""
            if self.monitor:
                self.monitor.reset()
            if self.progress_tracker:
                self.progress_tracker.reset()
            if self.alert_system:
                self.alert_system.reset()
            return jsonify({'success': True})
    
    def _setup_socket_events(self):
        """设置SocketIO事件"""
        
        @self.socketio.on('connect')
        def handle_connect():
            print('Client connected')
            emit('status', {'message': 'Connected to monitor server'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print('Client disconnected')
        
        @self.socketio.on('request_update')
        def handle_update_request():
            """处理更新请求"""
            self._emit_latest_data()
    
    def _emit_latest_data(self):
        """发送最新数据"""
        try:
            # 发送训练指标
            if self.monitor and self.monitor.training_metrics_history:
                latest_training = self.monitor.training_metrics_history[-1]
                self.socketio.emit('training_update', {
                    'timestamp': latest_training.timestamp.isoformat(),
                    'epoch': latest_training.epoch,
                    'step': latest_training.step,
                    'loss': latest_training.loss,
                    'learning_rate': latest_training.learning_rate,
                    'throughput': latest_training.throughput
                })
            
            # 发送系统指标
            if self.monitor and self.monitor.system_metrics_history:
                latest_system = self.monitor.system_metrics_history[-1]
                self.socketio.emit('system_update', {
                    'timestamp': latest_system.timestamp.isoformat(),
                    'cpu_percent': latest_system.cpu_percent,
                    'memory_percent': latest_system.memory_percent,
                    'gpu_utilization': latest_system.gpu_utilization
                })
            
            # 发送进度更新
            if self.progress_tracker:
                current = self.progress_tracker.get_current_progress()
                if current:
                    self.socketio.emit('progress_update', {
                        'current_epoch': current.epoch,
                        'current_step': current.step,
                        'total_progress': current.total_progress,
                        'eta_total': str(current.eta_total) if current.eta_total else None,
                        'throughput': current.throughput
                    })
            
            # 发送告警更新
            if self.alert_system and self.alert_system.alert_history:
                latest_alert = self.alert_system.alert_history[-1]
                self.socketio.emit('alert_update', {
                    'timestamp': latest_alert.timestamp.isoformat(),
                    'rule_name': latest_alert.rule_name,
                    'level': latest_alert.level.value,
                    'message': latest_alert.message
                })
                
        except Exception as e:
            print(f"Error emitting data: {e}")
    
    def _update_loop(self):
        """数据更新循环"""
        while self.running:
            try:
                self._emit_latest_data()
                time.sleep(1.0)  # 每秒更新一次
            except Exception as e:
                print(f"Update loop error: {e}")
                time.sleep(5.0)
    
    def start(self):
        """启动Web服务器"""
        print(f"🚀 Starting Web Monitor Server at http://{self.host}:{self.port}")
        
        # 启动数据更新线程
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        # 启动Flask-SocketIO服务器
        self.socketio.run(self.app, 
                         host=self.host, 
                         port=self.port, 
                         debug=self.debug,
                         allow_unsafe_werkzeug=True)
    
    def stop(self):
        """停止Web服务器"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5.0)
    
    def set_monitor(self, monitor: TrainingMonitor):
        """设置监控器"""
        self.monitor = monitor
    
    def set_progress_tracker(self, progress_tracker: ProgressTracker):
        """设置进度跟踪器"""
        self.progress_tracker = progress_tracker
    
    def set_alert_system(self, alert_system: AlertSystem):
        """设置告警系统"""
        self.alert_system = alert_system


def create_web_monitor(host: str = "0.0.0.0", 
                      port: int = 5000,
                      debug: bool = False) -> WebMonitorServer:
    """创建Web监控服务器"""
    return WebMonitorServer(host=host, port=port, debug=debug)
