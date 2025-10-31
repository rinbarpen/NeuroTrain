"""
告警系统 - 训练过程中的异常检测和告警功能

提供多种告警机制，包括阈值告警、趋势告警、异常检测等。
"""

import time
import logging
import smtplib
import json
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from src.recorder.meter import Meter
from src.utils.ndict import NDict


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """告警类型"""
    THRESHOLD = "threshold"
    TREND = "trend"
    ANOMALY = "anomaly"
    PERFORMANCE = "performance"
    SYSTEM = "system"


@dataclass
class AlertRule:
    """告警规则"""
    name: str
    alert_type: AlertType
    level: AlertLevel
    enabled: bool = True
    
    # 阈值告警参数
    threshold_value: Optional[float] = None
    threshold_operator: str = ">"  # >, <, >=, <=, ==, !=
    
    # 趋势告警参数
    trend_window: int = 10
    trend_threshold: float = 0.1  # 变化率阈值
    
    # 异常检测参数
    anomaly_method: str = "zscore"  # zscore, iqr, isolation_forest
    anomaly_threshold: float = 3.0
    
    # 冷却时间（秒）
    cooldown: float = 60.0
    
    # 告警消息模板
    message_template: str = "{name}: {value} {operator} {threshold}"
    
    # 最后触发时间
    last_triggered: Optional[datetime] = None


@dataclass
class Alert:
    """告警信息"""
    rule_name: str
    alert_type: AlertType
    level: AlertLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    value: Optional[float] = None
    threshold: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertConfig:
    """告警配置"""
    # 基础配置
    enable_console_output: bool = True
    enable_file_output: bool = True
    enable_email_output: bool = False
    log_file: Optional[Path] = None
    
    # 邮件配置
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    email_username: Optional[str] = None
    email_password: Optional[str] = None
    email_recipients: List[str] = field(default_factory=list)
    
    # 告警聚合
    enable_aggregation: bool = True
    aggregation_window: float = 300.0  # 5分钟
    max_alerts_per_window: int = 10


class AlertSystem:
    """告警系统"""
    
    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or AlertConfig()
        self.logger = self._setup_logger()
        
        # 告警规则
        self.rules: Dict[str, AlertRule] = {}
        
        # 告警历史
        self.alert_history: List[Alert] = []
        
        # 告警回调
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # 数据历史（用于趋势和异常检测）
        self.data_history: Dict[str, List[float]] = {}
        
        # 告警聚合
        self.alert_counts: Dict[str, int] = {}
        self.last_aggregation_reset = time.time()
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志器"""
        logger = logging.getLogger(f"AlertSystem_{id(self)}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # 控制台输出
            if self.config.enable_console_output:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)
            
            # 文件输出
            if self.config.enable_file_output and self.config.log_file:
                self.config.log_file.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(self.config.log_file)
                file_handler.setLevel(logging.INFO)
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
        
        return logger
    
    def add_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.rules[rule.name] = rule
        self.logger.info(f"添加告警规则: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """移除告警规则"""
        if rule_name in self.rules:
            del self.rules[rule_name]
            self.logger.info(f"移除告警规则: {rule_name}")
    
    def update_data(self, metric_name: str, value: float):
        """更新指标数据"""
        if metric_name not in self.data_history:
            self.data_history[metric_name] = []
        
        self.data_history[metric_name].append(value)
        
        # 限制历史数据长度
        if len(self.data_history[metric_name]) > 1000:
            self.data_history[metric_name] = self.data_history[metric_name][-1000:]
        
        # 检查告警
        self._check_alerts(metric_name, value)
    
    def _check_alerts(self, metric_name: str, value: float):
        """检查告警条件"""
        current_time = time.time()
        
        # 重置聚合计数
        if current_time - self.last_aggregation_reset > self.config.aggregation_window:
            self.alert_counts.clear()
            self.last_aggregation_reset = current_time
        
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            # 检查冷却时间
            if rule.last_triggered:
                time_since_last = (datetime.now() - rule.last_triggered).total_seconds()
                if time_since_last < rule.cooldown:
                    continue
            
            # 检查聚合限制
            if self.config.enable_aggregation:
                alert_count = self.alert_counts.get(rule_name, 0)
                if alert_count >= self.config.max_alerts_per_window:
                    continue
            
            # 根据告警类型检查
            should_alert = False
            alert_value = value
            alert_threshold = None
            
            if rule.alert_type == AlertType.THRESHOLD:
                should_alert, alert_threshold = self._check_threshold_alert(rule, value)
            elif rule.alert_type == AlertType.TREND:
                should_alert = self._check_trend_alert(rule, metric_name)
            elif rule.alert_type == AlertType.ANOMALY:
                should_alert = self._check_anomaly_alert(rule, metric_name, value)
            
            if should_alert:
                self._trigger_alert(rule, alert_value, alert_threshold, metric_name)
    
    def _check_threshold_alert(self, rule: AlertRule, value: float) -> Tuple[bool, Optional[float]]:
        """检查阈值告警"""
        if rule.threshold_value is None:
            return False, None
        
        threshold = rule.threshold_value
        operator = rule.threshold_operator
        
        if operator == ">":
            return value > threshold, threshold
        elif operator == "<":
            return value < threshold, threshold
        elif operator == ">=":
            return value >= threshold, threshold
        elif operator == "<=":
            return value <= threshold, threshold
        elif operator == "==":
            return abs(value - threshold) < 1e-6, threshold
        elif operator == "!=":
            return abs(value - threshold) >= 1e-6, threshold
        else:
            return False, None
    
    def _check_trend_alert(self, rule: AlertRule, metric_name: str) -> bool:
        """检查趋势告警"""
        if metric_name not in self.data_history:
            return False
        
        data = self.data_history[metric_name]
        if len(data) < rule.trend_window:
            return False
        
        # 计算趋势
        recent_data = data[-rule.trend_window:]
        trend = np.polyfit(range(len(recent_data)), recent_data, 1)[0]
        
        # 检查趋势是否超过阈值
        return abs(trend) > rule.trend_threshold
    
    def _check_anomaly_alert(self, rule: AlertRule, metric_name: str, value: float) -> bool:
        """检查异常告警"""
        if metric_name not in self.data_history:
            return False
        
        data = self.data_history[metric_name]
        if len(data) < 10:  # 需要足够的历史数据
            return False
        
        if rule.anomaly_method == "zscore":
            return self._zscore_anomaly_detection(data, value, rule.anomaly_threshold)
        elif rule.anomaly_method == "iqr":
            return self._iqr_anomaly_detection(data, value)
        else:
            return False
    
    def _zscore_anomaly_detection(self, data: List[float], value: float, threshold: float) -> bool:
        """Z-score异常检测"""
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return False
        
        zscore = abs(value - mean) / std
        return bool(zscore > threshold)
    
    def _iqr_anomaly_detection(self, data: List[float], value: float) -> bool:
        """IQR异常检测"""
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        return bool(value < lower_bound or value > upper_bound)
        
    def _trigger_alert(self, rule: AlertRule, value: float, threshold: Optional[float], metric_name: str):
        """触发告警"""
        # 更新规则状态
        rule.last_triggered = datetime.now()
        
        # 更新聚合计数
        if self.config.enable_aggregation:
            self.alert_counts[rule.name] = self.alert_counts.get(rule.name, 0) + 1
        
        # 创建告警
        message = rule.message_template.format(
            name=rule.name,
            value=value,
            operator=rule.threshold_operator,
            threshold=threshold or "N/A",
            metric=metric_name
        )
        
        alert = Alert(
            rule_name=rule.name,
            alert_type=rule.alert_type,
            level=rule.level,
            message=message,
            value=value,
            threshold=threshold,
            context={'metric_name': metric_name}
        )
        
        # 添加到历史记录
        self.alert_history.append(alert)
        
        # 限制历史记录长度
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        
        # 输出告警
        self._output_alert(alert)
        
        # 调用回调函数
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"告警回调出错: {e}")
    
    def _output_alert(self, alert: Alert):
        """输出告警"""
        # 控制台输出
        if self.config.enable_console_output:
            level_name = alert.level.value.upper()
            self.logger.warning(f"[{level_name}] {alert.message}")
        
        # 文件输出
        if self.config.enable_file_output and self.config.log_file:
            alert_data = {
                'timestamp': alert.timestamp.isoformat(),
                'rule_name': alert.rule_name,
                'alert_type': alert.alert_type.value,
                'level': alert.level.value,
                'message': alert.message,
                'value': alert.value,
                'threshold': alert.threshold,
                'context': alert.context
            }
            
            with open(self.config.log_file, 'a') as f:
                f.write(json.dumps(alert_data) + '\n')
        
        # 邮件输出
        if self.config.enable_email_output and self.config.email_recipients:
            self._send_email_alert(alert)
    
    def _send_email_alert(self, alert: Alert):
        """发送邮件告警"""
        if not self.config.email_username or not self.config.email_password:
            self.logger.warning("邮件配置不完整，跳过邮件发送")
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.email_username
            msg['To'] = ', '.join(self.config.email_recipients)
            msg['Subject'] = f"训练告警: {alert.rule_name}"
            
            body = f"""
告警信息:
- 规则名称: {alert.rule_name}
- 告警级别: {alert.level.value}
- 告警类型: {alert.alert_type.value}
- 消息: {alert.message}
- 时间: {alert.timestamp.isoformat()}
- 值: {alert.value}
- 阈值: {alert.threshold}
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            server.starttls()
            server.login(self.config.email_username, self.config.email_password)
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            self.logger.error(f"发送邮件告警失败: {e}")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """添加告警回调函数"""
        self.alert_callbacks.append(callback)
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """获取告警摘要"""
        # 按级别统计
        level_counts = {}
        for alert in self.alert_history:
            level = alert.level.value
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # 按规则统计
        rule_counts = {}
        for alert in self.alert_history:
            rule_name = alert.rule_name
            rule_counts[rule_name] = rule_counts.get(rule_name, 0) + 1
        
        # 最近告警
        recent_alerts = self.alert_history[-10:] if self.alert_history else []
        
        return {
            'total_alerts': len(self.alert_history),
            'level_counts': level_counts,
            'rule_counts': rule_counts,
            'recent_alerts': [
                {
                    'timestamp': alert.timestamp.isoformat(),
                    'rule_name': alert.rule_name,
                    'level': alert.level.value,
                    'message': alert.message
                }
                for alert in recent_alerts
            ],
            'active_rules': len([r for r in self.rules.values() if r.enabled])
        }
    
    def create_threshold_rule(self, 
                            name: str,
                            threshold_value: float,
                            operator: str = ">",
                            level: AlertLevel = AlertLevel.WARNING,
                            cooldown: float = 60.0) -> AlertRule:
        """创建阈值告警规则"""
        rule = AlertRule(
            name=name,
            alert_type=AlertType.THRESHOLD,
            level=level,
            threshold_value=threshold_value,
            threshold_operator=operator,
            cooldown=cooldown
        )
        return rule
    
    def create_trend_rule(self,
                         name: str,
                         trend_window: int = 10,
                         trend_threshold: float = 0.1,
                         level: AlertLevel = AlertLevel.WARNING,
                         cooldown: float = 60.0) -> AlertRule:
        """创建趋势告警规则"""
        rule = AlertRule(
            name=name,
            alert_type=AlertType.TREND,
            level=level,
            trend_window=trend_window,
            trend_threshold=trend_threshold,
            cooldown=cooldown
        )
        return rule
    
    def create_anomaly_rule(self,
                           name: str,
                           anomaly_method: str = "zscore",
                           anomaly_threshold: float = 3.0,
                           level: AlertLevel = AlertLevel.WARNING,
                           cooldown: float = 60.0) -> AlertRule:
        """创建异常检测告警规则"""
        rule = AlertRule(
            name=name,
            alert_type=AlertType.ANOMALY,
            level=level,
            anomaly_method=anomaly_method,
            anomaly_threshold=anomaly_threshold,
            cooldown=cooldown
        )
        return rule
    
    def save_rules(self, filepath: Path):
        """保存告警规则"""
        rules_data = {}
        for name, rule in self.rules.items():
            rules_data[name] = {
                'name': rule.name,
                'alert_type': rule.alert_type.value,
                'level': rule.level.value,
                'enabled': rule.enabled,
                'threshold_value': rule.threshold_value,
                'threshold_operator': rule.threshold_operator,
                'trend_window': rule.trend_window,
                'trend_threshold': rule.trend_threshold,
                'anomaly_method': rule.anomaly_method,
                'anomaly_threshold': rule.anomaly_threshold,
                'cooldown': rule.cooldown,
                'message_template': rule.message_template
            }
        
        with open(filepath, 'w') as f:
            json.dump(rules_data, f, indent=2)
    
    def load_rules(self, filepath: Path):
        """加载告警规则"""
        with open(filepath, 'r') as f:
            rules_data = json.load(f)
        
        self.rules.clear()
        for name, data in rules_data.items():
            rule = AlertRule(
                name=data['name'],
                alert_type=AlertType(data['alert_type']),
                level=AlertLevel(data['level']),
                enabled=data['enabled'],
                threshold_value=data['threshold_value'],
                threshold_operator=data['threshold_operator'],
                trend_window=data['trend_window'],
                trend_threshold=data['trend_threshold'],
                anomaly_method=data['anomaly_method'],
                anomaly_threshold=data['anomaly_threshold'],
                cooldown=data['cooldown'],
                message_template=data['message_template']
            )
            self.rules[name] = rule
    
    def reset(self):
        """重置告警系统"""
        self.alert_history.clear()
        self.data_history.clear()
        self.alert_counts.clear()
        self.last_aggregation_reset = time.time()
        
        # 重置规则状态
        for rule in self.rules.values():
            rule.last_triggered = None
