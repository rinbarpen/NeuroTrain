import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders
import os
import logging
from pathlib import Path
from typing import Union, List, Optional, TYPE_CHECKING
from datetime import datetime

# Import buildin annotation
from src.utils.annotation import buildin

if TYPE_CHECKING:
    from typing_extensions import Self
else:
    try:
        from typing import Self  # Python 3.11+
    except ImportError:
        from typing_extensions import Self


class EmailSender:
    def __init__(self, system_email: str, system_email_password: str, receiver_email: str):
        self.sender_email = system_email
        self.sender_password = system_email_password
        self.receiver_email = receiver_email
        self._subject = ""
        self._body = ""
        self.smtp_url = "smtp.gmail.com"
        self.smtp_port = 587
        self.smtp_server = None
        self.attachments: List[Path] = []
        
    def _connect(self):
        """建立SMTP连接"""
        try:
            if self.smtp_server is None:
                self.smtp_server = smtplib.SMTP(self.smtp_url, self.smtp_port)
                self.smtp_server.starttls()  # 启用TLS加密
                self.smtp_server.login(self.sender_email, self.sender_password)
            return True
        except Exception as e:
            logging.error(f"Failed to connect to SMTP server: {e}")
            return False
    
    def _disconnect(self):
        """断开SMTP连接"""
        if self.smtp_server:
            try:
                self.smtp_server.quit()
            except:
                pass
            self.smtp_server = None
    
    def subject(self, subject: str) -> Self:
        """设置邮件主题"""
        self._subject = subject
        return self
    
    def body(self, body: str) -> Self:
        """设置邮件正文"""
        self._body = body
        return self
    
    def attachment(self, attachment_path: Union[Path, str]) -> Self:
        """添加附件"""
        path = Path(attachment_path)
        if path.exists() and path.is_file():
            self.attachments.append(path)
            logging.debug(f"Added attachment: {path}")
        else:
            logging.warning(f"Attachment not found: {path}")
        return self
    
    def clear_attachments(self) -> Self:
        """清空附件列表"""
        self.attachments.clear()
        return self
    
    def complete_email(self, 
                                    task_name: str, 
                                    model_name: str, 
                                    total_epochs: int, 
                                    best_loss: float, 
                                    train_time: str,
                                    final_metrics: dict = None) -> Self:
        """设置训练完成邮件内容"""
        self.subject(f"🎉 Training Complete: {task_name} - {model_name}")
        
        # 构建邮件正文
        body_parts = [
            f"训练任务已成功完成！",
            f"",
            f"📋 任务信息:",
            f"  • 任务名称: {task_name}",
            f"  • 模型名称: {model_name}",
            f"  • 训练轮次: {total_epochs} epochs",
            f"  • 最佳损失: {best_loss:.6f}",
            f"  • 训练时长: {train_time}",
            f"  • 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        
        if final_metrics:
            body_parts.extend([
                f"",
                f"📊 最终性能指标:",
            ])
            for metric_name, value in final_metrics.items():
                body_parts.append(f"  • {metric_name}: {value:.4f}")
        
        body_parts.extend([
            f"",
            f"📁 输出文件已保存到相应目录，详细结果请查看附件。",
            f"",
            f"此邮件由 NeuroTrain 自动发送。"
        ])
        
        self.body("\n".join(body_parts))
        return self
    
    def failed_email(self, 
                                task_name: str, 
                                model_name: str, 
                                error_message: str,
                                current_epoch: int = None) -> Self:
        """设置训练失败邮件内容"""
        self.subject(f"❌ Training Failed: {task_name} - {model_name}")
        
        body_parts = [
            f"训练任务执行失败！",
            f"",
            f"📋 任务信息:",
            f"  • 任务名称: {task_name}",
            f"  • 模型名称: {model_name}",
        ]
        
        if current_epoch is not None:
            body_parts.append(f"  • 失败于第 {current_epoch} 轮")
            
        body_parts.extend([
            f"  • 失败时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"❌ 错误信息:",
            f"  {error_message}",
            f"",
            f"请检查日志文件以获取更详细的错误信息。",
            f"",
            f"此邮件由 NeuroTrain 自动发送。"
        ])
        
        self.body("\n".join(body_parts))
        return self

    @buildin(desc="compose training complete email with useful outputs and auto-attachments")
    def training_complete(
        self,
        output_dir: Union[str, Path],
        *,
        task_name: str,
        model_name: str,
        total_epochs: int,
        best_loss: float,
        train_time: str,
        final_metrics: Optional[dict] = None,
        include_images: bool = True,
        include_csvs: bool = True,
        include_logs: bool = True,
    ) -> "EmailSender":
        """
        设置训练完成邮件内容，并从输出目录中自动收集有用资料作为附件。
        附件包含：
        - config.toml
        - 日志文件 (*.log)
        - model_flop_count.txt
        - model_summary.txt
        - 一些总体图（*.png，默认仅收集顶层图和/或按需全量）
        - 数据CSV（如 train_loss.csv, valid_loss.csv, mean_metric.csv 等）
        """
        output_dir = Path(output_dir)
        # 设置主题与正文
        self.complete_email(
            task_name=task_name,
            model_name=model_name,
            total_epochs=total_epochs,
            best_loss=best_loss,
            train_time=train_time,
            final_metrics=final_metrics,
        )
        # 自动收集附件
        self._collect_useful_attachments(output_dir, include_images, include_csvs, include_logs)
        return self

    def _collect_useful_attachments(
        self,
        output_dir: Path,
        include_images: bool,
        include_csvs: bool,
        include_logs: bool,
    ) -> None:
        """从输出目录收集有用的资料并加入附件列表。"""
        if not output_dir.exists():
            logging.warning(f"Output directory not found: {output_dir}")
            return

        # 1) config.toml（train/test/predict目录下均可能有）
        candidate_configs = [
            output_dir / "config.toml",
            output_dir / "train" / "config.toml",
            output_dir / "test" / "config.toml",
            output_dir / "predict" / "config.toml",
        ]
        for cfg in candidate_configs:
            if cfg.exists():
                self.add_attachment(cfg)

        # 2) 日志文件（位于 output_dir 根目录，由 prepare_logger 生成）
        if include_logs:
            for p in sorted(output_dir.glob("*.log")):
                self.add_attachment(p)

        # 3) 模型信息
        for name in ["model_flop_count.txt", "model_summary.txt"]:
            p = output_dir / name
            if p.exists():
                self.add_attachment(p)

        # 4) 顶层总体图（避免把每个类别的所有图都塞进来，体积太大）
        if include_images:
            top_level_images = [
                "train_epoch_loss.png",
                "valid_epoch_loss.png",
                "epoch_metrics_curve.png",
                "epoch_metrics_curve_per_classes.png",
                "mean_metrics.png",
                "mean_metrics_per_classes.png",
            ]
            for name in top_level_images:
                p = output_dir / name
                if p.exists():
                    self.add_attachment(p)

        # 5) 常见CSV汇总
        if include_csvs:
            top_level_csvs = [
                "train_loss.csv",
                "valid_loss.csv",
                "mean_metric.csv",
                "std_metric.csv",
            ]
            for name in top_level_csvs:
                p = output_dir / name
                if p.exists():
                    self.add_attachment(p)

        # 6) 类别子目录的mean_metric.csv（体积较小，便于快速查看）
        if include_csvs:
            for class_dir in output_dir.iterdir():
                if class_dir.is_dir():
                    mean_csv = class_dir / "mean_metric.csv"
                    if mean_csv.exists():
                        self.add_attachment(mean_csv)

    def send(self, attachment_paths: Union[Path, List[Path], None] = None) -> bool:
        """发送邮件，支持多个附件。优先使用链式添加的附件，其次使用传入的附件列表"""
        if not self._connect():
            return False
            
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.receiver_email
            msg['Subject'] = self.subject

            # 添加邮件正文
            msg.attach(MIMEText(self.body, 'plain', 'utf-8'))

            # 处理附件：先用self.attachments，再附加传入的
            atts: List[Path] = list(self.attachments)
            if attachment_paths is not None:
                if isinstance(attachment_paths, Path):
                    atts.append(attachment_paths)
                else:
                    atts.extend([Path(p) for p in attachment_paths])
            for attachment_path in atts:
                if attachment_path and Path(attachment_path).exists():
                    self._attach_file(msg, Path(attachment_path))

            # 发送邮件
            self.smtp_server.sendmail(self.sender_email, self.receiver_email, msg.as_string())
            logging.info(f"Email sent successfully to {self.receiver_email}")
            return True
            
        except Exception as e:
            logging.warning(f"Error sending email: {e}")
            return False
        finally:
            self._disconnect()
    
    def _attach_file(self, msg: MIMEMultipart, file_path: Path):
        """添加附件到邮件"""
        try:
            # 根据文件类型选择合适的MIME类型
            if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                # 图片文件
                with file_path.open("rb") as f:
                    img_data = f.read()
                part = MIMEImage(img_data)
                part.add_header('Content-Disposition', f'attachment; filename="{file_path.name}"')
            else:
                # 其他文件类型
                with file_path.open("rb") as attachment:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(attachment.read())
                    encoders.encode_base64(part)
                    part.add_header("Content-Disposition", f"attachment; filename=\"{file_path.name}\"")
            
            msg.attach(part)
            logging.debug(f"Attached file: {file_path.name}")
            
        except Exception as e:
            logging.warning(f"Failed to attach file {file_path}: {e}")

    @classmethod
    def from_config(cls, config: dict):
        """从配置字典创建EmailSender实例"""
        email_config = config.get('email', {})
        if not email_config.get('enabled', False):
            return None
            
        required_keys = ['sender_email', 'sender_password', 'receiver_email']
        for key in required_keys:
            if key not in email_config:
                logging.warning(f"Email config missing required key: {key}")
                return None
        
        return cls(
            system_email=email_config['sender_email'],
            system_email_password=email_config['sender_password'],
            receiver_email=email_config['receiver_email']
        )


def send_training_completion_email(config: dict,
                                 output_dir: Union[str, Path],
                                 task_name: str,
                                 model_name: str,
                                 total_epochs: int,
                                 best_loss: float,
                                 train_time: str,
                                 final_metrics: dict = None,
                                 attachment_paths: Optional[List[Union[str, Path]]] = None):
    """发送训练完成通知邮件的便捷函数（自动收集输出目录的有用附件）"""
    email_sender = EmailSender.from_config(config)
    if email_sender is None:
        logging.debug("Email notification disabled or not configured")
        return False

    email_sender.training_complete(
        output_dir=output_dir,
        task_name=task_name,
        model_name=model_name,
        total_epochs=total_epochs,
        best_loss=best_loss,
        train_time=train_time,
        final_metrics=final_metrics,
        attachment_paths=attachment_paths,
    )
    
    return email_sender.send()


def send_failure_email(config: dict,
                               task_name: str,
                               model_name: str,
                               error_message: str,
                               current_epoch: int = None):
    """发送训练失败通知邮件的便捷函数"""
    email_sender = EmailSender.from_config(config)
    if email_sender is None:
        logging.debug("Email notification disabled or not configured")
        return False
    
    email_sender.failure_email(
        task_name=task_name,
        model_name=model_name,
        error_message=error_message,
        current_epoch=current_epoch
    )
    
    return email_sender.send()
