#!/usr/bin/env python3
"""
独立运行的监控服务器入口。

为 monitor 子模块提供 CLI 封装，可单独启动 Web 监控面板。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING


REPO_ROOT = Path(__file__).resolve().parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load .env so ALERT_EMAIL_* etc. are available
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass


if TYPE_CHECKING:  # pragma: no cover
    from src.monitor import (
        WebMonitor,
        MonitorConfig,
        ProgressConfig,
        AlertConfig,
    )


def _load_monitor_components():
    """延迟导入监控模块，避免获取帮助时加载重依赖。"""
    from src.monitor import (
        WebMonitor,
        MonitorConfig,
        ProgressConfig,
        AlertConfig,
    )

    return WebMonitor, MonitorConfig, ProgressConfig, AlertConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NeuroTrain monitor as an independent service."
    )
    parser.add_argument("--host", default="0.0.0.0", help="Web server host.")
    parser.add_argument("--port", type=int, default=5000, help="Web server port.")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode.")
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional config file created via WebMonitor.save_config().",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("runs/monitor"),
        help="Directory for monitor logs and exports.",
    )
    parser.add_argument(
        "--log-interval",
        type=float,
        default=1.0,
        help="Interval (seconds) for collecting system metrics.",
    )
    parser.add_argument(
        "--save-interval",
        type=float,
        default=60.0,
        help="Interval (seconds) for persisting metrics to disk.",
    )
    parser.add_argument(
        "--loss-threshold",
        type=float,
        default=10.0,
        help="Loss value threshold for alerts.",
    )
    parser.add_argument(
        "--memory-threshold",
        type=float,
        default=0.9,
        help="System memory usage threshold (0~1).",
    )
    parser.add_argument(
        "--gpu-memory-threshold",
        type=float,
        default=0.9,
        help="GPU memory usage threshold (0~1).",
    )
    parser.add_argument(
        "--quiet-monitor",
        action="store_true",
        help="Disable monitor module console logging.",
    )
    parser.add_argument(
        "--quiet-progress",
        action="store_true",
        help="Disable progress tracker console logging.",
    )
    parser.add_argument(
        "--quiet-alert",
        action="store_true",
        help="Disable alert system console logging.",
    )
    parser.add_argument(
        "--alert-email",
        action="store_true",
        help="Enable email alerts; requires --email-user and --email-recipients (password from ALERT_EMAIL_PASSWORD env).",
    )
    parser.add_argument("--smtp-server", default="smtp.gmail.com", help="SMTP server for email alerts.")
    parser.add_argument("--smtp-port", type=int, default=587, help="SMTP port.")
    parser.add_argument("--email-user", help="SMTP username (e.g. your@gmail.com).")
    parser.add_argument(
        "--email-password",
        default=None,
        help="SMTP password; prefer ALERT_EMAIL_PASSWORD env to avoid leaking in process list.",
    )
    parser.add_argument(
        "--email-recipients",
        help="Comma-separated recipient addresses for email alerts.",
    )
    return parser.parse_args()


def resolve_path(path_like: Path) -> Path:
    """Resolve user-provided paths relative to repo root."""
    path = path_like.expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def build_monitor(args: argparse.Namespace) -> "WebMonitor":
    (
        WebMonitor,
        MonitorConfig,
        ProgressConfig,
        AlertConfig,
    ) = _load_monitor_components()

    log_dir = resolve_path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    monitor_config = MonitorConfig(
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        loss_threshold=args.loss_threshold,
        memory_threshold=args.memory_threshold,
        gpu_memory_threshold=args.gpu_memory_threshold,
        log_dir=log_dir,
        enable_console_output=not args.quiet_monitor,
        enable_file_output=True,
    )

    progress_config = ProgressConfig(
        enable_console_output=not args.quiet_progress,
        enable_file_output=False,
    )

    alert_log_file = log_dir / "alerts.log"
    import os
    email_enabled = getattr(args, "alert_email", False)
    email_user = getattr(args, "email_user", None) or os.environ.get("ALERT_EMAIL_USER")
    email_password = getattr(args, "email_password", None) or os.environ.get("ALERT_EMAIL_PASSWORD")
    email_recipients_str = getattr(args, "email_recipients", None) or os.environ.get("ALERT_EMAIL_RECIPIENTS", "")
    email_recipients = [s.strip() for s in email_recipients_str.split(",") if s.strip()]
    alert_config = AlertConfig(
        enable_console_output=not args.quiet_alert,
        enable_file_output=True,
        log_file=alert_log_file,
        enable_email_output=email_enabled and bool(email_user and email_recipients),
        smtp_server=getattr(args, "smtp_server", "smtp.gmail.com"),
        smtp_port=getattr(args, "smtp_port", 587),
        email_username=email_user,
        email_password=email_password,
        email_recipients=email_recipients,
    )

    web_monitor = WebMonitor(
        host=args.host,
        port=args.port,
        debug=args.debug,
        monitor_config=monitor_config,
        progress_config=progress_config,
        alert_config=alert_config,
    )

    return web_monitor


def load_additional_config(web_monitor: "WebMonitor", config_path: Path) -> None:
    """Load JSON config if provided."""
    config_file = resolve_path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    web_monitor.load_config(config_file)
    print(f"Loaded configuration from {config_file}")


def main() -> None:
    args = parse_args()

    try:
        web_monitor = build_monitor(args)

        if args.config:
            load_additional_config(web_monitor, args.config)

        print(f"Starting monitor at {web_monitor.get_url()}")
        print("Press Ctrl+C to stop.")
        web_monitor.start(block=True)
    except KeyboardInterrupt:
        print("\nStopping monitor...")
    except Exception as exc:
        print(f"Failed to start monitor: {exc}")
        raise
    finally:
        if "web_monitor" in locals():
            web_monitor.stop()


if __name__ == "__main__":
    main()

