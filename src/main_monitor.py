"""Standalone monitor service for NeuroTrain.

This script launches the WebMonitor (default) and persists it across
training/inference sessions. Producers (trainer, predictor) can send
updates via ZeroMQ pub/sub or HTTP (future).  Currently it exposes a
simple IPC channel using multiprocessing Queue.

Usage:
    python -m src.main_monitor --host 0.0.0.0 --port 5000
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from src.monitor import WebMonitor, MonitorConfig, ProgressConfig, AlertConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("NeuroTrain Monitor Service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--log-dir", type=Path, default=Path("monitor_logs"))
    parser.add_argument("--alert-email", action="store_true", help="Enable email alerts.")
    parser.add_argument("--smtp-server", default="smtp.gmail.com")
    parser.add_argument("--smtp-port", type=int, default=587)
    parser.add_argument("--email-user", help="SMTP username.")
    parser.add_argument("--email-password", help="SMTP password; or set ALERT_EMAIL_PASSWORD.")
    parser.add_argument("--email-recipients", help="Comma-separated recipient addresses.")
    args = parser.parse_args()
    if args.alert_email:
        args.email_user = args.email_user or os.environ.get("ALERT_EMAIL_USER")
        args.email_password = args.email_password or os.environ.get("ALERT_EMAIL_PASSWORD")
        if not args.email_recipients:
            args.email_recipients = os.environ.get("ALERT_EMAIL_RECIPIENTS", "")
    return args


def monitor_loop(args: argparse.Namespace) -> None:
    monitor_config = MonitorConfig(log_dir=args.log_dir)
    progress_config = ProgressConfig()
    recipients = [s.strip() for s in (args.email_recipients or "").split(",") if s.strip()]
    alert_config = AlertConfig(
        log_file=args.log_dir / "alerts.log",
        enable_email_output=bool(args.alert_email and args.email_user and recipients),
        smtp_server=args.smtp_server,
        smtp_port=args.smtp_port,
        email_username=args.email_user,
        email_password=args.email_password,
        email_recipients=recipients,
    )

    web_monitor = WebMonitor(
        host=args.host,
        port=args.port,
        monitor_config=monitor_config,
        progress_config=progress_config,
        alert_config=alert_config,
    )

    def handle_signal(_sig, _frame):
        web_monitor.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    web_monitor.start(block=True)


def main() -> None:
    args = parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    monitor_loop(args)


if __name__ == "__main__":
    main()

