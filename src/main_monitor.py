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
import signal
import sys
from pathlib import Path

from src.monitor import WebMonitor, MonitorConfig, ProgressConfig, AlertConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("NeuroTrain Monitor Service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--log-dir", type=Path, default=Path("monitor_logs"))
    return parser.parse_args()


def monitor_loop(args: argparse.Namespace) -> None:
    monitor_config = MonitorConfig(log_dir=args.log_dir)
    progress_config = ProgressConfig()
    alert_config = AlertConfig(log_file=args.log_dir / "alerts.log")

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

