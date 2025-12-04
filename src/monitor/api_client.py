from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    import requests
except Exception:  # pragma: no cover - fallback when requests is missing
    requests = None  # type: ignore

from .progress_tracker import ProgressSnapshot


class MonitorApiClient:
    """轻量级 HTTP 客户端，用于向 Web 监控服务推送训练/进度数据。"""

    def __init__(self, base_url: str, timeout: float = 2.0, logger: Optional[logging.Logger] = None):
        if requests is None:
            raise RuntimeError("requests 未安装，无法使用 MonitorApiClient")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.logger = logger or logging.getLogger("monitor.api_client")
        self._failure_counts: Dict[str, int] = {}

    @classmethod
    def from_config(cls, monitor_conf: Dict[str, Any], logger: Optional[logging.Logger] = None) -> Optional["MonitorApiClient"]:
        base_url = monitor_conf.get("api_base") or os.environ.get("NEUROTRAIN_MONITOR_API")
        if not base_url:
            return None

        timeout = monitor_conf.get("api_timeout")
        if timeout is None:
            timeout_env = os.environ.get("NEUROTRAIN_MONITOR_TIMEOUT")
            if timeout_env is not None:
                try:
                    timeout = float(timeout_env)
                except ValueError:
                    timeout = None
        if timeout is None:
            timeout = 2.0

        try:
            return cls(base_url=base_url, timeout=timeout, logger=logger)
        except RuntimeError as exc:
            if logger:
                logger.warning("初始化 MonitorApiClient 失败：%s", exc)
            return None

    def _post(self, path: str, payload: Optional[Dict[str, Any]] = None) -> bool:
        if requests is None:
            return False

        url = f"{self.base_url}{path}"
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            self._failure_counts[path] = 0
            return True
        except Exception as exc:
            fail_count = self._failure_counts.get(path, 0)
            if fail_count < 3:
                self.logger.debug("Monitor API 请求失败 %s: %s", path, exc)
            elif fail_count == 3:
                self.logger.warning("Monitor API 在 %s 连续失败，将不再重复提示", path)
            self._failure_counts[path] = fail_count + 1
            return False

    def update_run_info(self, info: Dict[str, Any]) -> None:
        self._post("/api/run_info", info)

    def start_monitoring(self, info: Optional[Dict[str, Any]] = None) -> None:
        self._post("/api/control/start", info)

    def stop_monitoring(self) -> None:
        self._post("/api/control/stop", None)

    def send_training_metrics(
        self,
        *,
        epoch: int,
        step: int,
        loss: float,
        learning_rate: float,
        batch_size: int,
        throughput: float,
        eta_total_seconds: Optional[float] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "throughput": throughput,
        }
        if eta_total_seconds is not None:
            payload["eta_total_seconds"] = eta_total_seconds
        self._post("/api/training/update", payload)

    def send_progress_snapshot(self, snapshot: ProgressSnapshot) -> None:
        payload = {
            "timestamp": snapshot.timestamp.isoformat(),
            "epoch": snapshot.epoch,
            "step": snapshot.step,
            "total_steps": snapshot.total_steps,
            "total_epochs": snapshot.total_epochs,
            "step_time": snapshot.step_time,
            "epoch_time": snapshot.epoch_time,
            "throughput": snapshot.throughput,
            "epoch_progress": snapshot.epoch_progress,
            "total_progress": snapshot.total_progress,
            "eta_epoch_seconds": snapshot.eta_epoch.total_seconds() if snapshot.eta_epoch else None,
            "eta_total_seconds": snapshot.eta_total.total_seconds() if snapshot.eta_total else None,
        }
        self._post("/api/progress/update", payload)

