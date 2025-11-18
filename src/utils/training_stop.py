from __future__ import annotations

import logging
import signal
import threading
from pathlib import Path
from typing import Optional


class TrainingStopManager:
    """Singleton controller for graceful training termination."""

    _instance: "TrainingStopManager | None" = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self._event = threading.Event()
        self._reason: Optional[str] = None
        self._stop_file: Optional[Path] = None
        self._handlers_installed = False
        self._logger = logging.getLogger("train_stop")
        self._initialized = True

    def set_logger(self, logger: logging.Logger | None) -> None:
        if logger is not None:
            self._logger = logger

    def install_signal_handlers(self) -> None:
        """Install signal handlers for SIGINT/SIGTERM/SIGUSR1."""
        if self._handlers_installed:
            return

        try:
            main_thread = threading.main_thread()
        except Exception:  # pragma: no cover
            main_thread = None

        if threading.current_thread() is not main_thread:
            # Signal handlers must be installed from main thread
            self._logger.debug("Skip signal handler installation (not in main thread)")
            return

        monitored_signals = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGUSR1"):
            monitored_signals.append(signal.SIGUSR1)

        for sig in monitored_signals:
            try:
                signal.signal(sig, self._handle_signal)
            except (OSError, RuntimeError, ValueError) as exc:
                self._logger.debug("Skip signal %s: %s", sig, exc)

        self._handlers_installed = True

    def _handle_signal(self, signum, _frame) -> None:  # pragma: no cover - system signal
        try:
            sig_name = signal.Signals(signum).name
        except Exception:  # noqa: BLE001
            sig_name = str(signum)
        self._logger.warning("收到信号 %s，准备安全终止训练", sig_name)
        self.request_stop(reason=f"signal:{sig_name}")

    def register_stop_file(self, path: Path | str | None) -> None:
        """Register a file path that can be touched to request stop."""
        if path is None:
            return
        stop_path = Path(path)
        stop_path.parent.mkdir(parents=True, exist_ok=True)
        self._stop_file = stop_path

    def should_stop(self) -> bool:
        """Check whether a stop was requested."""
        if not self._event.is_set() and self._stop_file and self._stop_file.exists():
            self.request_stop(reason=f"file:{self._stop_file.name}")
        return self._event.is_set()

    def request_stop(self, *, reason: str = "manual") -> None:
        """Request a graceful stop."""
        if not self._event.is_set():
            self._reason = reason
            self._event.set()

    @property
    def reason(self) -> Optional[str]:
        return self._reason

    def acknowledge(self) -> None:
        """Mark the stop request as handled (cleanup stop file)."""
        if self._stop_file and self._stop_file.exists():
            try:
                self._stop_file.unlink()
            except OSError as exc:
                self._logger.debug("无法删除stop文件 %s: %s", self._stop_file, exc)

    def reset(self) -> None:
        """Clear stop state so that future training sessions can run."""
        self._event.clear()
        self._reason = None


