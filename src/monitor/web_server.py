"""
使用 FastAPI + Socket.IO 的 Web 监控服务器
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import socketio
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .alert_system import AlertSystem
from .monitor_utils import (
    analyze_performance_trends,
    detect_performance_anomalies,
    plot_progress_tracker,
    plot_system_metrics,
    plot_training_metrics,
)
from .progress_tracker import ProgressTracker
from .training_monitor import TrainingMonitor


class WebMonitorServer:
    """FastAPI 实现的 Web 监控服务器"""

    def __init__(
        self,
        monitor: Optional[TrainingMonitor] = None,
        progress_tracker: Optional[ProgressTracker] = None,
        alert_system: Optional[AlertSystem] = None,
        host: str = "0.0.0.0",
        port: int = 5000,
        debug: bool = False,
    ):
        self.monitor = monitor
        self.progress_tracker = progress_tracker
        self.alert_system = alert_system

        self.host = host
        self.port = port
        self.debug = debug

        self.running = False
        self.update_task: Optional[asyncio.Task] = None
        self.server: Optional[uvicorn.Server] = None
        self.current_run_info: Dict[str, Any] = {}

        templates_dir = Path(__file__).parent / "templates"
        static_dir = Path(__file__).parent / "static"

        self.app = FastAPI(title="NeuroTrain Monitor")
        self.templates = Jinja2Templates(directory=str(templates_dir))
        self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        self.socketio = socketio.AsyncServer(
            async_mode="asgi",
            cors_allowed_origins="*",
        )
        self.asgi_app = socketio.ASGIApp(self.socketio, other_asgi_app=self.app)

        self._setup_routes()
        self._setup_socket_events()
        self._setup_lifecycle_hooks()

    def _setup_lifecycle_hooks(self) -> None:
        @self.app.on_event("startup")
        async def startup_event():
            self.running = True
            self.update_task = asyncio.create_task(self._update_loop())

        @self.app.on_event("shutdown")
        async def shutdown_event():
            self.running = False
            if self.update_task:
                self.update_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self.update_task

    def _setup_routes(self) -> None:
        @self.app.get("/", response_class=HTMLResponse)
        async def index(request: Request):
            return self.templates.TemplateResponse(
                "index.html",
                {"request": request},
            )

        async def _update_run_info_from_request(request: Request) -> Optional[Dict[str, Any]]:
            if request.headers.get("content-length") in (None, "0"):
                return None
            try:
                data = await request.json()
            except Exception:
                return None
            if isinstance(data, dict):
                data.setdefault("notified_at", datetime.now().isoformat())
                self.current_run_info = data
                return data
            return None

        @self.app.get("/api/status")
        async def get_status():
            return {
                "monitor_active": self.monitor.is_monitoring if self.monitor else False,
                "progress_active": self.progress_tracker.is_training
                if self.progress_tracker
                else False,
                "alert_active": len(self.alert_system.rules) > 0
                if self.alert_system
                else False,
                "timestamp": datetime.now().isoformat(),
                "current_run": self.current_run_info,
            }

        @self.app.get("/api/run_info")
        async def get_run_info():
            return self.current_run_info

        @self.app.post("/api/run_info")
        async def set_run_info(request: Request):
            await _update_run_info_from_request(request)
            return {"success": True, "run_info": self.current_run_info}

        @self.app.get("/api/training_metrics")
        async def get_training_metrics():
            if not self.monitor or not self.monitor.training_metrics_history:
                return []

            metrics: List[Dict[str, Any]] = []
            for m in self.monitor.training_metrics_history[-100:]:
                metrics.append(
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "epoch": m.epoch,
                        "step": m.step,
                        "loss": m.loss,
                        "learning_rate": m.learning_rate,
                        "throughput": m.throughput,
                        "batch_size": m.batch_size,
                    }
                )
            return metrics

        @self.app.get("/api/system_metrics")
        async def get_system_metrics():
            if not self.monitor or not self.monitor.system_metrics_history:
                return []

            metrics: List[Dict[str, Any]] = []
            for m in self.monitor.system_metrics_history[-100:]:
                metrics.append(
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "cpu_percent": m.cpu_percent,
                        "memory_percent": m.memory_percent,
                        "memory_used_gb": m.memory_used_gb,
                        "gpu_utilization": m.gpu_utilization,
                        "gpu_memory_used_gb": m.gpu_memory_used_gb,
                    }
                )
            return metrics

        @self.app.get("/api/progress")
        async def get_progress():
            if not self.progress_tracker:
                return {}

            current = self.progress_tracker.get_current_progress()
            if not current:
                return {}

            return {
                "current_epoch": current.epoch,
                "current_step": current.step,
                "total_epochs": current.total_epochs,
                "total_steps": current.total_steps,
                "epoch_progress": current.epoch_progress,
                "total_progress": current.total_progress,
                "eta_epoch": str(current.eta_epoch) if current.eta_epoch else None,
                "eta_total": str(current.eta_total) if current.eta_total else None,
                "throughput": current.throughput,
            }

        @self.app.get("/api/progress_summary")
        async def get_progress_summary():
            if not self.progress_tracker:
                return {}
            return self.progress_tracker.get_progress_summary()

        @self.app.post("/api/training/update")
        async def push_training_metrics(request: Request):
            if not self.monitor:
                raise HTTPException(status_code=400, detail="Monitor not available")
            data = await request.json()
            try:
                eta_seconds = data.get("eta_total_seconds")
                eta = timedelta(seconds=float(eta_seconds)) if eta_seconds is not None else None
                self.monitor.update_training_metrics(
                    epoch=int(data.get("epoch", 0)),
                    step=int(data.get("step", 0)),
                    loss=float(data.get("loss", 0.0)),
                    learning_rate=float(data.get("learning_rate", 0.0)),
                    batch_size=int(data.get("batch_size", 0)),
                    throughput=float(data.get("throughput", 0.0)),
                    eta=eta,
                )
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid payload: {exc}")
            return {"success": True}

        @self.app.post("/api/progress/update")
        async def push_progress_update(request: Request):
            if not self.progress_tracker:
                raise HTTPException(
                    status_code=400, detail="Progress tracker not available"
                )
            data = await request.json()
            try:
                self.progress_tracker.record_remote_progress(data)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid payload: {exc}")
            return {"success": True}

        @self.app.get("/api/alerts")
        async def get_alerts():
            if not self.alert_system:
                return []

            alerts: List[Dict[str, Any]] = []
            for alert in self.alert_system.alert_history[-50:]:
                alerts.append(
                    {
                        "timestamp": alert.timestamp.isoformat(),
                        "rule_name": alert.rule_name,
                        "level": alert.level.value,
                        "message": alert.message,
                        "value": alert.value,
                        "threshold": alert.threshold,
                    }
                )
            return alerts

        @self.app.get("/api/alert_summary")
        async def get_alert_summary():
            if not self.alert_system:
                return {}
            return self.alert_system.get_alert_summary()

        @self.app.get("/api/performance_stats")
        async def get_performance_stats():
            if not self.progress_tracker:
                return {}
            return self.progress_tracker.get_performance_stats()

        @self.app.get("/api/trends")
        async def get_trends():
            if not self.monitor:
                return {}
            return analyze_performance_trends(self.monitor)

        @self.app.get("/api/anomalies")
        async def get_anomalies():
            if not self.monitor:
                return {}
            return detect_performance_anomalies(self.monitor)

        @self.app.get("/api/charts/training")
        async def get_training_chart():
            if not self.monitor:
                raise HTTPException(status_code=400, detail="Monitor not available")

            chart_dir = Path("web_charts")
            chart_dir.mkdir(exist_ok=True)
            plot_files = plot_training_metrics(
                self.monitor,
                chart_dir,
                metrics=["loss", "learning_rate", "throughput"],
            )
            return {"success": True, "files": [str(f) for f in plot_files]}

        @self.app.get("/api/charts/system")
        async def get_system_chart():
            if not self.monitor:
                raise HTTPException(status_code=400, detail="Monitor not available")

            chart_dir = Path("web_charts")
            chart_dir.mkdir(exist_ok=True)
            plot_files = plot_system_metrics(
                self.monitor,
                chart_dir,
                metrics=["cpu_percent", "memory_percent", "gpu_utilization"],
            )
            return {"success": True, "files": [str(f) for f in plot_files]}

        @self.app.get("/api/charts/progress")
        async def get_progress_chart():
            if not self.progress_tracker:
                raise HTTPException(
                    status_code=400, detail="Progress tracker not available"
                )

            chart_dir = Path("web_charts")
            chart_dir.mkdir(exist_ok=True)
            plot_files = plot_progress_tracker(self.progress_tracker, chart_dir)
            return {"success": True, "files": [str(f) for f in plot_files]}

        @self.app.get("/api/export/{format}")
        async def export_data(format: str):
            if format not in {"json", "csv", "parquet"}:
                raise HTTPException(status_code=400, detail="Unsupported format")

            from .monitor_utils import (
                export_alert_data,
                export_monitor_data,
                export_progress_data,
            )

            export_dir = Path("web_exports")
            export_dir.mkdir(exist_ok=True)

            files: List[str] = []
            if self.monitor:
                files.extend(
                    str(path)
                    for path in export_monitor_data(self.monitor, export_dir, [format])
                )
            if self.progress_tracker:
                files.extend(
                    str(path)
                    for path in export_progress_data(
                        self.progress_tracker, export_dir, [format]
                    )
                )
            if self.alert_system:
                files.extend(
                    str(path)
                    for path in export_alert_data(self.alert_system, export_dir, [format])
                )

            return {"success": True, "files": [str(f) for f in files]}

        @self.app.post("/api/control/start")
        async def start_monitoring(request: Request):
            await _update_run_info_from_request(request)
            if self.monitor:
                self.monitor.start_monitoring()
            return {"success": True, "run_info": self.current_run_info}

        @self.app.post("/api/control/stop")
        async def stop_monitoring():
            if self.monitor:
                self.monitor.stop_monitoring()
            return {"success": True}

        @self.app.post("/api/control/reset")
        async def reset_monitoring():
            if self.monitor:
                self.monitor.reset()
            if self.progress_tracker:
                self.progress_tracker.reset()
            if self.alert_system:
                self.alert_system.reset()
            return {"success": True}

    def _setup_socket_events(self) -> None:
        @self.socketio.event
        async def connect(sid, environ, auth=None):
            print(f"Client connected: {sid}")
            await self.socketio.emit(
                "status",
                {"message": "Connected to monitor server"},
                to=sid,
            )

        @self.socketio.event
        async def disconnect(sid):
            print(f"Client disconnected: {sid}")

        async def handle_update_request(sid):
            await self._emit_latest_data()

        register_update_handler = self.socketio.on("request_update")
        if register_update_handler:
            register_update_handler(handle_update_request)

    async def _emit_latest_data(self) -> None:
        try:
            if self.monitor and self.monitor.training_metrics_history:
                latest_training = self.monitor.training_metrics_history[-1]
                await self.socketio.emit(
                    "training_update",
                    {
                        "timestamp": latest_training.timestamp.isoformat(),
                        "epoch": latest_training.epoch,
                        "step": latest_training.step,
                        "loss": latest_training.loss,
                        "learning_rate": latest_training.learning_rate,
                        "throughput": latest_training.throughput,
                    },
                )

            if self.monitor and self.monitor.system_metrics_history:
                latest_system = self.monitor.system_metrics_history[-1]
                await self.socketio.emit(
                    "system_update",
                    {
                        "timestamp": latest_system.timestamp.isoformat(),
                        "cpu_percent": latest_system.cpu_percent,
                        "memory_percent": latest_system.memory_percent,
                        "gpu_utilization": latest_system.gpu_utilization,
                    },
                )

            if self.progress_tracker:
                current = self.progress_tracker.get_current_progress()
                if current:
                    await self.socketio.emit(
                        "progress_update",
                        {
                            "current_epoch": current.epoch,
                            "current_step": current.step,
                            "total_progress": current.total_progress,
                            "eta_total": str(current.eta_total)
                            if current.eta_total
                            else None,
                            "throughput": current.throughput,
                        },
                    )

            if self.alert_system and self.alert_system.alert_history:
                latest_alert = self.alert_system.alert_history[-1]
                await self.socketio.emit(
                    "alert_update",
                    {
                        "timestamp": latest_alert.timestamp.isoformat(),
                        "rule_name": latest_alert.rule_name,
                        "level": latest_alert.level.value,
                        "message": latest_alert.message,
                    },
                )
        except Exception as exc:
            print(f"Error emitting data: {exc}")

    async def _update_loop(self) -> None:
        while self.running:
            await self._emit_latest_data()
            await asyncio.sleep(1.0)

    def start(self):
        """启动 FastAPI 服务器"""
        print(f"🚀 Starting Web Monitor Server at http://{self.host}:{self.port}")
        config = uvicorn.Config(
            app=self.asgi_app,
            host=self.host,
            port=self.port,
            log_level="debug" if self.debug else "info",
        )
        self.server = uvicorn.Server(config)
        asyncio.run(self.server.serve())

    def stop(self):
        """请求停止服务器"""
        self.running = False
        if self.server:
            self.server.should_exit = True

    def set_monitor(self, monitor: TrainingMonitor):
        self.monitor = monitor

    def set_progress_tracker(self, progress_tracker: ProgressTracker):
        self.progress_tracker = progress_tracker

    def set_alert_system(self, alert_system: AlertSystem):
        self.alert_system = alert_system


def create_web_monitor(
    host: str = "0.0.0.0", port: int = 5000, debug: bool = False
) -> WebMonitorServer:
    return WebMonitorServer(host=host, port=port, debug=debug)
