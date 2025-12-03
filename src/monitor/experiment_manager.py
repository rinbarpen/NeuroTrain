"""
实验浏览与数据加载工具

提供在 `runs/{PROJECT}/{RUN_ID}/...` 目录结构下发现可用实验、
并从指定目录读取监控数据的辅助函数。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterable


TRAINING_FILE_NAME = "training_metrics.json"
SYSTEM_FILE_NAME = "system_metrics.json"
PROGRESS_CANDIDATES = ("progress_data.json", "progress.json")
ALERT_FILE_NAME = "alerts.json"


@dataclass
class ExperimentEntry:
    """实验元数据"""

    id: str  # 相对 runs 根目录的路径，指向监控数据所在目录
    project: str
    run_id: str
    relative_dir: str  # run_id 内的相对路径，便于显示
    has_training_metrics: bool
    has_system_metrics: bool
    has_progress: bool
    has_alerts: bool
    updated_at: Optional[str] = None


def _iter_monitor_dirs(run_dir: Path) -> Iterable[Path]:
    """找到 run 目录下所有包含训练指标的监控子目录"""
    seen = set()
    for training_file in run_dir.glob(f"**/{TRAINING_FILE_NAME}"):
        monitor_dir = training_file.parent
        try:
            rel = monitor_dir.relative_to(run_dir)
        except ValueError:
            continue
        # 避免重复目录
        key = rel.as_posix()
        if key in seen:
            continue
        seen.add(key)
        yield monitor_dir


def discover_experiments(runs_root: Path) -> List[Dict[str, Any]]:
    """
    扫描 runs 根目录，返回所有可用的实验列表
    """
    experiments: List[Dict[str, Any]] = []
    runs_root = runs_root.expanduser().resolve()
    if not runs_root.exists():
        return experiments

    for project_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        project = project_dir.name
        for run_dir in sorted(p for p in project_dir.iterdir() if p.is_dir()):
            run_id = run_dir.name
            for monitor_dir in _iter_monitor_dirs(run_dir):
                relative_dir = monitor_dir.relative_to(run_dir).as_posix()
                entry = ExperimentEntry(
                    id=monitor_dir.relative_to(runs_root).as_posix(),
                    project=project,
                    run_id=run_id,
                    relative_dir=relative_dir,
                    has_training_metrics=(monitor_dir / TRAINING_FILE_NAME).exists(),
                    has_system_metrics=(monitor_dir / SYSTEM_FILE_NAME).exists(),
                    has_progress=any(
                        (monitor_dir / candidate).exists()
                        for candidate in PROGRESS_CANDIDATES
                    ),
                    has_alerts=(monitor_dir / ALERT_FILE_NAME).exists(),
                    updated_at=_get_latest_mtime(monitor_dir),
                )
                experiments.append(asdict(entry))

    return experiments


def load_experiment_snapshot(runs_root: Path, experiment_id: str) -> Optional[Dict[str, Any]]:
    """
    读取指定实验（以相对路径表示）的监控快照数据
    """
    runs_root = runs_root.expanduser().resolve()
    monitor_dir = (runs_root / experiment_id).expanduser().resolve()

    # 安全校验，确保目录位于 runs 根目录内
    try:
        monitor_dir.relative_to(runs_root)
    except ValueError:
        return None

    if not monitor_dir.exists():
        return None

    training_metrics = _read_json_list(monitor_dir / TRAINING_FILE_NAME)
    system_metrics = _read_json_list(monitor_dir / SYSTEM_FILE_NAME)
    progress_data = _read_json_list(_find_existing_file(monitor_dir, PROGRESS_CANDIDATES))
    alerts = _read_json_list(monitor_dir / ALERT_FILE_NAME)

    project, run_id, relative_dir = _parse_experiment_id(runs_root, monitor_dir)

    snapshot = {
        "status": {
            "monitor_active": False,
            "progress_active": False,
            "alert_active": bool(alerts),
            "project": project,
            "run_id": run_id,
            "relative_dir": relative_dir,
            "updated_at": _get_latest_mtime(monitor_dir),
        },
        "training_metrics": training_metrics,
        "system_metrics": system_metrics,
        "progress": _build_progress_summary(progress_data, training_metrics),
        "alerts": alerts,
        "performance": _build_performance_stats(progress_data, training_metrics),
    }

    return snapshot


def _read_json_list(file_path: Optional[Path]) -> List[Dict[str, Any]]:
    """读取 JSON 列表文件"""
    if not file_path or not file_path.exists():
        return []
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def _find_existing_file(directory: Path, candidates: Iterable[str]) -> Optional[Path]:
    """返回第一个存在的候选文件"""
    for candidate in candidates:
        path = directory / candidate
        if path.exists():
            return path
    return None


def _parse_experiment_id(runs_root: Path, monitor_dir: Path) -> tuple[str, str, str]:
    """解析 experiment id，返回 (project, run_id, relative_dir)"""
    try:
        rel_path = monitor_dir.relative_to(runs_root)
    except ValueError:
        return "", "", ""

    parts = list(rel_path.parts)
    if len(parts) < 2:
        return "", "", rel_path.as_posix()

    project = parts[0]
    run_id = parts[1]
    relative_dir = Path(*parts[2:]).as_posix() if len(parts) > 2 else "."
    return project, run_id, relative_dir


def _get_latest_mtime(directory: Path) -> Optional[str]:
    """获取目录内相关文件的最后修改时间"""
    latest = None
    for pattern in (
        TRAINING_FILE_NAME,
        SYSTEM_FILE_NAME,
        *PROGRESS_CANDIDATES,
        ALERT_FILE_NAME,
    ):
        file_path = directory / pattern
        if file_path.exists():
            mtime = file_path.stat().st_mtime
            if latest is None or mtime > latest:
                latest = mtime
    if latest is None:
        return None
    return datetime.fromtimestamp(latest).isoformat(timespec="seconds")


def _build_progress_summary(progress_data: List[Dict[str, Any]], training_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """根据进度数据或训练指标构建摘要"""
    if progress_data:
        latest = progress_data[-1]
        return {
            "current_epoch": latest.get("epoch", 0),
            "current_step": latest.get("step", 0),
            "total_epochs": latest.get("total_epochs"),
            "total_steps": latest.get("total_steps"),
            "epoch_progress": latest.get("epoch_progress"),
            "total_progress": latest.get("total_progress"),
            "eta_epoch": latest.get("eta_epoch"),
            "eta_total": latest.get("eta_total"),
            "throughput": latest.get("throughput"),
        }

    # 如果没有独立的进度数据，尝试使用训练指标估计
    if training_metrics:
        latest = training_metrics[-1]
        return {
            "current_epoch": latest.get("epoch", 0),
            "current_step": latest.get("step", 0),
            "total_epochs": None,
            "total_steps": None,
            "epoch_progress": None,
            "total_progress": None,
            "eta_epoch": None,
            "eta_total": None,
            "throughput": latest.get("throughput"),
        }

    return {}


def _build_performance_stats(progress_data: List[Dict[str, Any]], training_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """依据已有数据构建性能统计信息"""
    stats: Dict[str, Any] = {
        "avg_step_time": None,
        "avg_epoch_time": None,
        "avg_throughput": None,
        "min_throughput": None,
        "max_throughput": None,
    }

    if progress_data:
        step_times = [item.get("step_time") for item in progress_data if item.get("step_time")]
        epoch_times = [item.get("epoch_time") for item in progress_data if item.get("epoch_time")]
        throughputs = [item.get("throughput") for item in progress_data if item.get("throughput")]
    else:
        step_times = []
        epoch_times = []
        throughputs = [item.get("throughput") for item in training_metrics if item.get("throughput")]

    if step_times:
        stats["avg_step_time"] = sum(step_times) / len(step_times)
    if epoch_times:
        stats["avg_epoch_time"] = sum(epoch_times) / len(epoch_times)
    if throughputs:
        stats["avg_throughput"] = sum(throughputs) / len(throughputs)
        stats["min_throughput"] = min(throughputs)
        stats["max_throughput"] = max(throughputs)

    return stats


