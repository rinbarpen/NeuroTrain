"""Helper utilities for constructing report contexts.

These helpers aggregate artifacts produced by training, evaluation and
monitoring pipelines into a context dictionary understood by
``HTMLReportGenerator``.

The goal is to keep report rendering code simple by providing sensible
defaults for common keys such as ``summary``, ``metrics`` and
``charts``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass
class ReportSummary:
    title: str
    task: Optional[str] = None
    model_name: Optional[str] = None
    model_info: Optional[str] = None
    dataset: Optional[str] = None
    samples: Optional[int] = None
    runtime: Optional[str] = None
    generated_at: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None


def build_report_context(
    summary: ReportSummary,
    metrics_overall: Optional[Dict[str, float]] = None,
    metrics_per_class: Optional[Dict[str, Dict[str, float]]] = None,
    charts: Optional[Sequence[Any]] = None,
    monitor_json: Optional[Path] = None,
    artifacts: Optional[Iterable[Dict[str, Any]]] = None,
    notes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Build a context dictionary for ``HTMLReportGenerator``.

    Parameters
    ----------
    summary:
        High level experiment summary (see :class:`ReportSummary`).
    metrics_overall:
        Mapping of metric name to value (e.g. global accuracy).
    metrics_per_class:
        Mapping of class name to individual metrics.
    charts:
        Sequence of image entries (paths or dictionaries).
    monitor_json:
        Path to monitor JSON file to embed in the report.
    artifacts:
        Iterable of dictionaries with ``name``, ``path`` and optional
        ``description`` keys for artefacts to list in the report.
    notes:
        Optional sequence of textual notes to include at the end.
    """

    metrics_section: Dict[str, Any] = {}
    if metrics_overall:
        metrics_section["overall"] = metrics_overall
    if metrics_per_class:
        metric_names: List[str] = sorted({m for v in metrics_per_class.values() for m in v})
        metrics_section["per_class"] = {
            "metric_names": metric_names,
            "values": metrics_per_class,
        }

    context: Dict[str, Any] = {
        "title": summary.title,
        "summary": asdict(summary),
        "metrics": metrics_section if metrics_section else None,
        "charts": list(charts or []),
        "monitor": str(monitor_json) if monitor_json else None,
        "artifacts": list(artifacts or []),
        "notes": list(notes or []),
    }

    return context


def load_metrics_json(metrics_json: Path) -> Dict[str, Any]:
    """Load metrics data saved by analyzers.

    Parameters
    ----------
    metrics_json:
        Path to the metrics JSON produced by ``MetricsAnalyzer`` or other
        analyzers.
    """

    return json.loads(metrics_json.read_text("utf-8"))

