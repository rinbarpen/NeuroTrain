"""Reporting utilities for NeuroTrain.

This package enables rendering experiment summaries as attractive HTML
reports, with optional PDF export.
"""

from .context_helpers import ReportSummary, build_report_context, load_metrics_json
from .html_reporter import HTMLReportGenerator, ReportRenderingError

__all__ = [
    "HTMLReportGenerator",
    "ReportRenderingError",
    "ReportSummary",
    "build_report_context",
    "load_metrics_json",
]

