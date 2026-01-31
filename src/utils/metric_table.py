"""
Unified metric scores table output for train/test/predict.
Prints per-class and mean scores for every metric to console (Rich Table).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from rich.console import Console
from rich.table import Table


def print_metric_scores_table(
    class_labels: List[str],
    metric_labels: List[str],
    class_table_rows: List[Tuple[str, Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]],
    summary_rows: List[Tuple[str, Dict[str, float]]],
    *,
    style_key: str = "default",
    title_class: str = "Metric Class Mean Score",
    title_summary: str = "Summary of Metric",
) -> None:
    """
    Print two Rich tables: per-class scores (mean ± std) and summary (mean per metric).

    Args:
        class_labels: List of class names.
        metric_labels: List of metric names.
        class_table_rows: List of (stage_prefix, mean_by_metric_class, std_by_metric_class).
            mean_by_metric_class[metric][class] = float, same for std.
        summary_rows: List of (stage_name, mean_scores). mean_scores[metric] = float.
        style_key: Key for get_style_sequence (e.g. 'train', 'test', 'default').
        title_class: Title for the per-class table.
        title_summary: Title for the summary table.
    """
    try:
        from src.config import get_style_sequence
    except Exception:
        get_style_sequence = None

    if get_style_sequence is not None:
        metric_styles = get_style_sequence(
            f"{style_key}.metric_table", len(metric_labels), fallback="default.metric_table"
        )
        summary_styles = get_style_sequence(
            f"{style_key}.summary_table", len(metric_labels), fallback="default.summary_table"
        )
    else:
        metric_styles = ["white"] * len(metric_labels)
        summary_styles = ["white"] * len(metric_labels)

    console = Console()

    # Per-class table
    table = Table(title=title_class)
    table.add_column("Class/Metric", justify="center")
    for metric, style in zip(metric_labels, metric_styles):
        table.add_column(metric, justify="center", style=style)
    for stage_prefix, mean_dict, std_dict in class_table_rows:
        for class_label in class_labels:
            row_label = f"{stage_prefix}/{class_label}"
            cells = []
            for metric in metric_labels:
                m = mean_dict.get(metric, {}).get(class_label, 0.0)
                s = std_dict.get(metric, {}).get(class_label, 0.0)
                cells.append(f"{m:.3f} ± {s:.3f}")
            table.add_row(row_label, *cells)
    console.print(table)

    # Summary table
    table = Table(title=title_summary)
    table.add_column("Metric", justify="center")
    for metric, style in zip(metric_labels, summary_styles):
        table.add_column(metric, justify="center", style=style)
    for stage_name, mean_scores in summary_rows:
        table.add_row(stage_name, *[f"{mean_scores.get(m, 0.0):.3f}" for m in metric_labels])
    console.print(table)
