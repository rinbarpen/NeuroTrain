#!/usr/bin/env python3
"""
NeuroTrain Monitor - Streamlit 前端

使用 Streamlit 构建的交互式监控面板，通过调用 FastAPI 监控后端的
REST API 获取实时训练与系统指标。
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--api-base",
        default=os.environ.get("NEUROTRAIN_MONITOR_API", "http://localhost:5000"),
        help="Monitor 后端的基础 URL（默认: http://localhost:5000）",
    )
    parser.add_argument(
        "--page-title",
        default="NeuroTrain Monitor",
        help="Streamlit 页面标题",
    )
    parser.add_argument(
        "--page-icon",
        default="🧠",
        help="页面图标",
    )
    parser.add_argument(
        "--default-timeout",
        type=float,
        default=float(os.environ.get("NEUROTRAIN_MONITOR_TIMEOUT", 5.0)),
        help="API 请求默认超时时间（秒）",
    )
    args, _ = parser.parse_known_args()
    return args


CLI_ARGS = _parse_cli_args()
st.set_page_config(
    page_title=CLI_ARGS.page_title,
    page_icon=CLI_ARGS.page_icon,
    layout="wide",
)


def _normalize_base(url: str) -> str:
    url = url.strip()
    if not url:
        return "http://localhost:5000"
    if url.endswith("/"):
        url = url[:-1]
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    return url


if "api_base" not in st.session_state:
    st.session_state["api_base"] = _normalize_base(CLI_ARGS.api_base)

if "request_timeout" not in st.session_state:
    st.session_state["request_timeout"] = max(1.0, float(CLI_ARGS.default_timeout))


SESSION = requests.Session()


def api_request(
    path: str,
    method: str = "GET",
    json: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """调用监控后端 API。"""
    base = st.session_state["api_base"]
    url = f"{base}{path}"
    try:
        response = SESSION.request(
            method=method,
            url=url,
            timeout=st.session_state["request_timeout"],
            json=json,
        )
        response.raise_for_status()
        if response.headers.get("content-type", "").startswith("application/json"):
            return response.json()
        return response.text
    except requests.RequestException as exc:
        st.toast(f"请求 {path} 失败: {exc}", icon="⚠️")
        return None


def records_to_dataframe(records: Optional[List[Dict[str, Any]]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
    return df


def render_metrics_row(status: Dict[str, Any], progress: Dict[str, Any], alerts: List[Dict[str, Any]]):
    col1, col2, col3, col4 = st.columns(4)
    monitor_label = "Active" if status.get("monitor_active") else "Inactive"
    col1.metric("Monitor", monitor_label)

    progress_value = progress.get("total_progress", 0.0)
    col2.metric("Progress (%)", f"{progress_value:.1f}")

    col3.metric("Alerts", len(alerts))

    throughput = progress.get("throughput")
    throughput_value = f"{throughput:.1f} samples/s" if throughput else "--"
    col4.metric("Throughput", throughput_value)


def render_training_charts(training_df: pd.DataFrame):
    st.subheader("Training Metrics")
    if training_df.empty:
        st.info("暂无训练指标数据")
        return
    training_df = training_df.set_index("timestamp")
    chart_cols = []
    for column in ["loss", "learning_rate", "throughput"]:
        if column in training_df.columns:
            chart_cols.append(column)
    if not chart_cols:
        st.warning("无可用训练指标字段")
        return
    st.line_chart(training_df[chart_cols])
    st.dataframe(training_df.tail(50))


def render_system_charts(system_df: pd.DataFrame):
    st.subheader("System Metrics")
    if system_df.empty:
        st.info("暂无系统指标数据")
        return
    system_df = system_df.set_index("timestamp")
    chart_cols = [col for col in ["cpu_percent", "memory_percent", "gpu_utilization"] if col in system_df.columns]
    st.line_chart(system_df[chart_cols])
    st.dataframe(system_df.tail(50))


def render_progress_section(progress: Dict[str, Any], performance: Dict[str, Any], summary: Dict[str, Any]):
    st.subheader("Progress")
    total_progress = progress.get("total_progress", 0.0)
    st.progress(min(max(total_progress / 100.0, 0.0), 1.0))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Epoch", f"{progress.get('current_epoch', 0)} / {progress.get('total_epochs', '--')}")
    col2.metric("Step", f"{progress.get('current_step', 0)} / {progress.get('total_steps', '--')}")
    col3.metric("Epoch Progress (%)", f"{progress.get('epoch_progress', 0.0):.1f}")
    col4.metric("ETA", progress.get("eta_total") or "--")

    st.markdown("**Performance Stats**")
    stats_cols = st.columns(4)
    stats_cols[0].metric("Avg Step Time (s)", f"{performance.get('avg_step_time', 0.0):.3f}")
    stats_cols[1].metric("Avg Epoch Time (s)", f"{performance.get('avg_epoch_time', 0.0):.3f}")
    stats_cols[2].metric("Avg Throughput", f"{performance.get('avg_throughput', 0.0):.2f}")
    stats_cols[3].metric("Min Throughput", f"{performance.get('min_throughput', 0.0):.2f}")

    if summary:
        st.json(summary, expanded=False)


def render_alerts_section(alerts: List[Dict[str, Any]], summary: Dict[str, Any]):
    st.subheader("Alerts")
    if not alerts:
        st.success("暂无告警")
    else:
        alerts_df = records_to_dataframe(alerts)
        st.dataframe(alerts_df.sort_values("timestamp", ascending=False).head(20))
    if summary:
        st.markdown("**Alert Summary**")
        st.json(summary, expanded=False)


def render_analysis_section(trends: Dict[str, Any], anomalies: Dict[str, Any]):
    st.subheader("Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Trends**")
        if trends:
            st.json(trends, expanded=False)
        else:
            st.write("暂无趋势数据")
    with col2:
        st.markdown("**Anomalies**")
        if anomalies:
            st.json(anomalies, expanded=False)
        else:
            st.write("暂无异常")


def render_control_panel():
    st.subheader("Control Panel")
    action_cols = st.columns(4)

    if action_cols[0].button("Start Monitoring", type="primary"):
        api_request("/api/control/start", method="POST")
        st.experimental_rerun()
    if action_cols[1].button("Stop Monitoring"):
        api_request("/api/control/stop", method="POST")
        st.experimental_rerun()
    if action_cols[2].button("Reset Data"):
        api_request("/api/control/reset", method="POST")
        st.experimental_rerun()

    export_format = action_cols[3].selectbox("Export Format", ["json", "csv", "parquet"])
    if action_cols[3].button("Export Data"):
        result = api_request(f"/api/export/{export_format}")
        if isinstance(result, dict) and result.get("success"):
            files = result.get("files", [])
            if files:
                st.success(f"导出成功: {files}")
            else:
                st.info("导出完成，但未返回文件路径")
        else:
            st.error(f"导出失败: {result}")


def main():
    st.title("NeuroTrain Streamlit Monitor")
    st.caption(
        "使用 Streamlit 构建的 NeuroTrain 监控前端，需配合 start_web_monitor.py 启动的 FastAPI 后端。",
    )

    with st.sidebar:
        st.header("Connection")
        api_base = st.text_input("Monitor API Base", st.session_state["api_base"])
        st.session_state["api_base"] = _normalize_base(api_base)
        timeout = st.slider("Request Timeout (s)", min_value=1, max_value=30, value=int(st.session_state["request_timeout"]))
        st.session_state["request_timeout"] = float(timeout)

        auto_refresh = st.checkbox("Auto Refresh", value=True)
        refresh_interval = st.slider("Refresh Interval (s)", min_value=1, max_value=10, value=2)
        if st.button("Refresh Now"):
            st.experimental_rerun()

    status_data = api_request("/api/status") or {}
    training_data = api_request("/api/training_metrics") or []
    system_data = api_request("/api/system_metrics") or []
    progress_data = api_request("/api/progress") or {}
    alerts_data = api_request("/api/alerts") or []
    performance_data = api_request("/api/performance_stats") or {}
    progress_summary = api_request("/api/progress_summary") or {}
    alert_summary = api_request("/api/alert_summary") or {}
    trend_data = api_request("/api/trends") or {}
    anomaly_data = api_request("/api/anomalies") or {}

    render_metrics_row(status_data, progress_data, alerts_data)

    chart_col1, chart_col2 = st.columns([2, 1])
    with chart_col1:
        render_training_charts(records_to_dataframe(training_data))
    with chart_col2:
        render_system_charts(records_to_dataframe(system_data))

    render_progress_section(progress_data, performance_data, progress_summary)
    render_alerts_section(alerts_data, alert_summary)
    render_analysis_section(trend_data, anomaly_data)
    render_control_panel()

    if auto_refresh:
        time.sleep(refresh_interval)
        st.experimental_rerun()


if __name__ == "__main__":
    main()

