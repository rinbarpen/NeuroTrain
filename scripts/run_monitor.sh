#!/usr/bin/env bash
set -euo pipefail

# 项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# 加速 matplotlib 初始化
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/matplotlib_config_$$

DEFAULT_ARGS=(
  --host 0.0.0.0
  --port 5000
  --log-dir runs/monitor
  --log-interval 1.0
  --save-interval 60.0
)

if [[ $# -gt 0 ]]; then
  RUN_ARGS=("$@")
else
  RUN_ARGS=("${DEFAULT_ARGS[@]}")
fi

run_with_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    return 1
  fi

  echo "[monitor] 使用 uv 启动：${RUN_ARGS[*]}"
  UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}" \
    uv run --no-sync python run_monitor.py "${RUN_ARGS[@]}" || return 1
}

run_with_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    return 1
  fi

  echo "[monitor] 使用 conda ntrain 启动：${RUN_ARGS[*]}"
  conda run -n ntrain python run_monitor.py "${RUN_ARGS[@]}" || return 1
}

if run_with_uv; then
  exit 0
fi

echo "[monitor] uv 不可用或执行失败，尝试切换到 conda 环境..."
if run_with_conda; then
  exit 0
fi

echo "❌ 无法启动 monitor，请确认已安装 uv 或 conda 的 ntrain 环境。"
exit 1

