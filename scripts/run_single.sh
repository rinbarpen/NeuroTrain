#!/usr/bin/env bash
# 单卡运行：训练 + 测试（使用 train.example.toml）
# 用法：./scripts/run_single.sh
#       CONFIG=configs/single/train.example.toml CUDA_VISIBLE_DEVICES=0 ./scripts/run_single.sh

set -e
cd "$(dirname "$0")/.."
CONFIG="${CONFIG:-configs/single/train.example.toml}"
RUN_ID="${RUN_ID:-run_$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if command -v uv &>/dev/null; then
    RUN_CMD="uv run python"
else
    RUN_CMD="python"
fi

[ ! -f "$CONFIG" ] && { echo "错误：配置文件不存在 $CONFIG"; exit 1; }

echo "单卡运行 | CONFIG=$CONFIG | RUN_ID=$RUN_ID"
$RUN_CMD main.py --train --test --config "$CONFIG" --run_id "$RUN_ID" \
    ${EPOCH:+--epoch "$EPOCH"} \
    ${MAIN_METRIC:+--main_metric "$MAIN_METRIC"} \
    ${BEST_BY:+--best_by "$BEST_BY"} \
    ${METRIC_DIRECTION:+--metric_direction "$METRIC_DIRECTION"} \
    ${EARLY_STOPPING:+--early_stopping} \
    ${PATIENCE:+--patience "$PATIENCE"} \
    "$@"
