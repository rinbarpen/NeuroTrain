#!/usr/bin/env bash
# 多卡运行 (DDP)：训练 + 测试（使用 train.example_ddp.toml）
# 用法：CUDA_VISIBLE_DEVICES=0,1 ./scripts/run_multicard.sh

set -e
cd "$(dirname "$0")/.."
CONFIG="${CONFIG:-configs/single/train.example_ddp.toml}"
RUN_ID="${RUN_ID:-run_ddp_$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
[ "$GPU_COUNT" -lt 2 ] && echo "建议至少 2 张 GPU，当前数量: $GPU_COUNT"

if command -v uv &>/dev/null; then
    RUN_CMD="uv run python"
else
    RUN_CMD="python"
fi

[ ! -f "$CONFIG" ] && { echo "错误：配置文件不存在 $CONFIG"; exit 1; }

echo "多卡运行 (nproc=$GPU_COUNT) | CONFIG=$CONFIG | RUN_ID=$RUN_ID"
torchrun --nproc_per_node="$GPU_COUNT" --master_port="${MASTER_PORT:-29500}" \
    main.py --train --config "$CONFIG" --run_id "$RUN_ID" \
    ${EPOCH:+--epoch "$EPOCH"} \
    ${MAIN_METRIC:+--main_metric "$MAIN_METRIC"} \
    ${BEST_BY:+--best_by "$BEST_BY"} \
    ${METRIC_DIRECTION:+--metric_direction "$METRIC_DIRECTION"} \
    ${EARLY_STOPPING:+--early_stopping} \
    ${PATIENCE:+--patience "$PATIENCE"} \
    "$@"

echo "推理测试 (Inferencer)..."
OUT_DIR=$(grep -E '^output_dir\s*=' "$CONFIG" | head -1 | sed 's/.*=\s*"\?\([^"]*\)"\?.*/\1/' | tr -d ' ')
TASK_NAME=$(grep -E '^task\s*=' "$CONFIG" | head -1 | sed 's/.*=\s*"\?\([^"]*\)"\?.*/\1/' | tr -d ' ')
CONTINUE_DIR="${OUT_DIR:-./runs}/${TASK_NAME:-FeatureTestDDP}/${RUN_ID}"
export CUDA_VISIBLE_DEVICES=0
$RUN_CMD main.py --test --config "$CONFIG" --run_id "$RUN_ID" --continue_from "$CONTINUE_DIR"
