#!/usr/bin/env bash
# 多卡功能测试 (DDP)：main_metric / best_by / early_stopping / Inferencer
# 用法：CUDA_VISIBLE_DEVICES=0,1 ./scripts/test_feature_multicard.sh

set -e
cd "$(dirname "$0")/.."
CONFIG="${CONFIG:-configs/single/train.example_ddp.toml}"
RUN_ID="feature_ddp_$(date +%Y%m%d_%H%M%S)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
if [ "$GPU_COUNT" -lt 2 ]; then
    echo "多卡测试建议至少 2 张 GPU，当前 CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (数量: $GPU_COUNT)"
    echo "可设置: CUDA_VISIBLE_DEVICES=0,1 ./scripts/test_feature_multicard.sh"
fi

echo "=========================================="
echo "多卡功能测试 (DDP, main_metric / best_by / Inferencer)"
echo "=========================================="
echo "CONFIG=$CONFIG"
echo "RUN_ID=$RUN_ID"
echo "GPU_COUNT=$GPU_COUNT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo ""

if [ ! -f "$CONFIG" ]; then
    echo "错误：配置文件不存在 $CONFIG"
    exit 1
fi

if command -v uv &>/dev/null; then
    RUN_CMD="uv run python"
else
    RUN_CMD="python"
fi

echo "[1/2] DDP 训练 (torchrun, nproc_per_node=$GPU_COUNT)..."
torchrun --nproc_per_node="$GPU_COUNT" \
    --master_port=29500 \
    main.py \
    --train \
    --config "$CONFIG" \
    --run_id "$RUN_ID" \
    --epoch 4 \
    --main_metric accuracy \
    --best_by main_metric \
    --metric_direction max \
    --early_stopping \
    --patience 2

echo ""
echo "[2/2] 推理测试 (Inferencer 全量指标，单进程)..."
export CUDA_VISIBLE_DEVICES=0
# test 需复用同 run：用 --continue_from 指向刚完成的 run 目录以加载权重
OUT_DIR=$(grep -E '^output_dir\s*=' "$CONFIG" | head -1 | sed 's/.*=\s*"\?\([^"]*\)"\?.*/\1/' | tr -d ' ')
TASK_NAME=$(grep -E '^task\s*=' "$CONFIG" | head -1 | sed 's/.*=\s*"\?\([^"]*\)"\?.*/\1/' | tr -d ' ')
CONTINUE_DIR="${OUT_DIR:-./runs}/${TASK_NAME:-FeatureTestDDP}/${RUN_ID}"
$RUN_CMD main.py \
    --test \
    --config "$CONFIG" \
    --run_id "$RUN_ID" \
    --continue_from "$CONTINUE_DIR"

EXIT_CODE=$?
echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "多卡功能测试通过"
else
    echo "多卡功能测试失败，退出码: $EXIT_CODE"
fi
echo "=========================================="
exit $EXIT_CODE
