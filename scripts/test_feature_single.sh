#!/usr/bin/env bash
# 单卡功能测试：main_metric / best_by / early_stopping / Inferencer
# 用法：./scripts/test_feature_single.sh  或  bash scripts/test_feature_single.sh

set -e
cd "$(dirname "$0")/.."
CONFIG="${CONFIG:-configs/single/train.example.toml}"
RUN_ID="feature_single_$(date +%Y%m%d_%H%M%S)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=========================================="
echo "单卡功能测试 (main_metric / best_by / Inferencer)"
echo "=========================================="
echo "CONFIG=$CONFIG"
echo "RUN_ID=$RUN_ID"
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

echo "[1/2] 训练 + [2/2] 推理测试 (一次执行：train 后自动 test，Inferencer 全量指标)"
$RUN_CMD main.py \
    --train \
    --test \
    --config "$CONFIG" \
    --run_id "$RUN_ID" \
    --epoch 4 \
    --main_metric accuracy \
    --best_by main_metric \
    --metric_direction max \
    --early_stopping \
    --patience 2

EXIT_CODE=$?
echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "单卡功能测试通过"
else
    echo "单卡功能测试失败，退出码: $EXIT_CODE"
fi
echo "=========================================="
exit $EXIT_CODE
