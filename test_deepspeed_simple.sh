#!/bin/bash
# DeepSpeed简单测试脚本

echo "=========================================="
echo "DeepSpeed多卡训练测试"
echo "=========================================="

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1  # 使用两块GPU
GPU_COUNT=2

# 配置文件
CONFIG_FILE="configs/deepspeed_quick_test.toml"

echo -e "\n使用 $GPU_COUNT 块GPU"
echo "配置文件: $CONFIG_FILE"
echo ""
echo "启动方式1: 使用 deepspeed 命令"
echo "=========================================="

# 使用 deepspeed 命令启动
deepspeed --num_gpus=$GPU_COUNT \
    --master_port=29501 \
    main_deepspeed.py \
    --config "$CONFIG_FILE" \
    --train \
    --task train \
    --run_id deepspeed_test_$(date +%Y%m%d_%H%M%S)

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ DeepSpeed测试成功!"
else
    echo "✗ DeepSpeed测试失败，退出码: $EXIT_CODE"
    echo ""
    echo "如果遇到错误，请尝试以下方案："
    echo ""
    echo "方案1: 使用 torchrun 代替 deepspeed 命令"
    echo "  torchrun --nproc_per_node=$GPU_COUNT main_deepspeed.py --config $CONFIG_FILE --task train"
    echo ""
    echo "方案2: 使用标准 DDP 训练（已测试成功）"
    echo "  ./test_multi_gpu.sh"
fi
echo "=========================================="

exit $EXIT_CODE

