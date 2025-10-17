#!/bin/bash
# DDP训练启动脚本（TOML配置版本）

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 根据实际GPU数量调整
export NCCL_DEBUG=INFO  # 可选：启用NCCL调试信息

# 检查配置文件是否存在
CONFIG_FILE=${1:-"configs/ddp_example.toml"}
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误：配置文件 $CONFIG_FILE 不存在"
    echo "使用方法：$0 [配置文件路径]"
    echo "示例：$0 configs/ddp_segmentation.toml"
    exit 1
fi

# 获取GPU数量
GPU_COUNT=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "检测到 $GPU_COUNT 个GPU"

# 启动DDP训练
echo "使用配置文件：$CONFIG_FILE"
echo "启动DDP训练..."

torchrun --nproc_per_node=$GPU_COUNT main.py \
    --config "$CONFIG_FILE" \
    --task train \
    --run_id ddp_run_$(date +%Y%m%d_%H%M%S)

# 或者使用python -m torch.distributed.launch启动（旧版本）
# python -m torch.distributed.launch --nproc_per_node=$GPU_COUNT main.py \
#     --config "$CONFIG_FILE" \
#     --task train \
#     --run_id ddp_run_$(date +%Y%m%d_%H%M%S)

echo "DDP训练完成"
