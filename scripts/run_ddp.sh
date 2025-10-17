#!/bin/bash
# DDP训练启动脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 根据实际GPU数量调整

# 启动DDP训练
torchrun --nproc_per_node=4 main.py \
    --config configs/ddp_example.yaml \
    --task train \
    --run_id ddp_run_$(date +%Y%m%d_%H%M%S)

# 或者使用python -m torch.distributed.launch启动（旧版本）
# python -m torch.distributed.launch --nproc_per_node=4 main.py \
#     --config configs/ddp_example.yaml \
#     --task train \
#     --run_id ddp_run_$(date +%Y%m%d_%H%M%S)
