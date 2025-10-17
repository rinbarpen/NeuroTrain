#!/bin/bash
# DeepSpeed训练启动脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 根据实际GPU数量调整

# 启动DeepSpeed训练
deepspeed --num_gpus=4 main_deepspeed.py \
    --config configs/your_config.yaml \
    --task train \
    --run_id deepspeed_run_$(date +%Y%m%d_%H%M%S)

# 或者使用torchrun启动（推荐）
# torchrun --nproc_per_node=4 main_deepspeed.py \
#     --config configs/your_config.yaml \
#     --task train \
#     --run_id deepspeed_run_$(date +%Y%m%d_%H%M%S)
