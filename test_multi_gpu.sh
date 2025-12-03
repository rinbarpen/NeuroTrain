#!/bin/bash
# 测试双卡DDP训练脚本

echo "=========================================="
echo "测试多卡训练环境"
echo "=========================================="

# 激活 conda 环境
echo -e "\n[1/5] 激活 conda ntrain 环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ntrain

# 检查环境激活状态
if [ "$CONDA_DEFAULT_ENV" != "ntrain" ]; then
    echo "错误：无法激活 ntrain 环境"
    exit 1
fi
echo "✓ 已激活环境: $CONDA_DEFAULT_ENV"

# 检查PyTorch和torchrun
echo -e "\n检查PyTorch版本..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}, CUDA可用: {torch.cuda.is_available()}, GPU数量: {torch.cuda.device_count()}')"

# 检查GPU状态
echo -e "\n[2/5] 检查GPU状态..."
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1  # 使用两块GPU
export NCCL_DEBUG=INFO  # 启用NCCL调试信息

# 获取GPU数量
GPU_COUNT=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo -e "\n[3/5] 将使用 $GPU_COUNT 块GPU进行测试"

# 配置文件
CONFIG_FILE="configs/ddp_quick_test.toml"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误：配置文件 $CONFIG_FILE 不存在"
    exit 1
fi

echo -e "\n[4/5] 使用配置文件：$CONFIG_FILE"
echo "启动DDP训练测试..."
echo "=========================================="

# 启动DDP训练
torchrun --nproc_per_node=$GPU_COUNT \
    --master_port=29500 \
    main.py \
    --config "$CONFIG_FILE" \
    --task train \
    --run_id ddp_test_$(date +%Y%m%d_%H%M%S)

# 检查退出状态
EXIT_CODE=$?

echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n[5/5] ✓ 多卡训练测试成功完成!"
else
    echo -e "\n[5/5] ✗ 多卡训练测试失败，退出码: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE

