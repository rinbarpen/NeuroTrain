#!/bin/bash
# DeepSpeed UNet + DRIVE 数据集训练脚本

echo "=========================================="
echo "DeepSpeed UNet + DRIVE 训练"
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

# 检查PyTorch和DeepSpeed
echo -e "\n[2/5] 检查环境..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}, CUDA可用: {torch.cuda.is_available()}, GPU数量: {torch.cuda.device_count()}')"

# 检查DeepSpeed是否安装
if python -c "import deepspeed" 2>/dev/null; then
    python -c "import deepspeed; print(f'✓ DeepSpeed已安装，版本: {deepspeed.__version__}')"
else
    echo "✗ DeepSpeed未安装"
    exit 1
fi

# 检查GPU状态
echo -e "\n[3/5] 检查GPU状态..."
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1  # 使用两块GPU
GPU_COUNT=2

# 配置文件
CONFIG_FILE="configs/deepspeed_drive_unet.toml"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误：配置文件 $CONFIG_FILE 不存在"
    exit 1
fi

echo -e "\n[4/5] 使用 $GPU_COUNT 块GPU"
echo "配置文件: $CONFIG_FILE"
echo ""
echo "启动 DeepSpeed 训练..."
echo "=========================================="

# 使用 deepspeed 命令启动
deepspeed --num_gpus=$GPU_COUNT \
    --master_port=29502 \
    main_deepspeed.py \
    --config "$CONFIG_FILE" \
    --train \
    --task train \
    --run_id drive_$(date +%Y%m%d_%H%M%S)

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "[5/5] ✓ DeepSpeed训练成功完成!"
    echo ""
    echo "查看训练结果："
    echo "  ls -lh runs/DeepSpeed_DRIVE_UNet/"
else
    echo "[5/5] ✗ DeepSpeed训练失败，退出码: $EXIT_CODE"
    echo ""
    echo "调试建议："
    echo "  1. 检查数据集是否存在: ls -lh data/DRIVE/"
    echo "  2. 检查配置文件: cat $CONFIG_FILE"
    echo "  3. 查看完整错误信息"
fi
echo "=========================================="

exit $EXIT_CODE

