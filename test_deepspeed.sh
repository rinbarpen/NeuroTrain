#!/bin/bash
# 测试双卡DeepSpeed训练脚本

echo "=========================================="
echo "测试DeepSpeed多卡训练环境"
echo "=========================================="

# 激活 conda 环境
echo -e "\n[1/6] 激活 conda ntrain 环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ntrain

# 检查环境激活状态
if [ "$CONDA_DEFAULT_ENV" != "ntrain" ]; then
    echo "错误：无法激活 ntrain 环境"
    exit 1
fi
echo "✓ 已激活环境: $CONDA_DEFAULT_ENV"

# 检查PyTorch和DeepSpeed
echo -e "\n[2/6] 检查PyTorch和DeepSpeed版本..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}, CUDA可用: {torch.cuda.is_available()}, GPU数量: {torch.cuda.device_count()}')"

# 检查DeepSpeed是否安装
echo -e "\n检查DeepSpeed安装状态..."
if python -c "import deepspeed" 2>/dev/null; then
    python -c "import deepspeed; print(f'✓ DeepSpeed已安装，版本: {deepspeed.__version__}')"
    DS_INSTALLED=true
else
    echo "✗ DeepSpeed未安装"
    echo ""
    echo "是否要安装DeepSpeed? (需要几分钟时间)"
    echo "建议使用以下命令安装:"
    echo "  conda activate ntrain"
    echo "  pip install deepspeed"
    echo ""
    DS_INSTALLED=false
fi

if [ "$DS_INSTALLED" = false ]; then
    echo "请先安装DeepSpeed后再运行此测试"
    exit 1
fi

# 检查GPU状态
echo -e "\n[3/6] 检查GPU状态..."
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1  # 使用两块GPU
export NCCL_DEBUG=INFO  # 启用NCCL调试信息

# 获取GPU数量
GPU_COUNT=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo -e "\n[4/6] 将使用 $GPU_COUNT 块GPU进行测试"

# 配置文件
CONFIG_FILE="configs/deepspeed_quick_test.toml"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误：配置文件 $CONFIG_FILE 不存在"
    exit 1
fi

echo -e "\n[5/6] 使用配置文件：$CONFIG_FILE"
echo "启动DeepSpeed训练测试..."
echo "=========================================="

# 启动DeepSpeed训练
# 方式1: 使用deepspeed命令
deepspeed --num_gpus=$GPU_COUNT \
    --master_port=29500 \
    main_deepspeed.py \
    --config "$CONFIG_FILE" \
    --train \
    --task train \
    --run_id deepspeed_test_$(date +%Y%m%d_%H%M%S)

# 如果deepspeed命令不可用，使用torchrun
# torchrun --nproc_per_node=$GPU_COUNT \
#     --master_port=29500 \
#     main_deepspeed.py \
#     --config "$CONFIG_FILE" \
#     --task train \
#     --run_id deepspeed_test_$(date +%Y%m%d_%H%M%S)

# 检查退出状态
EXIT_CODE=$?

echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n[6/6] ✓ DeepSpeed多卡训练测试成功完成!"
else
    echo -e "\n[6/6] ✗ DeepSpeed多卡训练测试失败，退出码: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE

