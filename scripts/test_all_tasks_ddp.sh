#!/bin/bash
set -e

# ========================================================
# 多卡训练综合测试脚本
# 包含：分类(MNIST)、分割(DRIVE)、识别(Brain MRI CLIP)
# ========================================================

# 检查是否在 conda 环境中
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "ntrain" ]; then
    echo "警告: 当前未激活 'ntrain' conda 环境。"
    echo "建议运行: conda activate ntrain"
    # 尝试自动激活 (仅在 bash 下有效)
    if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate ntrain || echo "无法自动激活 ntrain 环境，继续尝试..."
    fi
fi

# 检查 CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "错误: 未找到 nvidia-smi，请确保已安装 NVIDIA 驱动。"
    exit 1
fi

# 设置使用的 GPU (默认使用前两张卡)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

echo "=================================================="
echo "开始多卡 DDP 训练测试"
echo "使用 GPU: $CUDA_VISIBLE_DEVICES (共 $NUM_GPUS 张)"
echo "环境: ${CONDA_DEFAULT_ENV:-Unknown}"
echo "=================================================="

# 确保 output 目录存在
mkdir -p output runs

# --------------------------------------------------------
# 1. 分类任务 (Classification) - MNIST
# --------------------------------------------------------
echo -e "\n[1/3] 测试分类任务 (MNIST)..."
echo "配置文件: configs/single/train-mnist.toml"

# 使用 torchrun 启动 DDP
# --nproc_per_node: 每个节点的 GPU 数量
# --master_port: 主节点端口 (防止端口冲突，每个任务使用不同端口)
torchrun --nproc_per_node=$NUM_GPUS --master_port=29501 main.py \
    --config configs/single/train-mnist.toml \
    --task train \
    --run_id test_ddp_cls_mnist \
    --opts train.epoch=2 train.save_period=2 train.batch_size=64

if [ $? -eq 0 ]; then
    echo "✅ MNIST 分类任务测试通过"
else
    echo "❌ MNIST 分类任务测试失败"
    exit 1
fi

# --------------------------------------------------------
# 2. 分割任务 (Segmentation) - DRIVE
# --------------------------------------------------------
echo -e "\n[2/3] 测试分割任务 (DRIVE)..."
echo "配置文件: configs/single/train-drive.toml"

# 检查数据是否存在 (简单检查)
if [ ! -d "data/DRIVE" ]; then
    echo "警告: data/DRIVE 目录不存在，跳过分割任务测试，或者请先下载数据。"
else
    torchrun --nproc_per_node=$NUM_GPUS --master_port=29502 main.py \
        --config configs/single/train-drive.toml \
        --task train \
        --run_id test_ddp_seg_drive \
        --opts train.epoch=2 train.save_every_n_epoch=2 train.batch_size=2
        
    if [ $? -eq 0 ]; then
        echo "✅ DRIVE 分割任务测试通过"
    else
        echo "❌ DRIVE 分割任务测试失败"
        # 分割失败不退出，继续测试下一个
    fi
fi

# --------------------------------------------------------
# 3. 识别任务 (Recognition) - Brain MRI CLIP
# --------------------------------------------------------
echo -e "\n[3/3] 测试识别任务 (Brain MRI CLIP)..."
echo "配置文件: configs/single/train-clip-brain-mri.toml"

# 检查数据
if [ ! -d "data/brain_mri_clip" ]; then
    echo "警告: data/brain_mri_clip 目录不存在，跳过识别任务测试。"
else
    torchrun --nproc_per_node=$NUM_GPUS --master_port=29503 main.py \
        --config configs/single/train-clip-brain-mri.toml \
        --task train \
        --run_id test_ddp_rec_clip \
        --opts train.epoch=2 train.save_period=2 train.batch_size=8

    if [ $? -eq 0 ]; then
        echo "✅ Brain MRI CLIP 识别任务测试通过"
    else
        echo "❌ Brain MRI CLIP 识别任务测试失败"
    fi
fi

echo -e "\n=================================================="
echo "测试结束"
echo "=================================================="
