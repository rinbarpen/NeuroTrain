#!/bin/bash
# 多卡训练测试脚本：分类、分割、识别

echo "开始多卡训练测试..."

# 检查环境
if [ "$CONDA_DEFAULT_ENV" != "ntrain" ]; then
    echo "请先激活 ntrain 环境: conda activate ntrain"
    # 尝试激活
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate ntrain || exit 1
fi

export CUDA_VISIBLE_DEVICES=0,1
GPU_COUNT=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "使用 $GPU_COUNT 张 GPU"

# 1. 分类任务 (Classification) - MNIST
echo "=========================================="
echo "1. 测试分类任务 (MNIST)"
echo "=========================================="
if [ -f "configs/single/train-mnist.toml" ]; then
    torchrun --nproc_per_node=$GPU_COUNT --master_port=29501 main.py         --config configs/single/train-mnist.toml         --task train         --run_id test_cls_mnist_$(date +%s)
else
    echo "未找到 configs/single/train-mnist.toml"
fi

# 2. 分割任务 (Segmentation) - DRIVE (视网膜血管分割)
echo "=========================================="
echo "2. 测试分割任务 (DRIVE)"
echo "=========================================="
if [ -f "configs/single/train-drive.toml" ]; then
    torchrun --nproc_per_node=$GPU_COUNT --master_port=29502 main.py         --config configs/single/train-drive.toml         --task train         --run_id test_seg_drive_$(date +%s)
else
    echo "未找到 configs/single/train-drive.toml"
fi

# 3. 识别/CLIP任务 (Recognition) - Brain MRI CLIP
echo "=========================================="
echo "3. 测试识别/CLIP任务 (Brain MRI)"
echo "=========================================="
if [ -f "configs/single/train-clip-brain-mri.toml" ]; then
    torchrun --nproc_per_node=$GPU_COUNT --master_port=29503 main.py         --config configs/single/train-clip-brain-mri.toml         --task train         --run_id test_rec_clip_$(date +%s)
else
    echo "未找到 configs/single/train-clip-brain-mri.toml"
fi

echo "=========================================="
echo "测试完成"
