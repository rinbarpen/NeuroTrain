#!/bin/bash
# 训练脚本 - 支持普通训练、DDP多卡训练、DeepSpeed训练

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 默认值
CONFIG=""
MODE="train"  # train, test, predict
NUM_GPUS=1
TRAIN_MODE="single"  # single, ddp, deepspeed
CONDA_ENV="ntrain"
DEVICE="cuda:0"
EXTRA_ARGS=""

# 帮助信息
show_help() {
    cat << EOF
用法: $0 [选项]

训练脚本 - 支持普通训练、DDP多卡训练、DeepSpeed训练

选项:
    -c, --config FILE           配置文件路径 (必需)
    -m, --mode MODE             运行模式: train, test, predict (默认: train)
    -g, --gpus N                使用的GPU数量 (默认: 1)
    -t, --train-mode MODE       训练模式: single, ddp, deepspeed (默认: single)
    -e, --env ENV               Conda环境名称 (默认: ntrain)
    -d, --device DEVICE         设备 (默认: cuda:0)
    -h, --help                  显示此帮助信息

示例:
    # 单GPU训练
    $0 -c configs/single/train.toml --train

    # DDP多卡训练 (4卡)
    $0 -c configs/ddp_example.toml -t ddp -g 4

    # DeepSpeed训练 (2卡)
    $0 -c configs/deepspeed_example.yaml -t deepspeed -g 2

    # 测试模式
    $0 -c configs/single/train.toml -m test

    # 预测模式
    $0 -c configs/single/train.toml -m predict

EOF
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -g|--gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        -t|--train-mode)
            TRAIN_MODE="$2"
            shift 2
            ;;
        -e|--env)
            CONDA_ENV="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS="$@"
            break
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# 检查必需参数
if [ -z "$CONFIG" ]; then
    echo -e "${RED}错误: 必须指定配置文件 (-c)${NC}"
    show_help
    exit 1
fi

# 检查配置文件是否存在
if [ ! -f "$CONFIG" ]; then
    echo -e "${RED}错误: 配置文件不存在: $CONFIG${NC}"
    exit 1
fi

# 激活conda环境
if command -v conda &> /dev/null; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV"
    echo -e "${GREEN}已激活conda环境: $CONDA_ENV${NC}"
else
    echo -e "${YELLOW}警告: conda未找到，跳过环境激活${NC}"
fi

# 进入项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}NeuroTrain 训练脚本${NC}"
echo -e "${GREEN}========================================${NC}"
echo "配置文件: $CONFIG"
echo "运行模式: $MODE"
echo "训练模式: $TRAIN_MODE"
echo "GPU数量: $NUM_GPUS"
echo "设备: $DEVICE"
echo ""

# 根据训练模式执行
case "$TRAIN_MODE" in
    single)
        # 单GPU训练
        echo -e "${GREEN}启动单GPU训练...${NC}"
        python main.py \
            -c "$CONFIG" \
            --device "$DEVICE" \
            --$MODE \
            $EXTRA_ARGS
        ;;
    ddp)
        # DDP多卡训练
        if [ "$NUM_GPUS" -lt 2 ]; then
            echo -e "${YELLOW}警告: DDP模式需要至少2个GPU，使用单GPU模式${NC}"
            python main.py \
                -c "$CONFIG" \
                --device "$DEVICE" \
                --$MODE \
                $EXTRA_ARGS
        else
            echo -e "${GREEN}启动DDP多卡训练 (${NUM_GPUS}卡)...${NC}"
            torchrun \
                --nproc_per_node=$NUM_GPUS \
                --master_port=29500 \
                main.py \
                -c "$CONFIG" \
                --$MODE \
                $EXTRA_ARGS
        fi
        ;;
    deepspeed)
        # DeepSpeed训练
        if ! command -v deepspeed &> /dev/null; then
            echo -e "${RED}错误: DeepSpeed未安装，请运行: pip install deepspeed${NC}"
            exit 1
        fi
        
        if [ "$NUM_GPUS" -lt 1 ]; then
            NUM_GPUS=1
        fi
        
        echo -e "${GREEN}启动DeepSpeed训练 (${NUM_GPUS}卡)...${NC}"
        deepspeed \
            --num_gpus=$NUM_GPUS \
            main_deepspeed.py \
            -c "$CONFIG" \
            --$MODE \
            $EXTRA_ARGS
        ;;
    *)
        echo -e "${RED}错误: 未知的训练模式: $TRAIN_MODE${NC}"
        echo "支持的模式: single, ddp, deepspeed"
        exit 1
        ;;
esac

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}训练完成!${NC}"
echo -e "${GREEN}========================================${NC}"

