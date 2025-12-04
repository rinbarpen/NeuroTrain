#!/bin/bash
# 统一运行脚本 - 支持训练、测试、预测和分析

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认值
ACTION=""
CONFIG=""
NUM_GPUS=1
TRAIN_MODE="single"
CONDA_ENV="ntrain"
DEVICE="cuda:0"
EXTRA_ARGS=""

# 帮助信息
show_help() {
    cat << EOF
用法: $0 <action> [选项]

统一运行脚本 - 支持训练、测试、预测和分析

Actions:
    train          训练模型
    test           测试模型
    predict        预测
    analyze        运行分析器
    quick-test     快速测试数据集功能

训练选项 (train):
    -c, --config FILE           配置文件路径 (必需)
    -g, --gpus N                使用的GPU数量 (默认: 1)
    -t, --train-mode MODE       训练模式: single, ddp, deepspeed (默认: single)
    -d, --device DEVICE         设备 (默认: cuda:0)

测试选项 (test):
    -c, --config FILE           配置文件路径 (必需)

预测选项 (predict):
    -c, --config FILE           配置文件路径 (必需)

分析选项 (analyze):
    <analyzer_name>             分析器名称: metrics, dataset, attention, mask, relation, lora

通用选项:
    -e, --env ENV               Conda环境名称 (默认: ntrain)
    -h, --help                  显示此帮助信息

示例:
    # 单GPU训练
    $0 train -c configs/single/train.toml

    # DDP多卡训练 (4卡)
    $0 train -c configs/ddp_example.toml -t ddp -g 4

    # DeepSpeed训练 (2卡)
    $0 train -c configs/deepspeed_example.yaml -t deepspeed -g 2

    # 测试
    $0 test -c configs/single/train.toml

    # 预测
    $0 predict -c configs/single/train.toml

    # 运行指标分析器
    $0 analyze metrics --run_id experiment_001

    # 快速测试
    $0 quick-test

EOF
}

# 解析参数
ACTION="$1"
shift

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG="$2"
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

# 检查action
if [ -z "$ACTION" ]; then
    echo -e "${RED}错误: 必须指定action${NC}"
    show_help
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

# 根据action执行
case "$ACTION" in
    train)
        if [ -z "$CONFIG" ]; then
            echo -e "${RED}错误: 训练模式需要指定配置文件 (-c)${NC}"
            exit 1
        fi
        echo -e "${BLUE}使用训练脚本...${NC}"
        bash scripts/train.sh -c "$CONFIG" -m train -g "$NUM_GPUS" -t "$TRAIN_MODE" -d "$DEVICE" $EXTRA_ARGS
        ;;
    test)
        if [ -z "$CONFIG" ]; then
            echo -e "${RED}错误: 测试模式需要指定配置文件 (-c)${NC}"
            exit 1
        fi
        echo -e "${BLUE}运行测试...${NC}"
        bash scripts/train.sh -c "$CONFIG" -m test -d "$DEVICE" $EXTRA_ARGS
        ;;
    predict)
        if [ -z "$CONFIG" ]; then
            echo -e "${RED}错误: 预测模式需要指定配置文件 (-c)${NC}"
            exit 1
        fi
        echo -e "${BLUE}运行预测...${NC}"
        bash scripts/train.sh -c "$CONFIG" -m predict -d "$DEVICE" $EXTRA_ARGS
        ;;
    analyze)
        echo -e "${BLUE}使用分析器脚本...${NC}"
        bash scripts/analyze.sh $EXTRA_ARGS
        ;;
    quick-test)
        echo -e "${BLUE}运行快速测试...${NC}"
        python scripts/run_quick_test.py
        ;;
    *)
        echo -e "${RED}错误: 未知的action: $ACTION${NC}"
        echo "支持的actions: train, test, predict, analyze, quick-test"
        show_help
        exit 1
        ;;
esac

