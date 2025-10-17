#!/bin/bash
# DeepSpeed 训练启动脚本

# 设置默认参数
NUM_GPUS=2
CONFIG_FILE="configs/single/train-deepspeed.toml"
OUTPUT_DIR="./runs"
LOG_LEVEL="INFO"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --log_level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --help)
            echo "DeepSpeed 训练启动脚本"
            echo ""
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --num_gpus NUM        GPU 数量 (默认: 2)"
            echo "  --config FILE         配置文件路径 (默认: configs/single/train-deepspeed.toml)"
            echo "  --output_dir DIR      输出目录 (默认: ./runs)"
            echo "  --log_level LEVEL     日志级别 (默认: INFO)"
            echo "  --help                显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0 --num_gpus 4 --config configs/single/train-deepspeed.toml"
            echo "  $0 --num_gpus 2 --output_dir ./experiments/deepspeed"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 检查 DeepSpeed 是否安装
if ! command -v deepspeed &> /dev/null; then
    echo "错误: DeepSpeed 未安装"
    echo "请运行: pip install deepspeed"
    exit 1
fi

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export DEEPSPEED_LOG_LEVEL="$LOG_LEVEL"

echo "开始 DeepSpeed 训练..."
echo "GPU 数量: $NUM_GPUS"
echo "配置文件: $CONFIG_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "日志级别: $LOG_LEVEL"
echo ""

# 启动 DeepSpeed 训练
deepspeed \
    --num_gpus="$NUM_GPUS" \
    --master_port=29500 \
    main.py \
    --config "$CONFIG_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --deepspeed.log_level "$LOG_LEVEL"
