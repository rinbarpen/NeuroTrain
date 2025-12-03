#!/bin/bash
# Analyzer脚本 - 运行各种分析工具

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 默认值
ANALYZER=""
CONDA_ENV="ntrain"
EXTRA_ARGS=""

# Analyzer列表
AVAILABLE_ANALYZERS=(
    "metrics"      # 指标分析器
    "dataset"      # 数据集分析器
    "attention"    # 注意力分析器
    "mask"         # 掩码分析器
    "relation"     # 关系分析器
    "lora"         # LoRA分析器
)

# 帮助信息
show_help() {
    cat << EOF
用法: $0 [选项] <analyzer_name>

Analyzer脚本 - 运行各种分析工具

可用的Analyzer:
    metrics      - 指标分析器 (metrics_analyzer.py)
    dataset      - 数据集分析器 (dataset_analyzer.py)
    attention    - 注意力分析器 (attention_analyzer.py)
    mask         - 掩码分析器 (mask_analyzer.py)
    relation     - 关系分析器 (relation_analyzer.py)
    lora         - LoRA分析器 (lora_analyzer.py)

选项:
    -e, --env ENV       Conda环境名称 (默认: ntrain)
    -h, --help          显示此帮助信息

示例:
    # 运行指标分析器
    $0 metrics --run_id experiment_001

    # 运行数据集分析器
    $0 dataset --root_dir data/cifar10

    # 运行注意力分析器
    $0 attention --model_path runs/model.pth

    # 运行掩码分析器
    $0 mask --input_dir data/images --output_dir outputs/masks

    # 运行关系分析器
    $0 relation --config configs/relation.yaml

    # 运行LoRA分析器
    $0 lora --model_path runs/model.pth --lora_path runs/lora.pt

EOF
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            CONDA_ENV="$2"
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
            if [ -z "$ANALYZER" ]; then
                ANALYZER="$1"
            else
                EXTRA_ARGS="$EXTRA_ARGS $1"
            fi
            shift
            ;;
    esac
done

# 检查必需参数
if [ -z "$ANALYZER" ]; then
    echo -e "${RED}错误: 必须指定analyzer名称${NC}"
    show_help
    exit 1
fi

# 检查analyzer是否可用
if [[ ! " ${AVAILABLE_ANALYZERS[@]} " =~ " ${ANALYZER} " ]]; then
    echo -e "${RED}错误: 未知的analyzer: $ANALYZER${NC}"
    echo "可用的analyzer: ${AVAILABLE_ANALYZERS[*]}"
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
echo -e "${GREEN}NeuroTrain Analyzer${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Analyzer: $ANALYZER"
echo ""

# 根据analyzer类型执行
case "$ANALYZER" in
    metrics)
        echo -e "${GREEN}运行指标分析器...${NC}"
        python tools/analyzers/metrics_analyzer.py $EXTRA_ARGS
        ;;
    dataset)
        echo -e "${GREEN}运行数据集分析器...${NC}"
        python tools/analyzers/dataset_analyzer.py $EXTRA_ARGS
        ;;
    attention)
        echo -e "${GREEN}运行注意力分析器...${NC}"
        python tools/analyzers/attention_analyzer.py $EXTRA_ARGS
        ;;
    mask)
        echo -e "${GREEN}运行掩码分析器...${NC}"
        python tools/analyzers/mask_analyzer.py $EXTRA_ARGS
        ;;
    relation)
        echo -e "${GREEN}运行关系分析器...${NC}"
        python tools/analyzers/relation_analyzer.py $EXTRA_ARGS
        ;;
    lora)
        echo -e "${GREEN}运行LoRA分析器...${NC}"
        python tools/analyzers/lora_analyzer.py $EXTRA_ARGS
        ;;
    *)
        echo -e "${RED}错误: 未知的analyzer: $ANALYZER${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}分析完成!${NC}"
echo -e "${GREEN}========================================${NC}"

