#!/bin/bash
# 便捷脚本：使用conda环境运行data_to_latex.py
# 用法: ./tools/run_data_to_latex.sh [参数]

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"

cd "$PROJECT_ROOT"

# 使用conda的ntrain环境运行
conda run -n ntrain python tools/data_to_latex.py "$@"

