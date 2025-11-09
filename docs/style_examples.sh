#!/bin/bash
# 快速演示各种样式的示例脚本

echo "======================================"
echo "LaTeX表格样式快速演示"
echo "======================================"
echo

cd "$(dirname "$0")/.."

echo "1. Simple样式（默认）"
echo "--------------------------------------"
conda run -n ntrain python tools/data_to_latex.py \
  -i tools/example_data.csv -t table --style simple -c model accuracy | tail -15
echo

echo "2. Booktabs样式（专业）"
echo "--------------------------------------"
conda run -n ntrain python tools/data_to_latex.py \
  -i tools/example_data.csv -t table --style booktabs -c model accuracy | tail -15
echo

echo "3. Lined样式（全线条）"
echo "--------------------------------------"
conda run -n ntrain python tools/data_to_latex.py \
  -i tools/example_data.csv -t table --style lined -c model accuracy | tail -15
echo

echo "4. Fancy样式（美化）+ 自定义对齐"
echo "--------------------------------------"
conda run -n ntrain python tools/data_to_latex.py \
  -i tools/example_data.csv -t table --style fancy \
  --column-align "lr" -c model accuracy | tail -20
echo

echo "======================================"
echo "完成！更多信息请查看:"
echo "  python tools/data_to_latex.py --list-styles"
echo "  cat tools/TABLE_STYLES_GUIDE.md"
echo "======================================"

