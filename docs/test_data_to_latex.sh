#!/bin/bash
# 测试脚本：验证data_to_latex.py的各种功能

set -e  # 遇到错误时退出

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

echo "=========================================="
echo "测试 data_to_latex.py 工具"
echo "=========================================="
echo

# 创建输出目录
mkdir -p /tmp/latex_test_output

echo "1. 测试 CSV -> Table"
conda run -n ntrain python tools/data_to_latex.py \
  -i tools/example_data.csv -t table \
  --caption "Test Table" --label "tab:test" \
  -o /tmp/latex_test_output/table.tex
echo "✓ 完成"
echo

echo "2. 测试 JSON -> Itemize"
conda run -n ntrain python tools/data_to_latex.py \
  -i tools/example_data.json -t itemize \
  --template "{model}: {accuracy}" \
  -o /tmp/latex_test_output/itemize.tex
echo "✓ 完成"
echo

echo "3. 测试 CSV -> Enumerate (选择列)"
conda run -n ntrain python tools/data_to_latex.py \
  -i tools/example_data.csv -t enumerate \
  -c model accuracy \
  -o /tmp/latex_test_output/enumerate.tex
echo "✓ 完成"
echo

echo "4. 测试 CSV -> Description"
conda run -n ntrain python tools/data_to_latex.py \
  -i tools/example_data.csv -t description \
  --key-column model \
  -o /tmp/latex_test_output/description.tex
echo "✓ 完成"
echo

echo "5. 测试 CSV -> Longtable (限制行数)"
conda run -n ntrain python tools/data_to_latex.py \
  -i tools/example_data.csv -t longtable \
  --max-rows 3 \
  -o /tmp/latex_test_output/longtable.tex
echo "✓ 完成"
echo

echo "6. 测试 --show-info"
conda run -n ntrain python tools/data_to_latex.py \
  -i tools/example_data.csv --show-info
echo "✓ 完成"
echo

echo "=========================================="
echo "所有测试通过！"
echo "=========================================="
echo
echo "生成的文件位于: /tmp/latex_test_output/"
ls -lh /tmp/latex_test_output/
echo
echo "查看生成的table.tex文件:"
cat /tmp/latex_test_output/table.tex

