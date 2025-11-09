#!/bin/bash
# 测试所有表格样式

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

echo "=========================================="
echo "测试所有LaTeX表格样式"
echo "=========================================="
echo

# 创建输出目录
OUTPUT_DIR="/tmp/latex_style_test"
mkdir -p "$OUTPUT_DIR"

styles=("simple" "booktabs" "lined" "minimal" "fancy")

for style in "${styles[@]}"; do
    echo "测试样式: $style"
    conda run -n ntrain python tools/data_to_latex.py \
        -i tools/example_data.csv \
        -t table \
        --style "$style" \
        --caption "Test ${style^} Style" \
        --label "tab:$style" \
        -o "$OUTPUT_DIR/${style}_table.tex"
    echo "✓ 完成: $OUTPUT_DIR/${style}_table.tex"
    echo
done

echo "=========================================="
echo "测试自定义列对齐"
echo "=========================================="
echo

# 测试列对齐
echo "测试列对齐: lrccl"
conda run -n ntrain python tools/data_to_latex.py \
    -i tools/example_data.csv \
    -t table \
    --style booktabs \
    --column-align "lrccl" \
    --caption "Custom Column Alignment" \
    -o "$OUTPUT_DIR/custom_align_table.tex"
echo "✓ 完成"
echo

echo "=========================================="
echo "生成样式对比文档"
echo "=========================================="
echo

# 生成LaTeX对比文档
cat > "$OUTPUT_DIR/style_comparison.tex" << 'EOF'
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{array}
\usepackage{geometry}
\geometry{margin=1in}

\title{LaTeX Table Styles Comparison}
\author{NeuroTrain data\_to\_latex.py}
\date{\today}

\begin{document}

\maketitle

\section{Simple Style}
This is the default style with basic \texttt{\textbackslash hline} separators.

EOF

cat "$OUTPUT_DIR/simple_table.tex" >> "$OUTPUT_DIR/style_comparison.tex"

cat >> "$OUTPUT_DIR/style_comparison.tex" << 'EOF'

\section{Booktabs Style}
Professional style using the \texttt{booktabs} package. Requires \texttt{\textbackslash usepackage\{booktabs\}}.

EOF

cat "$OUTPUT_DIR/booktabs_table.tex" >> "$OUTPUT_DIR/style_comparison.tex"

cat >> "$OUTPUT_DIR/style_comparison.tex" << 'EOF'

\section{Lined Style}
Every row has a horizontal line separator.

EOF

cat "$OUTPUT_DIR/lined_table.tex" >> "$OUTPUT_DIR/style_comparison.tex"

cat >> "$OUTPUT_DIR/style_comparison.tex" << 'EOF'

\section{Minimal Style}
Only top and bottom horizontal lines.

EOF

cat "$OUTPUT_DIR/minimal_table.tex" >> "$OUTPUT_DIR/style_comparison.tex"

cat >> "$OUTPUT_DIR/style_comparison.tex" << 'EOF'

\section{Fancy Style}
Enhanced style with booktabs and increased line spacing. Requires \texttt{\textbackslash usepackage\{booktabs\}} and \texttt{\textbackslash usepackage\{array\}}.

EOF

cat "$OUTPUT_DIR/fancy_table.tex" >> "$OUTPUT_DIR/style_comparison.tex"

cat >> "$OUTPUT_DIR/style_comparison.tex" << 'EOF'

\section{Custom Column Alignment}
Example with custom column alignment (lrccl).

EOF

cat "$OUTPUT_DIR/custom_align_table.tex" >> "$OUTPUT_DIR/style_comparison.tex"

cat >> "$OUTPUT_DIR/style_comparison.tex" << 'EOF'

\section{Summary}

\begin{table}[htbp]
\centering
\caption{Style Comparison Summary}
\begin{tabular}{lll}
\toprule
Style & Required Packages & Best For \\
\midrule
Simple & None & Quick generation, general use \\
Booktabs & booktabs & Academic papers \\
Lined & None & Data-dense tables \\
Minimal & None & Modern layouts \\
Fancy & booktabs, array & High-quality publications \\
\bottomrule
\end{tabular}
\end{table}

\end{document}
EOF

echo "✓ 生成对比文档: $OUTPUT_DIR/style_comparison.tex"
echo

echo "=========================================="
echo "所有测试完成！"
echo "=========================================="
echo
echo "生成的文件位于: $OUTPUT_DIR/"
ls -lh "$OUTPUT_DIR/"
echo
echo "可以编译LaTeX文档查看效果:"
echo "  cd $OUTPUT_DIR"
echo "  pdflatex style_comparison.tex"

