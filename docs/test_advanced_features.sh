#!/bin/bash
# 测试所有高级功能

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

echo "=========================================="
echo "测试LaTeX表格高级功能"
echo "=========================================="
echo

# 创建输出目录
OUTPUT_DIR="/tmp/latex_advanced_test"
mkdir -p "$OUTPUT_DIR"

echo "1. 测试：高亮最佳和次佳值"
echo "--------------------------------------"
conda run -n ntrain python tools/data_to_latex.py \
  -i tools/test_advanced.csv \
  -t table \
  --style booktabs \
  --highlight-best \
  --highlight-second \
  --metric-columns accuracy f1_score \
  --higher-is-better True True \
  -o "$OUTPUT_DIR/test1_highlight.tex"
echo "✓ 完成: $OUTPUT_DIR/test1_highlight.tex"
echo

echo "2. 测试：标注我们的模型"
echo "--------------------------------------"
conda run -n ntrain python tools/data_to_latex.py \
  -i tools/test_advanced.csv \
  -t table \
  --style booktabs \
  --our-model "OurModel" \
  -o "$OUTPUT_DIR/test2_ourmodel.tex"
echo "✓ 完成: $OUTPUT_DIR/test2_ourmodel.tex"
echo

echo "3. 测试：组合功能（高亮+模型标注）"
echo "--------------------------------------"
conda run -n ntrain python tools/data_to_latex.py \
  -i tools/test_advanced.csv \
  -t table \
  --style booktabs \
  --highlight-best \
  --highlight-second \
  --metric-columns accuracy f1_score \
  --higher-is-better True True \
  --our-model "OurModel" \
  -o "$OUTPUT_DIR/test3_combined.tex"
echo "✓ 完成: $OUTPUT_DIR/test3_combined.tex"
echo

echo "4. 测试：多数据集分组"
echo "--------------------------------------"
conda run -n ntrain python tools/data_to_latex.py \
  -i tools/test_multidataset.csv \
  -t table \
  --style booktabs \
  --highlight-best \
  --highlight-second \
  --metric-columns accuracy f1_score \
  --higher-is-better True True \
  --our-model "OurModel" \
  --group-column dataset \
  -o "$OUTPUT_DIR/test4_multidataset.tex"
echo "✓ 完成: $OUTPUT_DIR/test4_multidataset.tex"
echo

echo "5. 测试：混合指标方向（accuracy越高越好，但params越低越好）"
echo "--------------------------------------"
# 创建测试数据
cat > "$OUTPUT_DIR/test_mixed_metrics.csv" << EOF
model,accuracy,params,inference_time
ModelA,0.95,25.6,120
ModelB,0.93,5.3,45
ModelC,0.97,138.4,200
ModelD,0.91,4.2,38
EOF

conda run -n ntrain python tools/data_to_latex.py \
  -i "$OUTPUT_DIR/test_mixed_metrics.csv" \
  -t table \
  --style booktabs \
  --highlight-best \
  --highlight-second \
  --metric-columns accuracy params inference_time \
  --higher-is-better True False False \
  -o "$OUTPUT_DIR/test5_mixed_direction.tex"
echo "✓ 完成: $OUTPUT_DIR/test5_mixed_direction.tex"
echo

echo "=========================================="
echo "生成演示LaTeX文档"
echo "=========================================="

# 生成完整的LaTeX文档展示所有功能
cat > "$OUTPUT_DIR/advanced_features_demo.tex" << 'EOFTEX'
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{array}
\usepackage{geometry}
\geometry{margin=1in}

\title{Advanced Table Features Demo}
\author{NeuroTrain data\_to\_latex.py}
\date{\today}

\begin{document}

\maketitle

\section{Feature 1: Highlight Best and Second Best}

The tool can automatically highlight the best values (bold) and second-best values (italic) in specified metric columns.

EOFTEX

cat "$OUTPUT_DIR/test1_highlight.tex" >> "$OUTPUT_DIR/advanced_features_demo.tex"

cat >> "$OUTPUT_DIR/advanced_features_demo.tex" << 'EOFTEX'

\section{Feature 2: Mark Our Model}

Our model is marked with an underline to make it stand out.

EOFTEX

cat "$OUTPUT_DIR/test2_ourmodel.tex" >> "$OUTPUT_DIR/advanced_features_demo.tex"

cat >> "$OUTPUT_DIR/advanced_features_demo.tex" << 'EOFTEX'

\section{Feature 3: Combined Features}

Combining highlighting with model marking for maximum clarity.

EOFTEX

cat "$OUTPUT_DIR/test3_combined.tex" >> "$OUTPUT_DIR/advanced_features_demo.tex"

cat >> "$OUTPUT_DIR/advanced_features_demo.tex" << 'EOFTEX'

\section{Feature 4: Multi-Dataset Comparison}

When comparing across multiple datasets, the tool automatically groups results and finds best/second-best within each group.

EOFTEX

cat "$OUTPUT_DIR/test4_multidataset.tex" >> "$OUTPUT_DIR/advanced_features_demo.tex"

cat >> "$OUTPUT_DIR/advanced_features_demo.tex" << 'EOFTEX'

\section{Feature 5: Mixed Metric Directions}

Different metrics have different optimization directions. The tool supports this:
\begin{itemize}
  \item \textbf{Accuracy}: higher is better (True)
  \item \textbf{Params}: lower is better (False) - fewer parameters
  \item \textbf{Inference Time}: lower is better (False) - faster inference
\end{itemize}

EOFTEX

cat "$OUTPUT_DIR/test5_mixed_direction.tex" >> "$OUTPUT_DIR/advanced_features_demo.tex"

cat >> "$OUTPUT_DIR/advanced_features_demo.tex" << 'EOFTEX'

\section{Summary}

These advanced features make it easy to create publication-quality tables that clearly communicate your results:

\begin{itemize}
  \item \textbf{Bold}: Best values
  \item \textit{Italic}: Second-best values  
  \item \underline{Underline}: Your proposed method
  \item Automatic grouping for multi-dataset/task scenarios
  \item Flexible metric directions (higher/lower is better)
\end{itemize}

\end{document}
EOFTEX

echo "✓ 生成演示文档: $OUTPUT_DIR/advanced_features_demo.tex"
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
echo "  pdflatex advanced_features_demo.tex"
echo
echo "查看生成的表格代码:"
echo "  cat $OUTPUT_DIR/test3_combined.tex"

