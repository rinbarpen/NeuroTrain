# 数据文件转LaTeX列表工具使用指南

## 功能简介

`data_to_latex.py` 是一个强大的工具，可以将各种数据文件（CSV、Excel、JSON等）转换为LaTeX格式的表格或列表。

## 支持的数据格式

- CSV (`.csv`)
- Excel (`.xls`, `.xlsx`)
- JSON (`.json`)
- Parquet (`.parquet`)
- TSV (`.tsv`, `.txt`)

## 支持的LaTeX格式

### 1. Table（标准表格）
生成标准的LaTeX表格，适合打印在论文中。

```bash
python tools/data_to_latex.py -i data.csv -t table --caption "实验结果" --label "tab:results"
```

### 2. Longtable（长表格）
生成支持跨页的长表格，适合数据量较大的情况。

```bash
python tools/data_to_latex.py -i data.csv -t longtable
```

### 3. Itemize（无序列表）
生成无序列表，可以使用模板自定义格式。

```bash
python tools/data_to_latex.py -i data.csv -t itemize --template "{name}: {score} points"
```

### 4. Enumerate（有序列表）
生成有序列表。

```bash
python tools/data_to_latex.py -i data.csv -t enumerate
```

### 5. Description（描述列表）
生成描述列表，适合展示键值对数据。

```bash
python tools/data_to_latex.py -i data.csv -t description --key-column name
```

## 使用示例

### 示例1：转换CSV文件为表格

```bash
python tools/data_to_latex.py -i data/results.csv -t table -o output/table.tex
```

### 示例2：只选择特定列

```bash
python tools/data_to_latex.py -i data/results.csv -t table -c name score accuracy -o output/table.tex
```

### 示例3：使用自定义模板的列表

```bash
python tools/data_to_latex.py -i data/models.csv -t itemize --template "{model}: Accuracy={accuracy}, F1={f1_score}"
```

### 示例4：限制行数

```bash
python tools/data_to_latex.py -i data/large_data.csv -t table --max-rows 50
```

### 示例5：查看数据信息

```bash
python tools/data_to_latex.py -i data/results.csv --show-info
```

## 输出示例

### Table输出
```latex
\begin{table}[htbp]
\centering
\caption{实验结果}
\label{tab:results}
\begin{tabular}{lll}
\hline
Model & Accuracy & F1-Score \\
\hline
ResNet50 & 0.95 & 0.94 \\
VGG16 & 0.92 & 0.91 \\
\hline
\end{tabular}
\end{table}
```

### Itemize输出
```latex
\begin{itemize}
  \item ResNet50: Accuracy=0.95, F1=0.94
  \item VGG16: Accuracy=0.92, F1=0.91
\end{itemize}
```

### Description输出
```latex
\begin{description}
  \item[ResNet50] Accuracy: 0.95, F1-Score: 0.94
  \item[VGG16] Accuracy: 0.92, F1-Score: 0.91
\end{description}
```

## 高级用法

### 批量处理
可以结合shell脚本批量处理多个文件：

```bash
for file in data/*.csv; do
    python tools/data_to_latex.py -i "$file" -t table -o "output/$(basename $file .csv).tex"
done
```

### 在LaTeX中使用

生成的代码可以直接复制到LaTeX文档中，或者使用 `\input{}` 命令包含：

```latex
\documentclass{article}
\usepackage{longtable}  % 如果使用longtable格式

\begin{document}

\section{实验结果}

\input{output/table.tex}

\end{document}
```

## 注意事项

1. **特殊字符**：工具会自动转义LaTeX特殊字符（如 `&`, `%`, `$`, `#`, `_` 等）
2. **长表格**：使用longtable格式时，需要在LaTeX文档中添加 `\usepackage{longtable}`
3. **编码**：输出文件使用UTF-8编码
4. **数据质量**：确保数据文件格式正确，缺失值会被转换为空字符串

## 常见问题

### Q: 如何处理中文数据？
A: 工具支持UTF-8编码，可以正常处理中文。在LaTeX中需要使用支持中文的编译器（如XeLaTeX）和相应的宏包（如ctex）。

### Q: 表格列太多怎么办？
A: 可以使用 `-c` 参数只选择需要的列，或者考虑使用横向排版或调整列宽。

### Q: 如何自定义表格样式？
A: 生成的代码是基础格式，你可以在生成后手动调整或修改工具源码来自定义样式。

## 示例数据文件

创建一个示例CSV文件用于测试：

```csv
model,accuracy,f1_score,params
ResNet50,0.9523,0.9412,25.6M
VGG16,0.9234,0.9145,138.4M
EfficientNet,0.9678,0.9589,5.3M
MobileNet,0.9012,0.8934,4.2M
```

保存为 `example.csv`，然后运行：

```bash
python tools/data_to_latex.py -i example.csv -t table --caption "Model Comparison" --label "tab:models"
```

