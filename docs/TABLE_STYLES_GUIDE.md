# LaTeX 表格样式指南

## 概述

`data_to_latex.py` 工具现在支持多种预设的LaTeX表格样式，每种样式都有不同的外观和特点。

## 可用样式

### 1. Simple (简单样式) - 默认
**描述**: 基础表格样式，使用 `\hline` 分隔

**特点**:
- 无需额外包
- 简单明了
- 适合快速生成表格

**使用示例**:
```bash
python tools/data_to_latex.py -i data.csv -t table --style simple
```

**输出示例**:
```latex
\begin{table}[htbp]
\centering
\begin{tabular}{lll}
\hline
model & accuracy & f1_score \\
\hline
ResNet50 & 0.9523 & 0.9412 \\
VGG16 & 0.9234 & 0.9145 \\
\hline
\end{tabular}
\end{table}
```

---

### 2. Booktabs (专业样式)
**描述**: 使用 `booktabs` 包的专业排版样式

**特点**:
- 专业的线条样式
- 学术论文标准
- 需要 `\usepackage{booktabs}`

**使用示例**:
```bash
python tools/data_to_latex.py -i data.csv -t table --style booktabs
```

**输出示例**:
```latex
\begin{table}[htbp]
\centering
\begin{tabular}{lll}
\toprule
model & accuracy & f1_score \\
\midrule
ResNet50 & 0.9523 & 0.9412 \\
VGG16 & 0.9234 & 0.9145 \\
\bottomrule
\end{tabular}
\end{table}
```

**LaTeX前言**:
```latex
\usepackage{booktabs}
```

---

### 3. Lined (全线条样式)
**描述**: 每行都有横线分隔

**特点**:
- 每行都有分隔线
- 清晰的行区分
- 适合数据密集的表格

**使用示例**:
```bash
python tools/data_to_latex.py -i data.csv -t table --style lined
```

**输出示例**:
```latex
\begin{table}[htbp]
\centering
\begin{tabular}{lll}
\hline
model & accuracy & f1_score \\
\hline
ResNet50 & 0.9523 & 0.9412 \\
\hline
VGG16 & 0.9234 & 0.9145 \\
\hline
\hline
\end{tabular}
\end{table}
```

---

### 4. Minimal (极简样式)
**描述**: 只有顶部和底部横线

**特点**:
- 极简设计
- 清爽的视觉效果
- 适合现代排版

**使用示例**:
```bash
python tools/data_to_latex.py -i data.csv -t table --style minimal
```

**输出示例**:
```latex
\begin{table}[htbp]
\centering
\begin{tabular}{lll}
\hline
model & accuracy & f1_score \\
\hline
ResNet50 & 0.9523 & 0.9412 \\
VGG16 & 0.9234 & 0.9145 \\
\hline
\end{tabular}
\end{table}
```

---

### 5. Fancy (美化样式)
**描述**: 使用 `booktabs` 和优化的行距

**特点**:
- 专业的线条样式
- 增加的行距（1.2倍）
- 更美观的排版
- 需要 `\usepackage{booktabs}` 和 `\usepackage{array}`

**使用示例**:
```bash
python tools/data_to_latex.py -i data.csv -t table --style fancy
```

**输出示例**:
```latex
\begin{table}[htbp]
\centering
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{lll}
\toprule
model & accuracy & f1_score \\
\midrule
ResNet50 & 0.9523 & 0.9412 \\
VGG16 & 0.9234 & 0.9145 \\
\bottomrule
\end{tabular}
\end{table}
```

**LaTeX前言**:
```latex
\usepackage{booktabs}
\usepackage{array}
```

---

## 列对齐方式

除了选择样式，还可以自定义列对齐方式：

### 对齐选项
- `l` - 左对齐（left）
- `c` - 居中对齐（center）
- `r` - 右对齐（right）

### 使用示例

```bash
# 第一列左对齐，第二列右对齐，第三列居中
python tools/data_to_latex.py -i data.csv -t table --column-align "lrc"

# 所有列居中
python tools/data_to_latex.py -i data.csv -t table --column-align "ccc"

# 组合使用：第一列左对齐，其余居中
python tools/data_to_latex.py -i data.csv -t table --column-align "lccc"
```

---

## 样式对比

| 样式 | 所需包 | 顶部线 | 表头线 | 行间线 | 底部线 | 行距 |
|------|--------|--------|--------|--------|--------|------|
| simple | 无 | `\hline` | `\hline` | 无 | `\hline` | 默认 |
| booktabs | booktabs | `\toprule` | `\midrule` | 无 | `\bottomrule` | 默认 |
| lined | 无 | `\hline` | `\hline` | `\hline` | `\hline` | 默认 |
| minimal | 无 | `\hline` | `\hline` | 无 | `\hline` | 默认 |
| fancy | booktabs, array | `\toprule` | `\midrule` | 无 | `\bottomrule` | 1.2x |

---

## 使用建议

### 学术论文
推荐使用 **booktabs** 或 **fancy** 样式：
```bash
python tools/data_to_latex.py -i results.csv -t table --style booktabs \
  --caption "实验结果" --label "tab:results"
```

### 技术报告
推荐使用 **simple** 或 **minimal** 样式：
```bash
python tools/data_to_latex.py -i data.csv -t table --style minimal
```

### 数据密集表格
推荐使用 **lined** 样式：
```bash
python tools/data_to_latex.py -i dense_data.csv -t table --style lined
```

---

## 完整示例

### 示例1: 学术论文表格
```bash
python tools/data_to_latex.py \
  -i experiment_results.csv \
  -t table \
  --style booktabs \
  --caption "Model Performance on Different Datasets" \
  --label "tab:model_performance" \
  --column-align "lcccc" \
  -o paper_table.tex
```

### 示例2: 简洁报告表格
```bash
python tools/data_to_latex.py \
  -i summary.csv \
  -t table \
  --style minimal \
  -c name value unit \
  --column-align "lrc"
```

### 示例3: 详细数据表格
```bash
python tools/data_to_latex.py \
  -i detailed_data.csv \
  -t longtable \
  --style fancy \
  --max-rows 100
```

---

## 查看所有样式

使用 `--list-styles` 参数查看所有可用样式：

```bash
python tools/data_to_latex.py --list-styles
```

输出：
```
================================================================================
                                  可用的LaTeX表格样式                                  
================================================================================

【simple】- 简单样式
  描述: 基础表格样式，使用\hline分隔
  需要的包: 无

【booktabs】- 专业样式
  描述: 使用booktabs包的专业样式
  需要的包: booktabs

...
```

---

## 常见问题

### Q: 如何知道需要哪些LaTeX包？
A: 工具会自动提示需要的包。使用需要特殊包的样式时，会显示：
```
📦 注意: 此样式需要以下LaTeX包: booktabs
   请在LaTeX文档中添加: \usepackage{booktabs}
```

### Q: 可以混合使用样式吗？
A: 不能直接混合，但你可以：
1. 生成不同样式的多个表格
2. 手动编辑生成的LaTeX代码

### Q: 如何添加自定义样式？
A: 在 `data_to_latex.py` 的 `TABLE_TEMPLATES` 字典中添加新样式定义。

### Q: booktabs样式的优势是什么？
A: booktabs是LaTeX表格排版的金标准，提供：
- 更专业的外观
- 更好的行距
- 符合出版标准
- 被大多数学术期刊接受

---

## 技巧和最佳实践

### 1. 选择合适的样式
- **学术论文**: booktabs或fancy
- **幻灯片**: minimal
- **技术文档**: simple
- **数据报告**: lined

### 2. 列对齐建议
- **文本列**: 左对齐 (`l`)
- **数值列**: 右对齐 (`r`)
- **标题/分类**: 居中 (`c`)

### 3. 包管理
使用需要特殊包的样式时，在LaTeX文档开头添加：
```latex
\usepackage{booktabs}  % 用于booktabs和fancy样式
\usepackage{array}     % 用于fancy样式
\usepackage{longtable} % 用于longtable类型
```

### 4. 组合使用
```bash
# 完整配置示例
python tools/data_to_latex.py \
  -i data.csv \
  -t table \
  --style booktabs \
  --column-align "lrcc" \
  --caption "My Table" \
  --label "tab:my" \
  -c col1 col2 col3 \
  --max-rows 50 \
  -o output.tex
```

---

## 更多资源

- **工具主文档**: `tools/data_to_latex_examples.md`
- **快速参考**: `tools/data_to_latex_quickref.md`
- **LaTeX示例**: `tools/latex_example.tex`
- **工具概览**: `tools/README.md`

