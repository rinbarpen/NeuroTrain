# data_to_latex.py 快速参考

## 基本语法
```bash
python tools/data_to_latex.py -i <输入文件> -t <格式类型> [选项]
```

## 常用命令速查

### 1. 标准表格 (Table)
```bash
# 基础表格
python tools/data_to_latex.py -i data.csv -t table

# 带标题和标签
python tools/data_to_latex.py -i data.csv -t table \
  --caption "实验结果" --label "tab:results"

# 保存到文件
python tools/data_to_latex.py -i data.csv -t table -o output.tex
```

### 2. 长表格 (Longtable) - 支持跨页
```bash
python tools/data_to_latex.py -i data.csv -t longtable
```

### 3. 无序列表 (Itemize)
```bash
# 默认格式
python tools/data_to_latex.py -i data.csv -t itemize

# 自定义模板
python tools/data_to_latex.py -i data.csv -t itemize \
  --template "{name}: {value} ({unit})"
```

### 4. 有序列表 (Enumerate)
```bash
python tools/data_to_latex.py -i data.csv -t enumerate
```

### 5. 描述列表 (Description)
```bash
# 自动使用第一列作为键
python tools/data_to_latex.py -i data.csv -t description

# 指定键列
python tools/data_to_latex.py -i data.csv -t description \
  --key-column name
```

## 常用选项

| 选项 | 说明 | 示例 |
|------|------|------|
| `-i, --input` | 输入文件 | `-i data.csv` |
| `-o, --output` | 输出文件 | `-o output.tex` |
| `-t, --type` | LaTeX格式 | `-t table` |
| `-c, --columns` | 选择列 | `-c col1 col2 col3` |
| `--caption` | 表格标题 | `--caption "Results"` |
| `--label` | 表格标签 | `--label "tab:res"` |
| `--style` | 表格样式 | `--style booktabs` |
| `--column-align` | 列对齐方式 | `--column-align "lrc"` |
| `--template` | 列表模板 | `--template "{x}: {y}"` |
| `--key-column` | 描述列表键列 | `--key-column name` |
| `--max-rows` | 最大行数 | `--max-rows 100` |
| `--show-info` | 显示信息 | `--show-info` |
| `--list-styles` | 列出所有样式 | `--list-styles` |

## 支持的文件格式
- CSV (`.csv`)
- Excel (`.xls`, `.xlsx`)  
- JSON (`.json`)
- Parquet (`.parquet`)
- TSV (`.tsv`, `.txt`)

## 5种LaTeX格式对比

| 格式 | 用途 | 适用场景 |
|------|------|----------|
| `table` | 标准表格 | 数据较少，不跨页 |
| `longtable` | 长表格 | 数据较多，需要跨页 |
| `itemize` | 无序列表 | 项目列表，无特定顺序 |
| `enumerate` | 有序列表 | 有序项目，需要编号 |
| `description` | 描述列表 | 键值对，术语定义 |

## 5种表格样式对比

| 样式 | 描述 | 所需包 | 适用场景 |
|------|------|--------|----------|
| `simple` | 简单样式（默认） | 无 | 快速生成，通用 |
| `booktabs` | 专业样式 | booktabs | 学术论文 |
| `lined` | 全线条样式 | 无 | 数据密集表格 |
| `minimal` | 极简样式 | 无 | 现代排版 |
| `fancy` | 美化样式 | booktabs, array | 高质量出版 |

## 实用技巧

### 1. 查看所有样式
```bash
# 列出所有可用的表格样式
python tools/data_to_latex.py --list-styles
```

### 2. 数据预览
```bash
# 先查看数据结构
python tools/data_to_latex.py -i data.csv --show-info
```

### 3. 选择样式
```bash
# 使用专业的booktabs样式
python tools/data_to_latex.py -i data.csv -t table --style booktabs

# 使用全线条样式
python tools/data_to_latex.py -i data.csv -t table --style lined
```

### 4. 自定义列对齐
```bash
# 第一列左对齐，第二列右对齐，其余居中
python tools/data_to_latex.py -i data.csv -t table --column-align "lrcc"
```

### 5. 选择列
```bash
# 只转换需要的列
python tools/data_to_latex.py -i data.csv -t table -c name score rank
```

### 6. 限制行数
```bash
# 避免表格过长
python tools/data_to_latex.py -i data.csv -t table --max-rows 50
```

### 7. 自定义模板
```bash
# itemize使用模板
python tools/data_to_latex.py -i models.csv -t itemize \
  --template "{model}: Acc={accuracy:.2f}, Params={params}"
```

### 8. 批量转换
```bash
# shell脚本批量处理
for file in data/*.csv; do
  python tools/data_to_latex.py -i "$file" -t table \
    --style booktabs \
    -o "output/$(basename $file .csv).tex"
done
```

## LaTeX文档中使用

### 1. 表格需要的包
```latex
\usepackage{longtable}  % 如果使用longtable
\usepackage{booktabs}   % 美化表格（可选）
```

### 2. 引用表格
```latex
\begin{document}
  % 方法1: 直接粘贴
  \begin{table}[htbp]
    % ... 生成的代码 ...
  \end{table}
  
  % 方法2: 使用\input
  \input{output/table.tex}
  
  % 引用
  如表 \ref{tab:results} 所示...
\end{document}
```

### 3. 编译
```bash
# 标准编译
pdflatex document.tex

# 中文文档
xelatex document.tex
```

## 常见问题

### Q: 中文显示乱码？
**A:** 使用XeLaTeX编译，并添加中文支持：
```latex
\usepackage{xeCJK}
\setCJKmainfont{SimSun}  % 或其他中文字体
```

### Q: 表格列太宽？
**A:** 使用 `-c` 选项只选择需要的列

### Q: 特殊字符显示错误？
**A:** 工具已自动转义常见LaTeX特殊字符（`&`, `%`, `$`, `#`, `_`, `{`, `}`等）

### Q: 需要更复杂的表格样式？
**A:** 生成基础代码后，可以手动添加：
- `\toprule`, `\midrule`, `\bottomrule`（需要booktabs包）
- 列对齐：`{lrccc}` （左、右、居中）
- 列宽：`{p{3cm}lc}`

## 示例数据

创建测试数据：
```csv
name,score,rank
Alice,95,1
Bob,88,2
Charlie,92,3
```

转换并查看：
```bash
python tools/data_to_latex.py -i test.csv -t table --caption "Student Scores"
```

## 更多信息
- 详细文档: `tools/data_to_latex_examples.md`
- LaTeX示例: `tools/latex_example.tex`
- 工具列表: `tools/README.md`

