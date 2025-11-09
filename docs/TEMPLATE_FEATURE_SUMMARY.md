# LaTeX模板功能实现总结

## ✨ 新增功能

已成功为 `data_to_latex.py` 工具添加了LaTeX表格模板选择功能。

## 🎯 实现内容

### 1. 5种预设表格样式

| 样式 | 描述 | 所需包 | 特点 |
|------|------|--------|------|
| **simple** | 简单样式（默认） | 无 | 基础\hline分隔 |
| **booktabs** | 专业样式 | booktabs | \toprule, \midrule, \bottomrule |
| **lined** | 全线条样式 | 无 | 每行都有\hline |
| **minimal** | 极简样式 | 无 | 只有顶部和底部线条 |
| **fancy** | 美化样式 | booktabs, array | 专业线条 + 增加行距(1.2x) |

### 2. 新增命令行参数

- `--style <样式名>`: 选择表格样式（默认: simple）
- `--column-align <对齐>`: 自定义列对齐方式（如 "lrc"）
- `--list-styles`: 列出所有可用样式

### 3. 核心实现

#### TABLE_TEMPLATES 字典
定义了所有样式的配置：
```python
TABLE_TEMPLATES = {
    'simple': {...},
    'booktabs': {...},
    'lined': {...},
    'minimal': {...},
    'fancy': {...},
}
```

#### 修改的方法
- `__init__()`: 添加 `table_style` 和 `column_align` 参数
- `to_table()`: 使用模板配置生成表格
- `save()`: 自动提示需要的LaTeX包
- 新增 `list_table_styles()`: 列出所有样式

## 📊 使用示例

### 基本用法
```bash
# 查看所有样式
python tools/data_to_latex.py --list-styles

# 使用booktabs专业样式
python tools/data_to_latex.py -i data.csv -t table --style booktabs

# 使用fancy样式 + 自定义列对齐
python tools/data_to_latex.py -i data.csv -t table --style fancy --column-align "lrcc"
```

### 输出对比

#### Simple样式:
```latex
\begin{table}[htbp]
\centering
\begin{tabular}{lll}
\hline
model & accuracy & f1_score \\
\hline
ResNet50 & 0.9523 & 0.9412 \\
\hline
\end{tabular}
\end{table}
```

#### Booktabs样式:
```latex
\begin{table}[htbp]
\centering
\begin{tabular}{lll}
\toprule
model & accuracy & f1_score \\
\midrule
ResNet50 & 0.9523 & 0.9412 \\
\bottomrule
\end{tabular}
\end{table}
```

#### Fancy样式:
```latex
\begin{table}[htbp]
\centering
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{lll}
\toprule
model & accuracy & f1_score \\
\midrule
ResNet50 & 0.9523 & 0.9412 \\
\bottomrule
\end{tabular}
\end{table}
```

## ✅ 测试结果

所有样式已通过测试：
- ✅ simple - 基础样式工作正常
- ✅ booktabs - 专业样式工作正常，自动提示需要booktabs包
- ✅ lined - 全线条样式工作正常
- ✅ minimal - 极简样式工作正常
- ✅ fancy - 美化样式工作正常，自动提示需要booktabs和array包
- ✅ --list-styles - 正确列出所有样式
- ✅ --column-align - 自定义列对齐正常工作

## 📚 更新的文档

1. **TABLE_STYLES_GUIDE.md** (新建)
   - 完整的样式指南
   - 每种样式的详细说明和示例
   - 使用建议和最佳实践

2. **data_to_latex_quickref.md** (更新)
   - 添加 `--style` 和 `--column-align` 参数
   - 添加样式对比表格
   - 新增样式选择技巧

3. **data_to_latex.py** (更新)
   - 添加模板系统
   - 更新help文档
   - 添加包依赖提示

4. **test_all_styles.sh** (新建)
   - 测试所有样式的脚本
   - 生成样式对比LaTeX文档

## 🎨 特色功能

### 1. 智能包提示
使用需要特殊包的样式时，自动提示：
```
📦 注意: 此样式需要以下LaTeX包: booktabs
   请在LaTeX文档中添加: \usepackage{booktabs}
```

### 2. 灵活的列对齐
支持自定义每列的对齐方式：
- `l` - 左对齐
- `c` - 居中
- `r` - 右对齐

示例: `--column-align "lrcc"` 表示第1列左对齐，第2列右对齐，第3、4列居中

### 3. 样式查看
使用 `--list-styles` 快速查看所有可用样式及其说明

## 🔧 技术细节

### 模板配置结构
```python
{
    'name': '样式名称',
    'description': '样式描述',
    'packages': ['需要的包1', '需要的包2'],  # 可以为空列表
    'column_spec': 'l',  # 默认列对齐
    'use_toprule': False,  # 是否使用\toprule
    'use_midrule': False,  # 是否使用\midrule
    'use_bottomrule': False,  # 是否使用\bottomrule
    'header_separator': r'\hline',  # 表头分隔符
    'row_separator': '',  # 行分隔符（可选）
    'end_separator': r'\hline',  # 结束分隔符
    'extra_preamble': r'\renewcommand{\arraystretch}{1.2}',  # 额外的前言（可选）
}
```

### 代码改动统计
- 新增代码：约150行
- 修改方法：4个
- 新增函数：1个
- 新增参数：2个

## 📖 使用建议

### 学术论文
推荐 **booktabs** 或 **fancy**:
```bash
python tools/data_to_latex.py -i results.csv -t table --style booktabs \
  --caption "Experimental Results" --label "tab:results"
```

### 技术报告  
推荐 **simple** 或 **minimal**:
```bash
python tools/data_to_latex.py -i data.csv -t table --style minimal
```

### 数据密集表格
推荐 **lined**:
```bash
python tools/data_to_latex.py -i data.csv -t table --style lined
```

## 🚀 后续可能的扩展

1. 添加更多预设样式（如colorful、compact等）
2. 支持自定义样式配置文件
3. 支持表格宽度设置（tabularx）
4. 支持合并单元格
5. 支持条件格式化（如数值高亮）

## 📝 文件清单

### 新建文件
- `tools/TABLE_STYLES_GUIDE.md` - 样式详细指南
- `tools/test_all_styles.sh` - 样式测试脚本
- `tools/TEMPLATE_FEATURE_SUMMARY.md` - 本文档

### 更新文件
- `tools/data_to_latex.py` - 主工具（添加模板支持）
- `tools/data_to_latex_quickref.md` - 快速参考（添加样式说明）

## ✨ 总结

成功实现了LaTeX表格模板选择功能：
- ✅ 5种预设样式
- ✅ 自定义列对齐
- ✅ 智能包提示
- ✅ 完整文档
- ✅ 测试验证

该功能大大增强了工具的灵活性，使用户可以根据不同场景选择合适的表格样式，满足从快速原型到高质量出版的各种需求。

