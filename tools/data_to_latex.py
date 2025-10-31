"""
数据文件转LaTeX列表工具

这个工具允许你：
1. 将CSV、Excel、JSON、Parquet等数据文件转换为LaTeX格式
2. 支持多种LaTeX输出格式：
   - table: LaTeX表格
   - itemize: 无序列表
   - enumerate: 有序列表
   - description: 描述列表
   - longtable: 长表格（支持跨页）
3. 自定义列选择和格式化选项
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
from typing import Optional, List

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


# LaTeX表格模板定义
TABLE_TEMPLATES = {
    'simple': {
        'name': '简单样式',
        'description': '基础表格样式，使用\\hline分隔',
        'packages': [],
        'column_spec': 'l',  # 默认左对齐
        'use_toprule': False,
        'use_midrule': False,
        'use_bottomrule': False,
        'header_separator': r'\hline',
        'row_separator': '',
        'end_separator': r'\hline',
    },
    'booktabs': {
        'name': '专业样式',
        'description': '使用booktabs包的专业样式',
        'packages': ['booktabs'],
        'column_spec': 'l',
        'use_toprule': True,
        'use_midrule': True,
        'use_bottomrule': True,
        'header_separator': r'\midrule',
        'row_separator': '',
        'end_separator': r'\bottomrule',
    },
    'lined': {
        'name': '全线条样式',
        'description': '每行都有横线分隔',
        'packages': [],
        'column_spec': 'l',
        'use_toprule': False,
        'use_midrule': False,
        'use_bottomrule': False,
        'header_separator': r'\hline',
        'row_separator': r'\hline',
        'end_separator': r'\hline',
    },
    'minimal': {
        'name': '极简样式',
        'description': '只有顶部和底部横线',
        'packages': [],
        'column_spec': 'l',
        'use_toprule': False,
        'use_midrule': False,
        'use_bottomrule': False,
        'header_separator': r'\hline',
        'row_separator': '',
        'end_separator': r'\hline',
    },
    'fancy': {
        'name': '美化样式',
        'description': '使用booktabs和列边距优化',
        'packages': ['booktabs', 'array'],
        'column_spec': 'l',
        'use_toprule': True,
        'use_midrule': True,
        'use_bottomrule': True,
        'header_separator': r'\midrule',
        'row_separator': '',
        'end_separator': r'\bottomrule',
        'extra_preamble': r'\renewcommand{\arraystretch}{1.2}',  # 增加行距
    },
}


class DataToLatexConverter:
    """数据文件到LaTeX的转换器"""
    
    def __init__(self, 
                 input_file: str, 
                 output_file: Optional[str] = None,
                 latex_type: str = 'table',
                 columns: Optional[List[str]] = None,
                 caption: Optional[str] = None,
                 label: Optional[str] = None,
                 max_rows: Optional[int] = None,
                 table_style: str = 'simple',
                 column_align: Optional[str] = None,
                 highlight_best: bool = False,
                 highlight_second: bool = False,
                 metric_columns: Optional[List[str]] = None,
                 higher_is_better: Optional[List[bool]] = None,
                 our_model: Optional[str] = None,
                 group_column: Optional[str] = None):
        """
        初始化转换器
        
        Args:
            input_file: 输入数据文件路径
            output_file: 输出LaTeX文件路径（可选，默认打印到控制台）
            latex_type: LaTeX格式类型 (table, itemize, enumerate, description, longtable)
            columns: 要包含的列（可选，默认所有列）
            caption: 表格标题（可选）
            label: 表格标签（可选）
            max_rows: 最大行数限制（可选）
            table_style: 表格样式模板 (simple, booktabs, lined, minimal, fancy)
            column_align: 列对齐方式（可选，如 'lrc' 表示左、右、居中，默认全部左对齐）
            highlight_best: 是否高亮最佳值（粗体）
            highlight_second: 是否高亮次佳值（斜体）
            metric_columns: 要进行高亮的指标列名列表
            higher_is_better: 对应每个指标列，True表示越高越好，False表示越低越好
            our_model: 我们的模型名称，会用下划线标注
            group_column: 分组列（用于多数据集/多任务场景）
        """
        self.input_file = Path(input_file)
        self.output_file = Path(output_file) if output_file else None
        self.latex_type = latex_type
        self.columns = columns
        self.caption = caption
        self.label = label
        self.max_rows = max_rows
        self.table_style = table_style
        self.column_align = column_align
        self.highlight_best = highlight_best
        self.highlight_second = highlight_second
        self.metric_columns = metric_columns or []
        self.higher_is_better = higher_is_better or []
        self.our_model = our_model
        self.group_column = group_column
        self.df = None
        
        # 验证表格样式
        if self.table_style not in TABLE_TEMPLATES:
            raise ValueError(f"不支持的表格样式: {self.table_style}。"
                           f"可用样式: {', '.join(TABLE_TEMPLATES.keys())}")
        
        # 验证指标列和higher_is_better长度匹配
        if self.metric_columns and self.higher_is_better:
            if len(self.metric_columns) != len(self.higher_is_better):
                raise ValueError(f"metric_columns和higher_is_better长度不匹配: "
                               f"{len(self.metric_columns)} vs {len(self.higher_is_better)}")
        
    def load_data(self):
        """加载数据文件"""
        suffix = self.input_file.suffix.lower()
        
        print(f"正在加载文件: {self.input_file}")
        
        try:
            if suffix == '.csv':
                self.df = pd.read_csv(self.input_file)
            elif suffix in ['.xls', '.xlsx']:
                self.df = pd.read_excel(self.input_file)
            elif suffix == '.json':
                self.df = pd.read_json(self.input_file)
            elif suffix == '.parquet':
                self.df = pd.read_parquet(self.input_file)
            elif suffix in ['.tsv', '.txt']:
                self.df = pd.read_csv(self.input_file, sep='\t')
            else:
                raise ValueError(f"不支持的文件格式: {suffix}")
            
            print(f"✓ 成功加载 {len(self.df)} 行 × {len(self.df.columns)} 列数据")
            
            # 如果指定了列，只保留这些列
            if self.columns:
                missing_cols = set(self.columns) - set(self.df.columns)
                if missing_cols:
                    print(f"警告: 以下列不存在: {missing_cols}")
                    self.columns = [c for c in self.columns if c in self.df.columns]
                self.df = self.df[self.columns]
                print(f"✓ 已选择列: {', '.join(self.columns)}")
            
            # 如果指定了最大行数，截取
            if self.max_rows and len(self.df) > self.max_rows:
                print(f"⚠ 数据行数超过 {self.max_rows}，将只转换前 {self.max_rows} 行")
                self.df = self.df.head(self.max_rows)
                
        except Exception as e:
            raise RuntimeError(f"加载文件失败: {e}")
    
    def escape_latex(self, text: str) -> str:
        """转义LaTeX特殊字符"""
        if pd.isna(text):
            return ''
        
        text = str(text)
        
        # LaTeX特殊字符转义
        replacements = {
            '\\': r'\textbackslash{}',
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '^': r'\textasciicircum{}',
        }
        
        for char, escaped in replacements.items():
            text = text.replace(char, escaped)
        
        return text
    
    def find_best_and_second(self, values: List[float], higher_is_better: bool = True):
        """
        找出最佳和次佳值的索引
        
        Args:
            values: 数值列表
            higher_is_better: True表示越高越好，False表示越低越好
            
        Returns:
            (best_idx, second_idx): 最佳和次佳值的索引
        """
        # 过滤掉NaN值
        valid_pairs = [(i, v) for i, v in enumerate(values) if not pd.isna(v)]
        if len(valid_pairs) < 2:
            return (valid_pairs[0][0] if valid_pairs else None, None)
        
        # 按值排序
        sorted_pairs = sorted(valid_pairs, key=lambda x: x[1], reverse=higher_is_better)
        
        return (sorted_pairs[0][0], sorted_pairs[1][0])
    
    def format_value(self, value, is_best: bool = False, is_second: bool = False) -> str:
        """
        格式化数值，添加粗体或斜体
        
        Args:
            value: 原始值
            is_best: 是否是最佳值
            is_second: 是否是次佳值
            
        Returns:
            格式化后的LaTeX字符串
        """
        value_str = self.escape_latex(str(value))
        
        if is_best and self.highlight_best:
            return r'\textbf{' + value_str + '}'
        elif is_second and self.highlight_second:
            return r'\textit{' + value_str + '}'
        else:
            return value_str
    
    def format_model_name(self, name: str) -> str:
        """
        格式化模型名称，如果是我们的模型则添加下划线
        
        Args:
            name: 模型名称
            
        Returns:
            格式化后的LaTeX字符串
        """
        escaped_name = self.escape_latex(str(name))
        
        if self.our_model and str(name) == self.our_model:
            return r'\underline{' + escaped_name + '}'
        else:
            return escaped_name
    
    def to_table(self, long_table: bool = False) -> str:
        """
        转换为LaTeX表格格式
        
        Args:
            long_table: 是否使用longtable环境（支持跨页）
        """
        lines = []
        
        # 获取模板配置
        template = TABLE_TEMPLATES[self.table_style]
        
        # 确定列对齐方式
        if self.column_align:
            col_spec = self.column_align
        else:
            col_spec = template['column_spec'] * len(self.df.columns)
        
        # 表格环境
        if long_table:
            lines.append(r'\begin{longtable}{' + col_spec + '}')
        else:
            lines.append(r'\begin{table}[htbp]')
            lines.append(r'\centering')
            if self.caption:
                lines.append(f'\\caption{{{self.escape_latex(self.caption)}}}')
            if self.label:
                lines.append(f'\\label{{{self.label}}}')
            
            # 添加额外的前言（如果有）
            if 'extra_preamble' in template:
                lines.append(template['extra_preamble'])
            
            lines.append(r'\begin{tabular}{' + col_spec + '}')
            
            # 顶部线条
            if template['use_toprule']:
                lines.append(r'\toprule')
            elif not long_table:
                lines.append(r'\hline')
        
        # 表头
        header = ' & '.join([self.escape_latex(col) for col in self.df.columns])
        lines.append(header + r' \\')
        lines.append(template['header_separator'])
        
        if long_table:
            lines.append(r'\endfirsthead')
            if template['use_toprule']:
                lines.append(r'\toprule')
            else:
                lines.append(r'\hline')
            lines.append(header + r' \\')
            lines.append(template['header_separator'])
            lines.append(r'\endhead')
        
        # 计算每个指标列的最佳和次佳值
        best_second_dict = {}
        if self.metric_columns and (self.highlight_best or self.highlight_second):
            if self.group_column and self.group_column in self.df.columns:
                # 分组模式：每个组内分别计算最佳/次佳
                for group in self.df[self.group_column].unique():
                    group_df = self.df[self.df[self.group_column] == group]
                    for col_idx, col in enumerate(self.metric_columns):
                        if col in self.df.columns:
                            higher = self.higher_is_better[col_idx] if col_idx < len(self.higher_is_better) else True
                            try:
                                values = pd.to_numeric(group_df[col], errors='coerce').tolist()
                                best_idx, second_idx = self.find_best_and_second(values, higher)
                                
                                # 将组内索引转换为全局索引
                                if best_idx is not None:
                                    global_best_idx = group_df.index[best_idx]
                                    best_second_dict[(global_best_idx, col)] = ('best', values[best_idx])
                                if second_idx is not None:
                                    global_second_idx = group_df.index[second_idx]
                                    best_second_dict[(global_second_idx, col)] = ('second', values[second_idx])
                            except Exception:
                                pass
            else:
                # 非分组模式：全局计算最佳/次佳
                for col_idx, col in enumerate(self.metric_columns):
                    if col in self.df.columns:
                        higher = self.higher_is_better[col_idx] if col_idx < len(self.higher_is_better) else True
                        try:
                            values = pd.to_numeric(self.df[col], errors='coerce').tolist()
                            best_idx, second_idx = self.find_best_and_second(values, higher)
                            
                            if best_idx is not None:
                                best_second_dict[(best_idx, col)] = ('best', values[best_idx])
                            if second_idx is not None:
                                best_second_dict[(second_idx, col)] = ('second', values[second_idx])
                        except Exception:
                            pass
        
        # 数据行
        for idx, row in self.df.iterrows():
            formatted_values = []
            for col in self.df.columns:
                value = row[col]
                
                # 检查是否是模型名称列（通常是第一列或名为'model'的列）
                if col == self.df.columns[0] or col.lower() in ['model', 'method', 'name']:
                    formatted_val = self.format_model_name(value)
                # 检查是否是指标列且需要高亮
                elif col in self.metric_columns:
                    key = (idx, col)
                    if key in best_second_dict:
                        status, _ = best_second_dict[key]
                        formatted_val = self.format_value(value, 
                                                         is_best=(status == 'best'),
                                                         is_second=(status == 'second'))
                    else:
                        formatted_val = self.escape_latex(value)
                else:
                    formatted_val = self.escape_latex(value)
                
                formatted_values.append(formatted_val)
            
            row_data = ' & '.join(formatted_values)
            lines.append(row_data + r' \\')
            
            # 添加行分隔符（如果有）
            if template['row_separator']:
                lines.append(template['row_separator'])
            
            # 如果有分组列，在组之间添加分隔线
            if self.group_column and self.group_column in self.df.columns:
                if idx < len(self.df) - 1:
                    current_group = row[self.group_column]
                    next_group = self.df.iloc[idx + 1][self.group_column]
                    if current_group != next_group:
                        lines.append(r'\midrule' if template['use_midrule'] else r'\hline')
        
        # 结束线条
        if template['use_bottomrule']:
            lines.append(r'\bottomrule')
        else:
            lines.append(template['end_separator'])
        
        if long_table:
            lines.append(r'\end{longtable}')
        else:
            lines.append(r'\end{tabular}')
            lines.append(r'\end{table}')
        
        return '\n'.join(lines)
    
    def to_itemize(self, template: Optional[str] = None) -> str:
        """
        转换为无序列表格式
        
        Args:
            template: 项目模板字符串，可以使用{列名}占位符
                     例如: "{name}: {value}"
        """
        lines = [r'\begin{itemize}']
        
        for _, row in self.df.iterrows():
            if template:
                # 使用模板
                item_text = template
                for col in self.df.columns:
                    item_text = item_text.replace(f'{{{col}}}', str(row[col]))
                item_text = self.escape_latex(item_text)
            else:
                # 默认格式：逗号分隔的列值
                item_text = ', '.join([f'{col}: {self.escape_latex(val)}' 
                                      for col, val in row.items()])
            
            lines.append(f'  \\item {item_text}')
        
        lines.append(r'\end{itemize}')
        return '\n'.join(lines)
    
    def to_enumerate(self, template: Optional[str] = None) -> str:
        """
        转换为有序列表格式
        
        Args:
            template: 项目模板字符串，可以使用{列名}占位符
        """
        lines = [r'\begin{enumerate}']
        
        for _, row in self.df.iterrows():
            if template:
                item_text = template
                for col in self.df.columns:
                    item_text = item_text.replace(f'{{{col}}}', str(row[col]))
                item_text = self.escape_latex(item_text)
            else:
                item_text = ', '.join([f'{col}: {self.escape_latex(val)}' 
                                      for col, val in row.items()])
            
            lines.append(f'  \\item {item_text}')
        
        lines.append(r'\end{enumerate}')
        return '\n'.join(lines)
    
    def to_description(self, key_column: Optional[str] = None, 
                       value_columns: Optional[List[str]] = None) -> str:
        """
        转换为描述列表格式
        
        Args:
            key_column: 作为描述项标题的列（如果为None，使用第一列）
            value_columns: 作为描述内容的列（如果为None，使用除key_column外的所有列）
        """
        lines = [r'\begin{description}']
        
        # 确定键列和值列
        if key_column is None:
            key_column = self.df.columns[0]
        
        if value_columns is None:
            value_columns = [c for c in self.df.columns if c != key_column]
        
        for _, row in self.df.iterrows():
            key = self.escape_latex(row[key_column])
            value_parts = [f'{col}: {self.escape_latex(row[col])}' 
                          for col in value_columns]
            value = ', '.join(value_parts)
            
            lines.append(f'  \\item[{key}] {value}')
        
        lines.append(r'\end{description}')
        return '\n'.join(lines)
    
    def convert(self, template: Optional[str] = None, 
                key_column: Optional[str] = None,
                value_columns: Optional[List[str]] = None) -> str:
        """
        执行转换
        
        Args:
            template: 用于itemize/enumerate的模板
            key_column: 用于description的键列
            value_columns: 用于description的值列
        """
        if self.df is None:
            raise RuntimeError("请先调用 load_data() 加载数据")
        
        print(f"\n正在生成 LaTeX {self.latex_type} 格式...")
        
        if self.latex_type == 'table':
            latex_code = self.to_table(long_table=False)
        elif self.latex_type == 'longtable':
            latex_code = self.to_table(long_table=True)
        elif self.latex_type == 'itemize':
            latex_code = self.to_itemize(template=template)
        elif self.latex_type == 'enumerate':
            latex_code = self.to_enumerate(template=template)
        elif self.latex_type == 'description':
            latex_code = self.to_description(key_column=key_column, 
                                            value_columns=value_columns)
        else:
            raise ValueError(f"不支持的LaTeX类型: {self.latex_type}")
        
        return latex_code
    
    def save(self, latex_code: str):
        """保存LaTeX代码到文件或打印到控制台"""
        if self.output_file:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            self.output_file.write_text(latex_code, encoding='utf-8')
            print(f"\n✓ LaTeX代码已保存到: {self.output_file}")
            
            # 如果使用了需要特殊包的模板，提示用户
            template = TABLE_TEMPLATES[self.table_style]
            if template['packages']:
                print(f"\n📦 注意: 此样式需要以下LaTeX包: {', '.join(template['packages'])}")
                print(f"   请在LaTeX文档中添加: \\usepackage{{{', '.join(template['packages'])}}}")
        else:
            print("\n" + "=" * 80)
            print("LaTeX代码:")
            print("=" * 80)
            print(latex_code)
            print("=" * 80)
            
            # 提示需要的包
            template = TABLE_TEMPLATES[self.table_style]
            if template['packages']:
                print(f"\n📦 注意: 此样式需要以下LaTeX包: {', '.join(template['packages'])}")
                print(f"   请在LaTeX文档中添加: \\usepackage{{{', '.join(template['packages'])}}}")


def list_table_styles():
    """列出所有可用的表格样式"""
    print("\n" + "=" * 80)
    print("可用的LaTeX表格样式".center(80))
    print("=" * 80)
    
    for style_name, style_info in TABLE_TEMPLATES.items():
        print(f"\n【{style_name}】- {style_info['name']}")
        print(f"  描述: {style_info['description']}")
        if style_info['packages']:
            print(f"  需要的包: {', '.join(style_info['packages'])}")
        else:
            print("  需要的包: 无")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='数据文件转LaTeX列表工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 将CSV文件转换为LaTeX表格
  python tools/data_to_latex.py -i data.csv -t table
  
  # 将Excel文件转换为无序列表，使用自定义模板
  python tools/data_to_latex.py -i data.xlsx -t itemize --template "{name} ({value})"
  
  # 将JSON文件转换为描述列表，指定键列
  python tools/data_to_latex.py -i data.json -t description --key-column name
  
  # 转换并保存到文件，只选择特定列
  python tools/data_to_latex.py -i data.csv -o output.tex -t table -c col1 col2 col3
  
  # 转换长表格（支持跨页），限制行数
  python tools/data_to_latex.py -i data.csv -t longtable --max-rows 100
  
  # 添加标题和标签
  python tools/data_to_latex.py -i data.csv -t table --caption "数据统计" --label "tab:stats"
  
  # 使用booktabs专业样式
  python tools/data_to_latex.py -i data.csv -t table --style booktabs
  
  # 自定义列对齐（左、居中、右）
  python tools/data_to_latex.py -i data.csv -t table --column-align "lccr"
  
  # 查看所有可用样式
  python tools/data_to_latex.py --list-styles
  
  # 高亮最佳和次佳值
  python tools/data_to_latex.py -i results.csv -t table --style booktabs \
    --highlight-best --highlight-second \
    --metric-columns accuracy f1_score \
    --higher-is-better True True
  
  # 标注我们的模型并高亮结果
  python tools/data_to_latex.py -i results.csv -t table --style booktabs \
    --our-model "OurModel" --highlight-best \
    --metric-columns accuracy loss \
    --higher-is-better True False
  
  # 多数据集比较（分组）
  python tools/data_to_latex.py -i multi_dataset.csv -t table --style booktabs \
    --group-column dataset --highlight-best \
    --metric-columns accuracy f1_score

支持的数据格式:
  - CSV (.csv)
  - Excel (.xls, .xlsx)
  - JSON (.json)
  - Parquet (.parquet)
  - TSV (.tsv, .txt)

支持的LaTeX格式:
  - table:       标准表格 (tabular)
  - longtable:   长表格（支持跨页）
  - itemize:     无序列表
  - enumerate:   有序列表
  - description: 描述列表

支持的表格样式:
  - simple:      简单样式（默认）
  - booktabs:    专业样式（需要booktabs包）
  - lined:       全线条样式
  - minimal:     极简样式
  - fancy:       美化样式（需要booktabs和array包）
        """
    )
    
    parser.add_argument('-i', '--input', type=str,
                       help='输入数据文件路径')
    parser.add_argument('-o', '--output', type=str,
                       help='输出LaTeX文件路径（可选，默认打印到控制台）')
    parser.add_argument('-t', '--type', type=str, 
                       choices=['table', 'itemize', 'enumerate', 'description', 'longtable'],
                       default='table',
                       help='LaTeX格式类型（默认: table）')
    parser.add_argument('-c', '--columns', nargs='+', type=str,
                       help='要包含的列名（可选，默认所有列）')
    parser.add_argument('--caption', type=str,
                       help='表格标题（仅用于table/longtable）')
    parser.add_argument('--label', type=str,
                       help='表格标签（仅用于table/longtable）')
    parser.add_argument('--style', type=str,
                       choices=list(TABLE_TEMPLATES.keys()),
                       default='simple',
                       help='表格样式模板（默认: simple）')
    parser.add_argument('--column-align', type=str,
                       help='列对齐方式，如"lrc"表示左、右、居中（可选）')
    parser.add_argument('--template', type=str,
                       help='项目模板字符串（用于itemize/enumerate），例如: "{name}: {value}"')
    parser.add_argument('--key-column', type=str,
                       help='描述列表的键列（用于description）')
    parser.add_argument('--value-columns', nargs='+', type=str,
                       help='描述列表的值列（用于description）')
    parser.add_argument('--max-rows', type=int,
                       help='最大行数限制（可选）')
    parser.add_argument('--show-info', action='store_true',
                       help='显示数据文件信息')
    parser.add_argument('--list-styles', action='store_true',
                       help='列出所有可用的表格样式')
    
    # 高级功能参数
    parser.add_argument('--highlight-best', action='store_true',
                       help='高亮最佳值（粗体）')
    parser.add_argument('--highlight-second', action='store_true',
                       help='高亮次佳值（斜体）')
    parser.add_argument('--metric-columns', nargs='+', type=str,
                       help='要进行高亮的指标列名列表')
    parser.add_argument('--higher-is-better', nargs='+', type=lambda x: x.lower() == 'true',
                       help='对应每个指标列，True表示越高越好，False表示越低越好（如: True False True）')
    parser.add_argument('--our-model', type=str,
                       help='我们的模型名称，会用下划线标注')
    parser.add_argument('--group-column', type=str,
                       help='分组列（用于多数据集/多任务场景）')
    
    args = parser.parse_args()
    
    # 如果要列出样式
    if args.list_styles:
        list_table_styles()
        return
    
    # 如果没有输入文件，显示帮助
    if not args.input:
        parser.print_help()
        return
    
    try:
        # 创建转换器
        converter = DataToLatexConverter(
            input_file=args.input,
            output_file=args.output,
            latex_type=args.type,
            columns=args.columns,
            caption=args.caption,
            label=args.label,
            max_rows=args.max_rows,
            table_style=args.style,
            column_align=args.column_align,
            highlight_best=args.highlight_best,
            highlight_second=args.highlight_second,
            metric_columns=args.metric_columns,
            higher_is_better=args.higher_is_better,
            our_model=args.our_model,
            group_column=args.group_column
        )
        
        # 加载数据
        converter.load_data()
        
        # 如果只是显示信息
        if args.show_info:
            print("\n数据信息:")
            print(f"  行数: {len(converter.df)}")
            print(f"  列数: {len(converter.df.columns)}")
            print(f"  列名: {', '.join(converter.df.columns)}")
            print(f"  当前表格样式: {args.style}")
            print("\n前5行预览:")
            print(converter.df.head())
            return
        
        # 执行转换
        latex_code = converter.convert(
            template=args.template,
            key_column=args.key_column,
            value_columns=args.value_columns
        )
        
        # 保存或打印
        converter.save(latex_code)
        
        print("\n✓ 转换完成！")
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

