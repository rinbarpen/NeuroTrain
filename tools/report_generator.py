"""
综合报告生成器工具

功能：
1. 收集训练/推理数据（从JSON、CSV等文件）
2. 生成统计图表（训练曲线、指标对比等）
3. 定位和强调生成的图像
4. 导出为LaTeX格式（用于论文撰写）
5. 支持ipynb使用
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ImageInfo:
    """图像信息"""
    path: Path
    caption: str = ""
    label: str = ""
    width: Optional[float] = None  # LaTeX中的宽度（如0.8\textwidth）
    highlight_region: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2) 用于强调的区域


@dataclass
class TableInfo:
    """表格信息"""
    data: pd.DataFrame
    caption: str = ""
    label: str = ""
    style: str = "booktabs"
    highlight_best: bool = False
    highlight_second: bool = False
    metric_columns: List[str] = field(default_factory=list)
    higher_is_better: List[bool] = field(default_factory=list)


class DataCollector:
    """数据收集器"""
    
    def __init__(self, base_dir: Union[str, Path]):
        self.base_dir = Path(base_dir)
        self.data_cache: Dict[str, Any] = {}
    
    def collect_from_json(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """从JSON文件收集数据"""
        filepath = Path(filepath)
        if not filepath.is_absolute():
            filepath = self.base_dir / filepath
        
        if str(filepath) in self.data_cache:
            return self.data_cache[str(filepath)]
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.data_cache[str(filepath)] = data
        return data
    
    def collect_from_csv(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """从CSV文件收集数据"""
        filepath = Path(filepath)
        if not filepath.is_absolute():
            filepath = self.base_dir / filepath
        
        if str(filepath) in self.data_cache:
            return self.data_cache[str(filepath)]
        
        df = pd.read_csv(filepath)
        self.data_cache[str(filepath)] = df
        return df
    
    def collect_from_parquet(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """从Parquet文件收集数据"""
        filepath = Path(filepath)
        if not filepath.is_absolute():
            filepath = self.base_dir / filepath
        
        if str(filepath) in self.data_cache:
            return self.data_cache[str(filepath)]
        
        df = pd.read_parquet(filepath)
        self.data_cache[str(filepath)] = df
        return df
    
    def collect_from_file(self, filepath: Union[str, Path]) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        根据文件扩展名自动选择读取方式
        
        Returns:
            JSON文件返回Dict，CSV/Parquet文件返回DataFrame
        """
        filepath = Path(filepath)
        if not filepath.is_absolute():
            filepath = self.base_dir / filepath
        
        suffix = filepath.suffix.lower()
        
        if suffix == '.json':
            return self.collect_from_json(filepath)
        elif suffix == '.csv':
            return self.collect_from_csv(filepath)
        elif suffix == '.parquet':
            return self.collect_from_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Supported formats: .json, .csv, .parquet")
    
    def _extract_metrics_from_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        从DataFrame中提取训练指标
        
        支持的列名格式：
        - train_loss, train_losses, train_loss_epoch
        - val_loss, val_losses, valid_loss, validation_loss
        - epoch, epochs
        - train_*, val_*, valid_* (用于各种指标)
        """
        metrics = {
            'train_losses': [],
            'val_losses': [],
            'train_scores': {},
            'val_scores': {},
            'learning_rates': [],
            'epochs': []
        }
        
        # 查找损失列
        train_loss_cols = [col for col in df.columns if 'train_loss' in col.lower() and 'epoch' not in col.lower()]
        val_loss_cols = [col for col in df.columns if any(x in col.lower() for x in ['val_loss', 'valid_loss', 'validation_loss']) and 'epoch' not in col.lower()]
        
        if train_loss_cols:
            metrics['train_losses'] = df[train_loss_cols[0]].dropna().tolist()
        if val_loss_cols:
            metrics['val_losses'] = df[val_loss_cols[0]].dropna().tolist()
        
        # 查找epoch列
        epoch_cols = [col for col in df.columns if col.lower() in ['epoch', 'epochs']]
        if epoch_cols:
            metrics['epochs'] = df[epoch_cols[0]].dropna().tolist()
        elif len(metrics['train_losses']) > 0:
            metrics['epochs'] = list(range(1, len(metrics['train_losses']) + 1))
        
        # 查找学习率列
        lr_cols = [col for col in df.columns if 'lr' in col.lower() or 'learning_rate' in col.lower()]
        if lr_cols:
            metrics['learning_rates'] = df[lr_cols[0]].dropna().tolist()
        
        # 查找训练指标列（train_开头但不是loss）
        train_metric_cols = [col for col in df.columns 
                            if col.lower().startswith('train_') 
                            and 'loss' not in col.lower()
                            and col not in train_loss_cols]
        for col in train_metric_cols:
            metric_name = col.replace('train_', '').replace('_', ' ').title()
            metrics['train_scores'][metric_name] = df[col].dropna().tolist()
        
        # 查找验证指标列（val_或valid_开头但不是loss）
        val_metric_cols = [col for col in df.columns 
                          if (col.lower().startswith('val_') or col.lower().startswith('valid_'))
                          and 'loss' not in col.lower()
                          and col not in val_loss_cols]
        for col in val_metric_cols:
            metric_name = col.replace('val_', '').replace('valid_', '').replace('_', ' ').title()
            metrics['val_scores'][metric_name] = df[col].dropna().tolist()
        
        return metrics
    
    def collect_training_metrics(self, 
                                metrics_file: Optional[Union[str, Path]] = None,
                                recovery_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """收集训练指标数据，支持JSON、CSV和Parquet格式"""
        metrics = {
            'train_losses': [],
            'val_losses': [],
            'train_scores': {},
            'val_scores': {},
            'learning_rates': [],
            'epochs': []
        }
        
        # 从metrics文件读取
        if metrics_file:
            filepath = Path(metrics_file)
            if not filepath.is_absolute():
                filepath = self.base_dir / filepath
            
            if not filepath.exists():
                raise FileNotFoundError(f"Metrics file not found: {filepath}")
            
            suffix = filepath.suffix.lower()
            
            if suffix == '.json':
                # JSON格式：嵌套字典
                data = self.collect_from_json(filepath)
                metrics['train_losses'] = data.get('train_losses', [])
                metrics['val_losses'] = data.get('val_losses', [])
                metrics['train_scores'] = data.get('train_epoch_scores', {})
                metrics['val_scores'] = data.get('valid_epoch_scores', {})
            elif suffix in ['.csv', '.parquet']:
                # CSV/Parquet格式：表格形式
                if suffix == '.csv':
                    df = self.collect_from_csv(filepath)
                else:
                    df = self.collect_from_parquet(filepath)
                
                # 从DataFrame提取指标
                df_metrics = self._extract_metrics_from_dataframe(df)
                metrics.update(df_metrics)
            else:
                raise ValueError(f"Unsupported file format: {suffix}. Supported formats: .json, .csv, .parquet")
        
        # 从recovery目录读取
        if recovery_dir:
            recovery_dir = Path(recovery_dir)
            if not recovery_dir.is_absolute():
                recovery_dir = self.base_dir / recovery_dir
            
            # 尝试多种可能的文件名
            possible_files = [
                recovery_dir / "recovery_info.json",
                recovery_dir / "recovery_info.csv",
                recovery_dir / "recovery_info.parquet",
                recovery_dir / "metrics.json",
                recovery_dir / "metrics.csv",
                recovery_dir / "metrics.parquet",
            ]
            
            recovery_file = None
            for pf in possible_files:
                if pf.exists():
                    recovery_file = pf
                    break
            
            if recovery_file:
                suffix = recovery_file.suffix.lower()
                
                if suffix == '.json':
                    data = self.collect_from_json(recovery_file)
                    metrics['train_losses'] = data.get('train_losses', [])
                    metrics['val_losses'] = data.get('val_losses', [])
                    metrics['train_scores'] = data.get('train_epoch_scores', {})
                    metrics['val_scores'] = data.get('valid_epoch_scores', {})
                    if not metrics['epochs']:
                        metrics['epochs'] = list(range(1, len(metrics['train_losses']) + 1))
                elif suffix in ['.csv', '.parquet']:
                    if suffix == '.csv':
                        df = self.collect_from_csv(recovery_file)
                    else:
                        df = self.collect_from_parquet(recovery_file)
                    
                    df_metrics = self._extract_metrics_from_dataframe(df)
                    # 合并指标，优先使用已有的数据
                    for key in ['train_losses', 'val_losses', 'learning_rates', 'epochs']:
                        if df_metrics[key] and not metrics[key]:
                            metrics[key] = df_metrics[key]
                    for key in ['train_scores', 'val_scores']:
                        if df_metrics[key] and not metrics[key]:
                            metrics[key] = df_metrics[key]
        
        return metrics
    
    def find_images(self, 
                   directory: Union[str, Path],
                   patterns: Optional[List[str]] = None) -> List[Path]:
        """查找目录中的图像文件"""
        directory = Path(directory)
        if not directory.is_absolute():
            directory = self.base_dir / directory
        
        if patterns is None:
            patterns = ['*.png', '*.jpg', '*.jpeg', '*.pdf', '*.svg']
        
        images = []
        for pattern in patterns:
            images.extend(directory.rglob(pattern))
        
        return sorted(images)


class PlotGenerator:
    """图表生成器"""
    
    def __init__(self, output_dir: Union[str, Path], style: str = 'seaborn-v0_8-paper'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.style = style
        plt.style.use(style)
        
        # 设置matplotlib字体为英语
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
        plt.rcParams['axes.unicode_minus'] = False
    
    def _resolve_save_path(self, save_path: Optional[Union[str, Path]], default_name: str) -> Path:
        """
        解析保存路径，确保路径正确且目录存在
        
        Args:
            save_path: 用户指定的保存路径
            default_name: 默认文件名（当save_path为None时使用）
        
        Returns:
            解析后的绝对路径
        """
        if save_path is None:
            save_path = self.output_dir / default_name
        else:
            save_path = Path(save_path)
            if save_path.is_absolute():
                # 绝对路径，直接使用
                pass
            else:
                # 相对路径：检查是否已经在output_dir下
                # 如果用户传入 output_dir / "file.png"，这会是相对路径
                # 我们需要检查它是否已经在output_dir下
                try:
                    # 尝试解析为绝对路径
                    abs_path = save_path.resolve()
                    abs_output_dir = self.output_dir.resolve()
                    # 检查是否在output_dir下
                    abs_path.relative_to(abs_output_dir)
                    # 已经在output_dir下，使用解析后的绝对路径
                    save_path = abs_path
                except (ValueError, RuntimeError):
                    # 不在output_dir下，或者路径不存在，添加到output_dir
                    # 如果只有文件名（parent是.或空），只取文件名
                    if save_path.parent == Path('.') or save_path.parent == Path(''):
                        save_path = self.output_dir / save_path.name
                    else:
                        # 有子目录，直接添加到output_dir下
                        save_path = self.output_dir / save_path
        
        # 确保父目录存在
        save_path.parent.mkdir(parents=True, exist_ok=True)
        return save_path
    
    def plot_training_curves(self,
                            train_losses: List[float],
                            val_losses: Optional[List[float]] = None,
                            title: str = "Training Curves",
                            save_path: Optional[Union[str, Path]] = None) -> Path:
        """绘制训练曲线"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        
        if val_losses:
            ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        default_name = f"training_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        save_path = self._resolve_save_path(save_path, default_name)
        
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return save_path
    
    def plot_metrics_comparison(self,
                               metrics_data: Dict[str, List[float]],
                               title: str = "Metrics Comparison",
                               save_path: Optional[Union[str, Path]] = None) -> Path:
        """绘制指标对比图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(list(metrics_data.values())[0]) + 1)
        
        for metric_name, values in metrics_data.items():
            ax.plot(epochs, values, label=metric_name, linewidth=2, marker='o', markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        default_name = f"metrics_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        save_path = self._resolve_save_path(save_path, default_name)
        
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return save_path
    
    def plot_bar_chart(self,
                      data: Dict[str, float],
                      title: str = "Bar Chart",
                      xlabel: str = "Category",
                      ylabel: str = "Value",
                      save_path: Optional[Union[str, Path]] = None) -> Path:
        """绘制柱状图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = list(data.keys())
        values = list(data.values())
        
        bars = ax.bar(categories, values, alpha=0.7, edgecolor='black', linewidth=1.2)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        default_name = f"bar_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        save_path = self._resolve_save_path(save_path, default_name)
        
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return save_path
    
    def plot_confusion_matrix(self,
                             cm: np.ndarray,
                             class_names: List[str],
                             title: str = "Confusion Matrix",
                             save_path: Optional[Union[str, Path]] = None) -> Path:
        """绘制混淆矩阵"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        default_name = f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        save_path = self._resolve_save_path(save_path, default_name)
        
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return save_path


class ImageHighlighter:
    """图像强调工具"""
    
    @staticmethod
    def highlight_region(image_path: Union[str, Path],
                        region: Tuple[int, int, int, int],
                        output_path: Optional[Union[str, Path]] = None,
                        color: Tuple[int, int, int] = (255, 0, 0),
                        thickness: int = 3) -> Path:
        """
        在图像上强调指定区域
        
        Args:
            image_path: 输入图像路径
            region: (x1, y1, x2, y2) 强调区域
            output_path: 输出路径（可选）
            color: RGB颜色
            thickness: 线条粗细
        
        Returns:
            输出图像路径
        """
        image_path = Path(image_path)
        img = cv2.imread(str(image_path))
        
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        x1, y1, x2, y2 = region
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        if output_path is None:
            output_path = image_path.parent / f"{image_path.stem}_highlighted{image_path.suffix}"
        else:
            output_path = Path(output_path)
        
        cv2.imwrite(str(output_path), img)
        return output_path
    
    @staticmethod
    def highlight_multiple_regions(image_path: Union[str, Path],
                                  regions: List[Tuple[int, int, int, int]],
                                  output_path: Optional[Union[str, Path]] = None,
                                  colors: Optional[List[Tuple[int, int, int]]] = None,
                                  thickness: int = 3) -> Path:
        """在图像上强调多个区域"""
        image_path = Path(image_path)
        img = cv2.imread(str(image_path))
        
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        if colors is None:
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, (x1, y1, x2, y2) in enumerate(regions):
            color = colors[i % len(colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        if output_path is None:
            output_path = image_path.parent / f"{image_path.stem}_highlighted{image_path.suffix}"
        else:
            output_path = Path(output_path)
        
        cv2.imwrite(str(output_path), img)
        return output_path


class LatexExporter:
    """LaTeX导出器"""
    
    def __init__(self, output_file: Union[str, Path]):
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.sections: List[str] = []
        self.images_dir = self.output_file.parent / "figures"
        self.images_dir.mkdir(exist_ok=True)
    
    def escape_latex(self, text: str) -> str:
        """转义LaTeX特殊字符"""
        if pd.isna(text):
            return ''
        
        text = str(text)
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
    
    def add_table(self, table_info: TableInfo):
        """添加表格到LaTeX文档"""
        import sys
        from pathlib import Path
        # 确保可以导入data_to_latex
        tools_path = Path(__file__).parent
        if str(tools_path) not in sys.path:
            sys.path.insert(0, str(tools_path))
        from data_to_latex import DataToLatexConverter
        
        # 创建临时CSV文件
        temp_csv = self.output_file.parent / "temp_table.csv"
        table_info.data.to_csv(temp_csv, index=False)
        
        # 使用DataToLatexConverter转换
        converter = DataToLatexConverter(
            input_file=str(temp_csv),
            latex_type='table',
            caption=table_info.caption,
            label=table_info.label,
            table_style=table_info.style,
            highlight_best=table_info.highlight_best,
            highlight_second=table_info.highlight_second,
            metric_columns=table_info.metric_columns,
            higher_is_better=table_info.higher_is_better
        )
        converter.load_data()
        latex_code = converter.convert()
        
        # 清理临时文件
        temp_csv.unlink()
        
        self.sections.append(latex_code)
    
    def add_image(self, image_info: ImageInfo):
        """添加图像到LaTeX文档"""
        # 复制图像到figures目录
        image_name = image_info.path.name
        dest_path = self.images_dir / image_name
        
        # 如果源图像存在，复制它
        if image_info.path.exists():
            import shutil
            shutil.copy2(image_info.path, dest_path)
        
        # 生成LaTeX代码
        width = image_info.width or r"0.8\textwidth"
        caption = self.escape_latex(image_info.caption) if image_info.caption else ""
        label = image_info.label if image_info.label else ""
        
        latex_code = f"""
\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width={width}]{{figures/{image_name}}}
"""
        if caption:
            latex_code += f"\\caption{{{caption}}}\n"
        if label:
            latex_code += f"\\label{{{label}}}\n"
        latex_code += "\\end{figure}\n"
        
        self.sections.append(latex_code)
    
    def add_section(self, title: str, content: str = ""):
        """添加章节"""
        title_escaped = self.escape_latex(title)
        section = f"\\section{{{title_escaped}}}\n"
        if content:
            section += content + "\n"
        self.sections.append(section)
    
    def add_subsection(self, title: str, content: str = ""):
        """添加子章节"""
        title_escaped = self.escape_latex(title)
        subsection = f"\\subsection{{{title_escaped}}}\n"
        if content:
            subsection += content + "\n"
        self.sections.append(subsection)
    
    def export(self, title: str = "Report", author: str = "", documentclass: str = "article"):
        """导出完整的LaTeX文档"""
        # 生成LaTeX文档
        latex_doc = f"""\\documentclass{{{documentclass}}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{longtable}}
\\usepackage{{float}}
\\usepackage{{hyperref}}

\\title{{{self.escape_latex(title)}}}
\\author{{{self.escape_latex(author)}}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

"""
        
        # 添加所有章节
        for section in self.sections:
            latex_doc += section + "\n"
        
        latex_doc += "\\end{document}\n"
        
        # 保存文件
        self.output_file.write_text(latex_doc, encoding='utf-8')
        
        return self.output_file


class ReportGenerator:
    """综合报告生成器"""
    
    def __init__(self, 
                 base_dir: Union[str, Path],
                 output_dir: Union[str, Path],
                 latex_output: Optional[Union[str, Path]] = None):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.collector = DataCollector(self.base_dir)
        self.plotter = PlotGenerator(self.output_dir)
        self.highlighter = ImageHighlighter()
        
        if latex_output:
            self.latex_exporter = LatexExporter(latex_output)
        else:
            self.latex_output = self.output_dir / "report.tex"
            self.latex_exporter = LatexExporter(self.latex_output)
        
        self.images: List[ImageInfo] = []
        self.tables: List[TableInfo] = []
    
    def generate_training_report(self,
                                metrics_file: Optional[Union[str, Path]] = None,
                                recovery_dir: Optional[Union[str, Path]] = None,
                                include_curves: bool = True,
                                include_metrics: bool = True) -> Dict[str, Path]:
        """生成训练报告"""
        # 收集数据
        metrics = self.collector.collect_training_metrics(metrics_file, recovery_dir)
        
        generated_files = {}
        
        # 生成训练曲线
        if include_curves and metrics['train_losses']:
            curve_path = self.plotter.plot_training_curves(
                train_losses=metrics['train_losses'],
                val_losses=metrics['val_losses'] if metrics['val_losses'] else None,
                title="Training and Validation Loss"
            )
            generated_files['training_curves'] = curve_path
            
            # 添加到图像列表
            self.images.append(ImageInfo(
                path=curve_path,
                caption="Training and validation loss curves",
                label="fig:training_curves"
            ))
        
        # 生成指标对比图
        if include_metrics and metrics['train_scores']:
            metrics_data = {}
            for metric_name, scores in metrics['train_scores'].items():
                if isinstance(scores, list):
                    metrics_data[f"Train {metric_name}"] = scores
            
            if metrics_data:
                metrics_path = self.plotter.plot_metrics_comparison(
                    metrics_data=metrics_data,
                    title="Training Metrics Comparison"
                )
                generated_files['metrics_comparison'] = metrics_path
                
                self.images.append(ImageInfo(
                    path=metrics_path,
                    caption="Training metrics comparison",
                    label="fig:metrics_comparison"
                ))
        
        return generated_files
    
    def highlight_image_regions(self,
                               image_path: Union[str, Path],
                               regions: List[Tuple[int, int, int, int]],
                               caption: str = "",
                               label: str = "") -> Path:
        """强调图像区域并添加到报告"""
        image_path = Path(image_path)
        if not image_path.is_absolute():
            image_path = self.base_dir / image_path
        
        highlighted_path = self.highlighter.highlight_multiple_regions(
            image_path=image_path,
            regions=regions
        )
        
        self.images.append(ImageInfo(
            path=highlighted_path,
            caption=caption or f"Highlighted regions in {image_path.name}",
            label=label or f"fig:highlighted_{image_path.stem}",
            highlight_region=regions[0] if regions else None
        ))
        
        return highlighted_path
    
    def add_table_from_data(self,
                           data: Union[pd.DataFrame, Dict, List[Dict]],
                           caption: str = "",
                           label: str = "",
                           **kwargs) -> TableInfo:
        """从数据创建表格"""
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, list):
            data = pd.DataFrame(data)
        
        table_info = TableInfo(
            data=data,
            caption=caption,
            label=label,
            **kwargs
        )
        
        self.tables.append(table_info)
        return table_info
    
    def export_to_latex(self, title: str = "Training Report", author: str = ""):
        """导出为LaTeX文档"""
        # 添加所有表格
        for table_info in self.tables:
            self.latex_exporter.add_table(table_info)
        
        # 添加所有图像
        for image_info in self.images:
            self.latex_exporter.add_image(image_info)
        
        # 导出
        return self.latex_exporter.export(title=title, author=author)


def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='综合报告生成器')
    parser.add_argument('--base-dir', type=str, default='.',
                       help='基础目录（训练/推理结果所在目录）')
    parser.add_argument('--output-dir', type=str, default='./reports',
                       help='输出目录')
    parser.add_argument('--latex-output', type=str,
                       help='LaTeX输出文件路径')
    parser.add_argument('--metrics-file', type=str,
                       help='训练指标JSON文件')
    parser.add_argument('--recovery-dir', type=str,
                       help='恢复信息目录')
    parser.add_argument('--title', type=str, default='Training Report',
                       help='报告标题')
    parser.add_argument('--author', type=str, default='',
                       help='作者')
    
    args = parser.parse_args()
    
    # 创建报告生成器
    generator = ReportGenerator(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        latex_output=args.latex_output
    )
    
    # 生成报告
    generator.generate_training_report(
        metrics_file=args.metrics_file,
        recovery_dir=args.recovery_dir
    )
    
    # 导出LaTeX
    latex_file = generator.export_to_latex(title=args.title, author=args.author)
    print(f"Report generated: {latex_file}")


if __name__ == '__main__':
    main()

