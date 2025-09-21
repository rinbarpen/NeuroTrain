# analyze the number of model params and each project metrics
# TODO:
import os
import os.path
import torch
import re
from typing import TypedDict
from pathlib import Path
from pydantic import BaseModel, Field
from datetime import datetime

from src.utils.util import load_model, load_model_ext, model_gflops, Timer
from src.utils.typed import ClassLabelManyScoreDict
from src.utils.painter import Plot

# 尝试导入PDF相关库
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


MODEL_FILE = str|Path
class AnalyzeParams(BaseModel):
    model: str|Path = Field(default=None)
    model_ext: str|Path = Field(default=None)

class AnalyzeMetricParams(BaseModel):
    model_name: str
    task: str
    class_metrics: ClassLabelManyScoreDict
    super_params: dict

"""
Data Format:
task:
    run_id:
        predict:
            {xx}
            config[.json|.toml|.yaml]
        test:
            {class}:
                mean_metrics[.csv|.parquet]
            mean_metrics.png
            mean_metrics[.csv|.parquet]
            config[.json|.toml|.yaml]
        train:
            {class}:
                all_metrics[.csv|.parquet]
                mean_metrics[.csv|.parquet]
            weights:
                best.pt
                last.pt
                {net}-{epoch}of{num_epochs}.pt
                best-ext.pt                         | optional
                last-ext.pt                         | optional
                {net}-{epoch}of{num_epochs}-ext.pt  | optional
            train_loss[.csv|.parquet]
            train_epoch_loss.png
            valid_loss[.csv|.parquet]               | optional
            valid_epoch_loss.png                    | optional
            epoch_metrics.png
            mean_metric.png
            mean_metrics[.csv|.parquet]
            config[.json|.toml|.yaml]
            best.pt                                 | optional, soft link
            last.pt                                 | optional, soft link
"""

class Analyzer:
    def __init__(self, result_dir: Path):
        self.result_dir = result_dir

    def __del__(self):
        pass

    def __call__(self, x: AnalyzeParams):
        pass

    # def analyze_metric(self, params: list[AnalyzeMetricParams]):
    #     header = " ".join(["Methods"].extend(["#" + k for k in p.super_params.keys()]))
    #     content = ""
    #     for p in params:
    #         content += p.model_name.join(["<b>", "<\b>"])

    def analyze_log(self, log_file: Path):
        loss_pattern = re.compile(
            r"Epoch (?P<epoch>\d+)/(?P<num_epochs>\d+), (?:Train Loss: (?P<train_loss>[\d.]+)|Valid Loss: (?P<valid_loss>[\d.]+))"
        )

        train_losses = []
        valid_losses = []
        with log_file.open(encoding='utf-8') as f:
            line = f.readline()
            matches = loss_pattern.search(line)
            
            epoch = matches.group('epoch')
            num_epochs = matches.group('num_epochs')
            train_loss = matches.group('train_loss')
            valid_loss = matches.group('valid_loss')
            if train_loss:
                train_losses.append(train_loss)
            elif valid_loss:
                valid_losses.append(valid_loss)

    def analyze_images(self):
        train_dir, test_dir = self.result_dir / "train", self.result_dir / "test"

        n = 0

        def file_is_exist(filename: Path):
            if filename.is_file():
                n += 1
                return filename
            return None

        # train_loss_image = train_dir / "train_epoch_loss.png"
        # train_epoch_metrics_image = train_dir / "epoch_metrics.png"
        # train_mean_metrics_image = train_dir / "mean_metrics.png"
        # valid_loss_image = train_dir / "valid_epoch_loss.png"
        # test_mean_metrics_image = test_dir / "mean_metrics.png"
        
        train_loss_image = file_is_exist(train_dir / "train_epoch_loss.png")
        train_epoch_metrics_image = file_is_exist(train_dir / "epoch_metrics.png")
        train_mean_metrics_image = file_is_exist(train_dir / "mean_metrics.png")
        valid_loss_image = file_is_exist(train_dir / "valid_epoch_loss.png")
        test_mean_metrics_image = file_is_exist(test_dir / "mean_metrics.png")

        images = []
        if train_loss_image: 
            images.append(train_loss_image)
        if valid_loss_image: 
            images.append(valid_loss_image)
        if train_epoch_metrics_image: 
            images.append(train_epoch_metrics_image)
        if train_mean_metrics_image: 
            images.append(train_mean_metrics_image)
        if test_mean_metrics_image: 
            images.append(test_mean_metrics_image)

        nrows, ncols = 1, n
        plot = Plot(nrows, ncols)
        plot.images(images)
        plot.save('images_preview.png')

    def _comparative_score_table(self, scores_map: dict) -> str:
        """
        生成跨模型、跨任务的性能对比表
        
        Args:
            scores_map: 包含模型性能数据的字典，格式为:
                {
                    'model_name': {
                        'task_name': {
                            'metric_name': {
                                'score': float,
                                'up': bool  # True表示分数越高越好
                            }
                        }
                    }
                }
        
        Returns:
            str: 格式化的对比表字符串
        """
        if not scores_map:
            return "无性能数据可对比"
        
        # 收集所有任务和指标
        all_tasks = set()
        all_metrics = set()
        for model_data in scores_map.values():
            for task_name, task_metrics in model_data.items():
                all_tasks.add(task_name)
                for metric_name in task_metrics.keys():
                    all_metrics.add(metric_name)
        
        # 如果每个任务只有一个指标，将任务和指标合并显示
        if len(all_metrics) == 1:
            metric_name = list(all_metrics)[0]
            headers = [f"{task}\n{metric_name}" for task in sorted(all_tasks)]
        else:
            headers = []
            for task in sorted(all_tasks):
                for metric in sorted(all_metrics):
                    headers.append(f"{task}\n{metric}")
        
        # 构建表格
        table_lines = []
        
        # 表头
        header_line = "模型".ljust(20) + " | " + " | ".join(header.ljust(15) for header in headers)
        table_lines.append(header_line)
        table_lines.append("-" * len(header_line))
        
        # 数据行
        for model_name in sorted(scores_map.keys()):
            model_data = scores_map[model_name]
            row = [model_name.ljust(20)]
            
            for task in sorted(all_tasks):
                if task in model_data:
                    task_metrics = model_data[task]
                    for metric in sorted(all_metrics):
                        if metric in task_metrics:
                            score_data = task_metrics[metric]
                            if isinstance(score_data, dict):
                                score = score_data.get('score', 'N/A')
                                up = score_data.get('up', True)
                            else:
                                score = score_data
                                up = True
                            
                            # 格式化分数
                            if isinstance(score, (int, float)):
                                score_str = f"{score:.4f}"
                            else:
                                score_str = str(score)
                            
                            # 添加方向指示
                            if up and isinstance(score, (int, float)):
                                score_str += " ↑"
                            elif not up and isinstance(score, (int, float)):
                                score_str += " ↓"
                            
                            row.append(score_str.ljust(15))
                        else:
                            row.append("N/A".ljust(15))
                else:
                    for _ in sorted(all_metrics):
                        row.append("N/A".ljust(15))
            
            table_lines.append(" | ".join(row))
        
        return "\n".join(table_lines)
    
    def generate_comparative_report(self, scores_map: dict, output_path: Path = None) -> str:
        """
        生成完整的性能对比报告
        
        Args:
            scores_map: 性能数据字典
            output_path: 输出文件路径，如果为None则返回字符串
        
        Returns:
            str: 报告内容
        """
        report_lines = []
        report_lines.append("# 模型性能对比报告")
        report_lines.append("")
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 添加对比表
        report_lines.append("## 性能对比表")
        report_lines.append("")
        report_lines.append("```")
        report_lines.append(self._comparative_score_table(scores_map))
        report_lines.append("```")
        report_lines.append("")
        
        # 添加统计信息
        report_lines.append("## 统计信息")
        report_lines.append("")
        report_lines.append(f"- 模型数量: {len(scores_map)}")
        
        all_tasks = set()
        all_metrics = set()
        for model_data in scores_map.values():
            for task_name, task_metrics in model_data.items():
                all_tasks.add(task_name)
                for metric_name in task_metrics.keys():
                    all_metrics.add(metric_name)
        
        report_lines.append(f"- 任务数量: {len(all_tasks)}")
        report_lines.append(f"- 指标数量: {len(all_metrics)}")
        report_lines.append("")
        
        # 添加最佳性能模型
        report_lines.append("## 最佳性能模型")
        report_lines.append("")
        
        for task in sorted(all_tasks):
            best_model = None
            best_score = None
            best_metric = None
            
            for model_name, model_data in scores_map.items():
                if task in model_data:
                    task_metrics = model_data[task]
                    for metric_name, metric_data in task_metrics.items():
                        if isinstance(metric_data, dict):
                            score = metric_data.get('score')
                            up = metric_data.get('up', True)
                        else:
                            score = metric_data
                            up = True
                        
                        if isinstance(score, (int, float)):
                            if best_score is None:
                                best_score = score
                                best_model = model_name
                                best_metric = metric_name
                            elif up and score > best_score:
                                best_score = score
                                best_model = model_name
                                best_metric = metric_name
                            elif not up and score < best_score:
                                best_score = score
                                best_model = model_name
                                best_metric = metric_name
            
            if best_model:
                report_lines.append(f"- **{task}**: {best_model} ({best_metric}: {best_score:.4f})")
        
        report_content = "\n".join(report_lines)
        
        if output_path:
            output_path.write_text(report_content, encoding='utf-8')
            print(f"报告已保存到: {output_path}")
        
        return report_content
    
    def export_to_pdf(self, scores_map: dict, output_path: Path) -> bool:
        """
        将性能对比报告导出为PDF格式
        
        Args:
            scores_map: 性能数据字典
            output_path: PDF输出文件路径
        
        Returns:
            bool: 导出是否成功
        """
        if not REPORTLAB_AVAILABLE:
            print("警告: reportlab库未安装，无法导出PDF。请运行: pip install reportlab")
            return False
        
        try:
            # 创建PDF文档
            doc = SimpleDocTemplate(str(output_path), pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # 标题样式
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=1  # 居中
            )
            
            # 添加标题
            story.append(Paragraph("模型性能对比报告", title_style))
            story.append(Spacer(1, 12))
            
            # 添加生成时间
            time_style = ParagraphStyle(
                'TimeStyle',
                parent=styles['Normal'],
                fontSize=10,
                textColor=colors.grey
            )
            story.append(Paragraph(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", time_style))
            story.append(Spacer(1, 20))
            
            # 添加统计信息
            story.append(Paragraph("统计信息", styles['Heading2']))
            story.append(Spacer(1, 12))
            
            all_tasks = set()
            all_metrics = set()
            for model_data in scores_map.values():
                for task_name, task_metrics in model_data.items():
                    all_tasks.add(task_name)
                    for metric_name in task_metrics.keys():
                        all_metrics.add(metric_name)
            
            stats_data = [
                ['模型数量', str(len(scores_map))],
                ['任务数量', str(len(all_tasks))],
                ['指标数量', str(len(all_metrics))]
            ]
            
            stats_table = Table(stats_data, colWidths=[2*inch, 2*inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(stats_table)
            story.append(Spacer(1, 20))
            
            # 添加性能对比表
            story.append(Paragraph("性能对比表", styles['Heading2']))
            story.append(Spacer(1, 12))
            
            # 构建表格数据
            if len(all_metrics) == 1:
                metric_name = list(all_metrics)[0]
                headers = [f"{task}\n{metric_name}" for task in sorted(all_tasks)]
            else:
                headers = []
                for task in sorted(all_tasks):
                    for metric in sorted(all_metrics):
                        headers.append(f"{task}\n{metric}")
            
            # 表头
            table_data = [["模型"] + headers]
            
            # 数据行
            for model_name in sorted(scores_map.keys()):
                model_data = scores_map[model_name]
                row = [model_name]
                
                for task in sorted(all_tasks):
                    if task in model_data:
                        task_metrics = model_data[task]
                        for metric in sorted(all_metrics):
                            if metric in task_metrics:
                                score_data = task_metrics[metric]
                                if isinstance(score_data, dict):
                                    score = score_data.get('score', 'N/A')
                                    up = score_data.get('up', True)
                                else:
                                    score = score_data
                                    up = True
                                
                                # 格式化分数
                                if isinstance(score, (int, float)):
                                    score_str = f"{score:.4f}"
                                    if up:
                                        score_str += " ↑"
                                    else:
                                        score_str += " ↓"
                                else:
                                    score_str = str(score)
                                
                                row.append(score_str)
                            else:
                                row.append("N/A")
                    else:
                        for _ in sorted(all_metrics):
                            row.append("N/A")
                
                table_data.append(row)
            
            # 创建表格
            table = Table(table_data, colWidths=[1.5*inch] + [1*inch] * len(headers))
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            
            story.append(table)
            story.append(Spacer(1, 20))
            
            # 添加最佳性能模型
            story.append(Paragraph("最佳性能模型", styles['Heading2']))
            story.append(Spacer(1, 12))
            
            for task in sorted(all_tasks):
                best_model = None
                best_score = None
                best_metric = None
                
                for model_name, model_data in scores_map.items():
                    if task in model_data:
                        task_metrics = model_data[task]
                        for metric_name, metric_data in task_metrics.items():
                            if isinstance(metric_data, dict):
                                score = metric_data.get('score')
                                up = metric_data.get('up', True)
                            else:
                                score = metric_data
                                up = True
                            
                            if isinstance(score, (int, float)):
                                if best_score is None:
                                    best_score = score
                                    best_model = model_name
                                    best_metric = metric_name
                                elif up and score > best_score:
                                    best_score = score
                                    best_model = model_name
                                    best_metric = metric_name
                                elif not up and score < best_score:
                                    best_score = score
                                    best_model = model_name
                                    best_metric = metric_name
                
                if best_model:
                    story.append(Paragraph(f"• <b>{task}</b>: {best_model} ({best_metric}: {best_score:.4f})", styles['Normal']))
            
            # 构建PDF
            doc.build(story)
            print(f"PDF报告已保存到: {output_path}")
            return True
            
        except Exception as e:
            print(f"PDF导出失败: {e}")
            return False
