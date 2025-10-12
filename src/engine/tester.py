from pathlib import Path
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
import logging
from rich.table import Table
from rich.console import Console
import numpy as np

from src.config import get_config, ALL_STYLES
from src.utils import select_postprocess_fn
from src.recorder import MeterRecorder, DataSaver
from src.constants import TestOutputFilenameEnv

class Tester:
    def __init__(self, output_dir: Path, model: nn.Module):
        self.output_dir = output_dir
        self.model = model

        self.filename_env = TestOutputFilenameEnv()
        self.filename_env.register(test_dir=self.output_dir)
        self.filename_env.prepare_dir()

        self.logger = logging.getLogger('test')
        self.data_saver = DataSaver()

        c = get_config()
        class_labels = c['classes']
        metric_labels = c['metrics']
        self.calculator = MeterRecorder(class_labels, metric_labels, logger=self.logger, saver=self.data_saver, prefix="test_")

        postprocess_name = c.get('postprocess', "")
        self.postprocess = select_postprocess_fn(postprocess_name)
        assert postprocess_name is not None, f"Not supported postprocess function {postprocess_name}, please set 'postprocess' in config file"

    @torch.no_grad()
    def test(self, test_dataloader: DataLoader):
        c = get_config()
        device = torch.device(c['device'])
        self.model = self.model.to(device)

        with tqdm(test_dataloader, desc="Testing...") as pbar:
            for i, batch_inputs in enumerate(pbar):
                inputs, targets = batch_inputs['image'].to(device), batch_inputs['mask'].to(device)
                outputs = self.model(inputs)
                if self.postprocess is not None:
                    targets, outputs = self.postprocess(targets, outputs)
                self.calculator.finish_one_batch(
                    targets.detach().cpu().numpy(),
                    outputs.detach().cpu().numpy())
                pbar.set_postfix({'step': f'{i+1}/{len(test_dataloader)}'})

        # 对于Tester，所有batch处理完后直接记录结果
        self.calculator.record_batches(
            output_dir=self.output_dir,
            filenames={
                'all_metric_by_class': str(self.output_dir / "{class_label}" / "all_metric.csv"),
                'mean_metric_by_class': str(self.output_dir / "{class_label}" / "mean_metric.csv"),
                'std_metric_by_class': str(self.output_dir / "{class_label}" / "std_metric.csv"),
                'mean_metric': str(self.output_dir / "mean_metric.csv"),
                'std_metric': str(self.output_dir / "std_metric.csv"),
            }
        )
        self.logger.info(f"Test metrics saved to {self.output_dir}")
        
        self._print_table()

    def _print_table(self):
        c = get_config()
        class_labels  = c['classes']
        metric_labels = c['metrics']
        
        # 使用MeterRecorder的内置统计功能
        test_mean_scores = self.calculator.get_current_metrics()
        
        # 对于Tester，直接从batch_meters计算最终结果
        cm1_mean = self.calculator._compute_cm1_from_batch(np.mean)
        cm1_std = self.calculator._compute_cm1_from_batch(np.std)
        
        # 将cm1转换为mc1格式以适配表格
        mc1_mean = {}
        mc1_std = {}
        for metric in metric_labels:
            mc1_mean[metric] = {cls: cm1_mean[cls][metric] for cls in class_labels}
            mc1_std[metric] = {cls: cm1_std[cls][metric] for cls in class_labels}
        test_metric_class_scores = (mc1_mean, mc1_std)
        
        styles = ALL_STYLES
        console = Console()
        table = Table(title='Metric Class Mean Score(Test)')
        table.add_column("Class/Metric", justify="center")
        for metric, style in zip(metric_labels, styles[:len(metric_labels)]):
            table.add_column(metric, justify="center", style=style)
        for class_label in class_labels:
            mean_scores = test_metric_class_scores[0]
            std_scores = test_metric_class_scores[1]
            table.add_row("Test/" + class_label, *[f'{mean_scores[metric][class_label]:.3f} ± {std_scores[metric][class_label]:.3f}' for metric in metric_labels])
        console.print(table)
        table = Table(title='Summary of Metric(Test)')
        table.add_column("Metric", justify="center")
        for metric, style in zip(metric_labels, styles[:len(metric_labels)]):
            table.add_column(metric, justify="center", style=style)
        table.add_row("Test", *[f'{score:.3f}' for score in test_mean_scores.values()])
        console.print(table)
