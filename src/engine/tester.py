from pathlib import Path
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
import logging
from rich.table import Table
from rich.console import Console

from src.config import get_config, ALL_STYLES
from src.utils import DataSaver, select_postprocess_fn
from src.recorder.metric_recorder import ScoreAggregator
from src.recorder.metric_recorder import MetricRecorder
from src.visualizer.painter import Plot

class Tester:
    def __init__(self, output_dir: Path, model: nn.Module):
        self.output_dir = output_dir
        self.model = model

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger('test')
        self.data_saver = DataSaver(output_dir)

        c = get_config()
        class_labels = c['classes']
        metric_labels = c['metrics']
        self.calculator = MetricRecorder(output_dir, class_labels, metric_labels, logger=self.logger, saver=self.data_saver)

        postprocess_name = c.get('postprocess', "")
        self.postprocess = select_postprocess_fn(postprocess_name)
        assert postprocess_name is not None, f"Not supported postprocess function {postprocess_name}, please set 'postprocess' in config file"

    @torch.no_grad()
    def test(self, test_dataloader: DataLoader):
        c = get_config()
        device = torch.device(c['device'])
        self.model = self.model.to(device)

        for batch_inputs in tqdm(test_dataloader, desc="Testing..."):
            inputs, targets = batch_inputs['image'].to(device), batch_inputs['mask'].to(device)
            outputs = self.model(inputs)
            targets, outputs = self.postprocess(targets, outputs)
            self.calculator.finish_one_batch(
                targets.detach().cpu().numpy(),
                outputs.detach().cpu().numpy())

        self.calculator.record_batches()
        self._print_table()

    def _print_table(self):
        c = get_config()
        class_labels  = c['classes']
        metric_labels = c['metrics']
        scores = ScoreAggregator(self.calculator.all_metric_label_scores)
        test_mean_scores = scores.ml1_mean
        test_metric_class_scores = (scores.mc1_mean, scores.mc1_std)
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
