from pathlib import Path
from typing import Any, Mapping, Sequence

from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
import logging
from rich.table import Table
from rich.console import Console
import numpy as np

from src.config import get_config, ALL_STYLES
from src.utils import (
    select_postprocess_fn,
    Timer,
    model_flops,
    reset_peak_memory_stats,
    log_memory_cost,
)
from src.recorder import MeterRecorder, DataSaver
from src.constants import TestOutputFilenameEnv
from src.utils.ndict import ModelOutput, NDict

class Tester:
    def __init__(self, output_dir: Path, model: nn.Module):
        self.output_dir = output_dir
        self.model = model

        c = get_config()
        # 先获取 class_labels 和 metric_labels
        class_labels = c['classes']
        metric_labels = c['metrics']

        # 注册 class_labels 后再创建目录
        self.filename_env = TestOutputFilenameEnv()
        self.filename_env.register(test_dir=self.output_dir, class_labels=class_labels)
        self.filename_env.prepare_dir()

        self.logger = logging.getLogger('test')
        self.data_saver = DataSaver()
        self.calculator = MeterRecorder(class_labels, metric_labels, logger=self.logger, saver=self.data_saver, prefix="test_")
        self.timer = Timer(precision="s")
        self._flops_value = self._prepare_model_flops()

        postprocess_name = c.get('postprocess', "")
        self.postprocess = select_postprocess_fn(postprocess_name)
        assert postprocess_name is not None, f"Not supported postprocess function {postprocess_name}, please set 'postprocess' in config file"

    @torch.no_grad()
    def test(self, test_dataloader: DataLoader):
        c = get_config()
        device = torch.device(c['device'])
        self.model = self.model.to(device)
        reset_peak_memory_stats()
        self.timer.reset()

        with self.timer.timeit('test_total'):
            with tqdm(test_dataloader, desc="Testing...") as pbar:
                for i, batch_inputs in enumerate(pbar):
                    batch_inputs = self._move_to_device(batch_inputs, device)
                    outputs = self._run_model(batch_inputs)
                    preds = outputs.get('preds')
                    targets = outputs.get('targets')

                    if not isinstance(preds, torch.Tensor) or not isinstance(targets, torch.Tensor):
                        raise ValueError("Model must return tensor 'preds' and 'targets' for evaluation.")

                    metric_targets, metric_preds = targets, preds
                    if (
                        self.postprocess is not None
                        and isinstance(metric_targets, torch.Tensor)
                        and isinstance(metric_preds, torch.Tensor)
                    ):
                        metric_targets, metric_preds = self.postprocess(metric_targets, metric_preds)

                    self.calculator.finish_one_batch(
                        metric_targets.detach().cpu().numpy(),
                        metric_preds.detach().cpu().numpy()
                    )
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
        time_cost = self.timer.elapsed_time('test_total')
        if not np.isnan(time_cost):
            self.logger.info(f"Test Time Cost: {time_cost:.3f}s")
        if self._flops_value:
            self.logger.info(f"Test FLOPS: {self._format_flops(self._flops_value)}")
        log_memory_cost("Test", self.logger)

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

    def _move_to_device(self, value: Any, device: torch.device):
        if isinstance(value, torch.Tensor):
            return value.to(device, non_blocking=True)
        if isinstance(value, NDict):
            return NDict({k: self._move_to_device(v, device) for k, v in value.items()})
        if isinstance(value, Mapping):
            return {k: self._move_to_device(v, device) for k, v in value.items()}
        if isinstance(value, tuple):
            return tuple(self._move_to_device(v, device) for v in value)
        if isinstance(value, list):
            return [self._move_to_device(v, device) for v in value]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [self._move_to_device(v, device) for v in value]
        return value

    def _run_model(self, batch_inputs: Any) -> ModelOutput:
        inputs_obj = batch_inputs
        targets = None

        if isinstance(batch_inputs, tuple) and len(batch_inputs) == 2:
            inputs_obj, targets = batch_inputs

        if isinstance(inputs_obj, Mapping):
            model_input = inputs_obj
            targets = inputs_obj.get('targets', targets)
        else:
            model_input = inputs_obj

        result = self.model(model_input)
        model_loss: Any = None
        if isinstance(result, tuple) and len(result) == 2:
            outputs_raw, model_loss = result
        else:
            outputs_raw = result

        if isinstance(outputs_raw, ModelOutput):
            outputs = outputs_raw
        elif isinstance(outputs_raw, Mapping):
            outputs = ModelOutput(**outputs_raw)
        else:
            outputs = ModelOutput(preds=outputs_raw)

        if 'targets' not in outputs and targets is not None:
            outputs['targets'] = targets

        return outputs

    def _prepare_model_flops(self) -> float | None:
        try:
            c = get_config()
            input_sizes = c["model"]["config"]["input_sizes"]
            return model_flops(self.output_dir, self.model, input_sizes, device=c["device"], rich_print=False)
        except Exception as exc:
            self.logger.warning(f"Failed to compute FLOPS for tester: {exc}")
            return None

    @staticmethod
    def _format_flops(flops: float) -> str:
        if flops >= 1e12:
            return f"{flops / 1e12:.3f} TFLOPs"
        if flops >= 1e9:
            return f"{flops / 1e9:.3f} GFLOPs"
        if flops >= 1e6:
            return f"{flops / 1e6:.3f} MFLOPs"
        return f"{flops:.0f} FLOPs"
