import logging
import colorlog
import os.path
import os
import time
from time import strftime

import cv2
import numpy as np
import torch
from pathlib import Path
from PIL.Image import Image 
from torch import nn
from torchsummary import summary

from config import CONFIG, get_config, ALL_METRIC_LABELS
from utils.recorder import Recorder
from utils.scores import scores
from utils.painter import Plot
from utils.typed import ClassLabelsList, ClassMetricManyScoreDict, ClassMetricOneScoreDict, MetricClassManyScoreDict, MetricClassOneScoreDict, MetricLabelOneScoreDict, MetricLabelsList

def prepare_logger():
    log_colors = {
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'FATAL': 'bold_red',
    }
    formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s %(levelname)s %(name)s | %(message)s',
        log_colors=log_colors
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File handler
    os.makedirs('logs', exist_ok=True)
    filename = os.path.join('logs', strftime('%Y%m%d_%H%M%S.log', time.localtime()))
    file_handler = logging.FileHandler(filename, encoding='utf-8', delay=True)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s %(name)s | %(message)s'
    ))

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if CONFIG['private']['verbose'] else logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # train_logger = logging.getLogger('train')
    # train_logger.setLevel(logging.DEBUG if CONFIG['private']['verbose'] else logging.INFO)
    # train_logger.addHandler(console_handler)
    # train_logger.addHandler(file_handler)
    # recorder_logger = logging.getLogger('recorder')
    # train_logger.setLevel(logging.DEBUG if CONFIG['private']['verbose'] else logging.INFO)
    # recorder_logger.addHandler(console_handler)
    # recorder_logger.addHandler(file_handler)

def save_model(path: Path, model: nn.Module, *, 
               ext_path: Path|None=None,
               optimizer=None, lr_scheduler=None, scaler=None, **kwargs):
    model_cp = model.state_dict()

    try:
        torch.save(model_cp, path)
    except FileExistsError as e:
        path = path.parent / (path.stem +
                              strftime("%Y%m%d_%H%M%S", time.localtime()))
        torch.save(model_cp, path)

    if ext_path:
        ext_cp = dict()
        if optimizer:
            ext_cp["optimizer"] = optimizer.state_dict()
        if lr_scheduler:
            ext_cp["lr_scheduler"] = lr_scheduler.state_dict()
        if scaler:
            ext_cp["scaler"] = scaler.state_dict()
        for k, v in kwargs.items():
            ext_cp[k] = v
        try:
            torch.save(ext_cp, ext_path)
        except FileExistsError as e:
            ext_path = ext_path.parent / (ext_path.stem +
                                strftime("%Y%m%d_%H%M%S", time.localtime()))
            torch.save(ext_cp, ext_path)


def load_model(path: Path, map_location: str = 'cuda'):
    return torch.load(path, 
                      map_location=torch.device(map_location))
def load_model_ext(ext_path: Path, map_location: str = 'cuda'):
    return torch.load(ext_path, 
                      map_location=torch.device(map_location))

def save_model_to_onnx(path: Path, model: nn.Module, input_size: tuple):
    dummy_input = torch.randn(input_size)
    torch.onnx.export(model, dummy_input, path)

def summary_model_info(model_src: Path | torch.nn.Module, input_size: torch.Tensor, device: str="cpu"):
    if isinstance(model_src, Path):
        checkpoint = load_model(model_src, device)
        summary(checkpoint, input_size=input_size, device=device)
    elif isinstance(model_src, torch.nn.Module):
        summary(model_src, input_size=input_size, device=device)


def save_numpy_data(path: Path, data: np.ndarray | torch.Tensor):
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()

    try:
        np.save(path, data)
    except FileNotFoundError as e:
        path.parent.mkdir(parents=True)
        np.save(path, data)

def load_numpy_data(path: Path):
    try:
        data = np.load(path)
        return data
    except FileNotFoundError as e:
        colorlog.error(f'File is not found: {e}')
        raise e


def tuple2list(t: tuple):
    return list(t)

def list2tuple(l: list):
    return tuple(l)

def image_to_numpy(img: Image|cv2.Mat) -> np.ndarray:
    if isinstance(img, Image):
        img_np = np.array(img) # (H, W, C)
        if img_np.ndim == 3:
            img_np = img_np.transpose(2, 0, 1) # (C, H, W)
    elif isinstance(img, cv2.Mat):
        img_np = np.array(img)
    
    # output shape: (C, H, W) for RGB or (H, W) for gray
    return img_np

class ScoreCalculator:
    def __init__(self, class_labels: ClassLabelsList, metric_labels: MetricLabelsList=ALL_METRIC_LABELS):
        self.class_labels = class_labels
        self.metric_labels = metric_labels
        self.is_prepared = False

        # {'recall': {
        #   '0': [], '1': []},
        #  'precision: {
        #   '0': [], '1': []}}
        self.record_metrics: MetricClassManyScoreDict = {metric_label: {class_label: []} for metric_label in metric_labels for class_label in class_labels}
        # but epoch_metrics is saved by epoch
        self.epoch_metrics: MetricClassManyScoreDict = {metric_label: {class_label: []} for metric_label in metric_labels for class_label in class_labels}

        # 
        # {'label': {'f1': [], 
        #         'recall': []}}
        # 
        self.all_record: ClassMetricManyScoreDict = {class_label: {metric: [] for metric in metric_labels} for class_label in class_labels}
        # 
        # {'label': {'f1': mean score, 
        #         'recall': mean score}}
        # 
        self.mean_record: ClassMetricOneScoreDict = {class_label: {metric: 0.0 for metric in metric_labels} for class_label in class_labels}
        # 
        # {'f1': {'label': mean_score}}
        # 
        self.metric_label_record: MetricClassOneScoreDict = {metric: {label: 0.0 for label in class_labels} for metric in metric_labels}
        # 
        # {'f1': mean_label_score}
        # 
        self.metric_record: MetricLabelOneScoreDict = {metric: 0.0 for metric in metric_labels}


    def add_one_batch(self, targets: np.ndarray, outputs: np.ndarray):
        metrics, _ = scores(targets, outputs, self.class_labels, metric_labels=self.metric_labels)
        for metric_label, label_scores in metrics.items():
            for label, score in label_scores.items():
                self.record_metrics[metric_label][label].append(score)

    def finish_one_epoch(self):
        for metric_label, label_scores in self.record_metrics.items():
            for label, scores in label_scores.items():
                self.epoch_metrics[metric_label][label].append(np.mean(scores))


    def record_batches(self, output_dir: Path):
        self._prepare(output_dir)

        for label, metrics in self.mean_record.items():
            label_dir = output_dir / label
            Recorder.record_mean_metrics(metrics, label_dir)
        Recorder.record_metrics(self.metric_record, output_dir)

        # paint mean metrics for all classes
        # n = len(self.class_labels)
        # nrows, ncols = (n + 2) // 3, 3 if n >= 3 else 1, n
        # Plot(nrows, ncols).metrics(self.metric_record).save(output_dir / "mean-metrics.png")
        Plot(1, 1).subplot().many_metrics(self.mean_record).complete().save(output_dir / "mean-metrics.png")

        CONFIG = get_config()
        if CONFIG['private']['wandb']:
            import wandb
            wandb.log({
                'metric': {
                    'score': {
                        'all': self.all_record,
                        'mean': self.mean_record,
                    },
                    'image': {
                        'mean': output_dir / "mean-metrics.png",
                    }
                },
            })

    def record_epochs(self, output_dir: Path, n_epochs: int):
        self._prepare(output_dir)

        epoch_mean_metrics: dict[str, dict[str, list]] = {class_label: {metric: [] for metric in self.metric_labels} for class_label in self.class_labels}
        for metric_name, label_scores in self.epoch_metrics.items():
            for label, scores in label_scores.items():
                epoch_mean_metrics[label][metric_name] = scores

        for label, metrics in epoch_mean_metrics.items():
            label_dir = output_dir / label
            Recorder.record_all_metrics(metrics, label_dir)
        for label, metrics in self.mean_record.items():
            label_dir = output_dir / label
            Recorder.record_mean_metrics(metrics, label_dir)
        Recorder.record_metrics(self.metric_record, output_dir)

        # paint metrics curve for all classes in one figure
        n = len(self.metric_labels)
        nrows, ncols = ((n + 2) // 3, 3) if n > 3 else (1, n)
        plot = Plot(nrows, ncols)
        for metric in self.metric_labels:
            plot.subplot().many_epoch_metrics(n_epochs, self.epoch_metrics[metric], self.class_labels, title=metric).complete()
        plot.save(output_dir / "epoch-metrics.png")

        # paint mean metrics for all classes
        # n = len(self.class_labels)
        # nrows, ncols = ((n + 2) // 3, 3) if n > 3 else (1, n)
        Plot(1, 1).subplot().many_metrics(self.mean_record).complete().save(output_dir / "mean-metrics.png")

        CONFIG = get_config()
        if CONFIG['private']['wandb']:
            import wandb
            wandb.log({
                'metric': {
                    'score': {
                        'all': self.all_record,
                        'mean': self.mean_record,
                    },
                    'image': {
                        'mean': output_dir / "mean-metrics.png",
                        'epoch': output_dir / "epoch-metrics.png",
                    }
                },
            })

    def _prepare(self, output_dir: Path):
        if self.is_prepared:
            return

        for metric_name, label_scores in self.record_metrics.items():
            for label, scores in label_scores.items():
                self.all_record[label][metric_name] = scores
        for label, metrics in self.all_record.items():
            for metric_name, scores in metrics.items():
                self.mean_record[label][metric_name] = np.mean(scores)
        for metric_name, label_scores in self.record_metrics.items():
            for label, scores in label_scores.items():
                self.metric_label_record[metric_name][label] = np.mean(scores)
        for metric_name, label_scores in self.record_metrics.items():
            mean_scores = []
            for label, scores in label_scores.items():
                mean_scores.append(np.mean(scores))
            self.metric_record[metric_name] = np.mean(mean_scores)
        
        for label in self.class_labels:
            label_dir = output_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)
        self.is_prepared = True
