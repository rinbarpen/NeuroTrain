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
from PIL import Image
from torch import nn
from torchsummary import summary

from config import get_config
from utils.recorder import Recorder
from utils.scores import scores
from utils.painter import Plot


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
    file_handler = logging.FileHandler(filename, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s %(name)s | %(message)s'
    ))

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


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
    def __init__(self, class_labels: list[str], metric_labels: list[str]=['iou', 'accuracy', 'precision', 'recall', 'f1', 'dice']):
        self.class_labels = class_labels
        self.metric_labels = metric_labels
        self.record_metrics = {metric_label: {class_label: []} for metric_label in metric_labels for class_label in class_labels}

        self.epoch_metrics = {metric_label: {class_label: []} for metric_label in metric_labels for class_label in class_labels}

    def add_one_batch(self, targets: np.ndarray, outputs: np.ndarray):
        metrics, _ = scores(targets, outputs, self.class_labels[1:], metric_labels=self.metric_labels)
        for metric_label, label_scores in metrics.items():
            for label, score in label_scores.items():
                self.record_metrics[metric_label][label].append(score)

    def finish_one_epoch(self):
        for metric_label, label_scores in self.record_metrics.items():
            for label, scores in label_scores.items():
                self.epoch_metrics[metric_label][label].append(np.mean(scores))


    def record_batches(self, output_dir: Path):
        # 
        # {'label': {'f1': [], 
        #         'recall': []}}
        # 
        all_record: dict[str, dict[str, list]] = {}
        # 
        # {'label': {'f1': mean score, 
        #         'recall': mean score}}
        # 
        mean_record: dict[str, dict[str, np.float64]] = {}
        # 
        # {'f1': mean_label_score}
        # 
        metric_record: dict[str, np.float64] = {}

        for metric_name, label_scores in self.record_metrics.items():
            for label, scores in label_scores.items():
                all_record[label][metric_name] = scores
        for label, metrics in all_record.items():
            for metric, scores in metrics:
                mean_record[label][metric] = np.mean(scores)
        for metric_name, label_scores in self.record_metrics.items():
            mean_scores = []
            for label, scores in label_scores.items():
                mean_scores.append(np.mean(scores)) 
            metric_record[metric_name] = np.mean(mean_scores)

        for label, metrics in mean_record.items():
            Recorder.record_mean_metrics(mean_record[label], output_dir / label)
        Recorder.record_metrics(metric_record, output_dir)

        # paint mean metrics for all classes
        Plot(1, len(self.class_labels)).metrics(metric_record).save(output_dir / "mean-metrics.png")

        CONFIG = get_config()
        if CONFIG['private']['wandb']:
            import wandb
            wandb.log({
                'metric': {
                    'score': {
                        'all': all_record,
                        'mean': mean_record,
                        'record': mean_record,
                    },
                    'image': {
                        'mean': output_dir / "mean-metrics.png",
                    }
                },
            })

    def record_epochs(self, output_dir: Path, n_epochs: int):
        # 
        # {'label': {'f1': [], 
        #         'recall': []}}
        # 
        all_record: dict[str, dict[str, list]] = {}
        # 
        # {'label': {'f1': mean score, 
        #         'recall': mean score}}
        # 
        mean_record: dict[str, dict[str, np.float64]] = {}
        # 
        # {'f1': mean_label_score}
        # 
        metric_record: dict[str, np.float64] = {}

        for metric_name, label_scores in self.epoch_metrics.items():
            for label, scores in label_scores.items():
                all_record[label][metric_name] = scores
        for label, metrics in all_record.items():
            for metric, scores in metrics:
                mean_record[label][metric] = np.mean(scores)
        for metric_name, label_scores in self.record_metrics.items():
            mean_scores = []
            for label, scores in label_scores.items():
                mean_scores.append(np.mean(scores)) 
            metric_record[metric_name] = np.mean(mean_scores)

        for label, metrics in all_record.items():
            Recorder.record_all_metrics(all_record[label], output_dir / label)
        for label, metrics in mean_record.items():
            Recorder.record_mean_metrics(mean_record[label], output_dir / label)
        Recorder.record_metrics(metric_record, output_dir)

        # paint metrics curve for all classes in one figure
        Plot(1, 1).subplot().many_epoch_metrics(n_epochs, all_record, self.class_labels).complete().save(output_dir / "epoch-metrics.png")
        # paint mean metrics for all classes
        Plot(1, len(self.class_labels)).metrics(metric_record).save(output_dir / "mean-metrics.png")

        CONFIG = get_config()
        if CONFIG['private']['wandb']:
            import wandb
            wandb.log({
                'metric': {
                    'score': {
                        'all': all_record,
                        'mean': mean_record,
                        'record': mean_record,
                    },
                    'image': {
                        'mean': output_dir / "mean-metrics.png",
                        'epoch': output_dir / "epoch-metrics.png",
                    }
                },
            })
