from pathlib import Path
from tqdm import tqdm
import torch
from torch import nn
from PIL import Image
import numpy as np
import logging

from src.config import get_config
from src.utils import Timer, get_transforms, select_postprocess_fn
from src.monitor import TrainingMonitor, ProgressTracker

class Predictor:
    def __init__(self, output_dir: Path, model: nn.Module):
        self.output_dir = output_dir
        self.model = model
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger('predict')
        self.timer = Timer()
        c = get_config()
        postprocess_name = c.get('postprocess', "")
        self.postprocess = select_postprocess_fn(postprocess_name)
        assert postprocess_name is not None, f"Not supported postprocess function {postprocess_name}, please set 'postprocess' in config file"
        self.monitor = None
        self.progress_tracker = None
        self.web_monitor = None
        monitor_conf = c.get('monitor', {})
        if monitor_conf.get('enabled', False):
            try:
                self.monitor = TrainingMonitor(monitor_conf.get('config'))
                self.monitor.start_monitoring()
                self.progress_tracker = ProgressTracker()
                self.logger.info("Prediction monitor enabled")
            except Exception as e:
                self.logger.warning(f"Monitor initialization failed: {e}")
                self.monitor = None
                self.progress_tracker = None
        if monitor_conf.get('web'):
            try:
                from monitor import create_web_monitor
                self.web_monitor = create_web_monitor()
                self.web_monitor.start(block=False)
                self.logger.info("Web monitor started")
            except Exception as e:
                self.logger.warning(f"Web monitor initialization failed: {e}")
                self.web_monitor = None

    @torch.inference_mode()
    def predict(self, inputs: list[Path], **kwargs):
        c = get_config()
        device = torch.device(c['device'])
        self.model = self.model.to(device)
        self.model.eval()
        if self.progress_tracker:
            self.progress_tracker.start_training(1, len(inputs))
        if self.web_monitor:
            try:
                self.web_monitor.start_training(1, len(inputs))
            except Exception as exc:
                self.logger.warning(f"Web monitor start_training failed: {exc}")
                self.web_monitor = None
        for idx, input in enumerate(tqdm(inputs, desc="Predicting..."), 1):
            if self.progress_tracker:
                if idx == 1:
                    self.progress_tracker.start_epoch(0)
                self.progress_tracker.start_step(idx)
            if self.web_monitor:
                try:
                    self.web_monitor.start_step(idx)
                except Exception as exc:
                    self.logger.warning(f"Web monitor start_step failed: {exc}")
                    self.web_monitor = None
            input_filename = input.name
            with self.timer.timeit(task=input_filename + '.preprocess'):
                image = Image.open(input).convert('L')
                size = image.size # (H, W)
                transforms = get_transforms()
                image_tensor = transforms(image).unsqueeze(0)
            with self.timer.timeit(task=input_filename + '.inference'):
                image_tensor = image_tensor.to(device)
                pred_tensor = self.model(image_tensor)
            with self.timer.timeit(task=input_filename + '.postprocess'):
                import torch.nn.functional as F
                pred = F.sigmoid(pred_tensor)
                pred[pred >= 0.5] = 255
                pred[pred < 0.5] = 0
                pred = pred.detach().cpu().numpy()
                pred = pred.squeeze(0).squeeze(0)
                pred = pred.astype(np.uint8)
            output_filename = self.output_dir / input_filename
            pred_image = Image.fromarray(pred, mode='L')
            pred_image = pred_image.resize(size)
            pred_image.save(output_filename)
            if self.monitor:
                self.monitor.update_training_metrics(
                    epoch=0,
                    step=idx,
                    loss=0.0,
                    learning_rate=0.0,
                    batch_size=1,
                    throughput=1.0 / max(self.timer.elapsed_time(input_filename + '.inference'), 1e-6)
                )
            if self.progress_tracker:
                self.progress_tracker.end_step(1)
            if self.web_monitor:
                try:
                    self.web_monitor.update_metrics(
                        epoch=0,
                        step=idx,
                        loss=0.0,
                        learning_rate=0.0,
                        batch_size=1,
                        throughput=1.0 / max(self.timer.elapsed_time(input_filename + '.inference'), 1e-6)
                    )
                    self.web_monitor.end_step()
                except Exception as exc:
                    self.logger.warning(f"Web monitor update failed: {exc}")
                    self.web_monitor = None
        if self.progress_tracker and self.progress_tracker.is_training:
            self.progress_tracker.end_training()
        if self.monitor:
            self.monitor.stop_monitoring()
        if self.web_monitor:
            try:
                self.web_monitor.end_training()
                self.web_monitor.stop()
            except Exception as exc:
                self.logger.warning(f"Failed to stop web monitor: {exc}")
        cost = self.timer.total_elapsed_time()
        self.logger.info(f'Predicting had cost {cost}s')
