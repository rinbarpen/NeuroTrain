import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

from tqdm import tqdm
import torch
from torch import nn
from PIL import Image
import numpy as np
import logging

from src.config import get_config
from src.utils import (
    Timer,
    get_transforms,
    select_postprocess_fn,
    get_predict_postprocess_fn,
    reset_peak_memory_stats,
    log_memory_cost,
)
from src.monitor import TrainingMonitor, ProgressTracker
from src.utils.ndict import ModelOutput, Sample, NDict
from src.utils.progress_bar import format_progress_desc
from src.utils.metric_table import print_metric_scores_table
from src.metrics import get_metric_fns, many_metrics


def get_predictor(
    output_dir: Path,
    model: nn.Module,
    task_or_postprocess: str | None = None,
    **kwargs: Any,
) -> "Predictor":
    """Factory: return a Predictor for the given task/postprocess (from config if None)."""
    c = get_config()
    postprocess_name = task_or_postprocess or c.get("postprocess", "")
    return Predictor(output_dir, model)


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
        self.predict_postprocess_fn = get_predict_postprocess_fn(postprocess_name)
        if self.predict_postprocess_fn is None and postprocess_name:
            self.predict_postprocess_fn = get_predict_postprocess_fn("binary_segmentation")
        assert postprocess_name, f"postprocess must be set in config file"
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
        reset_peak_memory_stats()
        if self.progress_tracker:
            self.progress_tracker.start_training(1, len(inputs))
        if self.web_monitor:
            try:
                self.web_monitor.start_training(1, len(inputs))
            except Exception as exc:
                self.logger.warning(f"Web monitor start_training failed: {exc}")
                self.web_monitor = None
        classification_results: list[dict] = []
        total = len(inputs)
        pbar = tqdm(inputs, desc="Predicting...")
        for idx, input in enumerate(pbar, 1):
            pbar.set_description(format_progress_desc("Predicting...", idx, total, idx, total, 1, 1))
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
                size = image.size  # (W, H) from PIL
                transforms = get_transforms()
                image_tensor = transforms(image)
                if not isinstance(image_tensor, torch.Tensor):
                    image_tensor = torch.as_tensor(np.array(image), dtype=torch.float32)
                image_tensor = image_tensor.unsqueeze(0)
            with self.timer.timeit(task=input_filename + '.inference'):
                image_tensor = image_tensor.to(device)
                batch_sample = Sample(inputs=image_tensor.to(device))
                outputs = self._run_model(batch_sample)
                pred_tensor = outputs.get('preds')
                if not isinstance(pred_tensor, torch.Tensor):
                    raise ValueError("ModelOutput missing tensor 'preds' during predict.")
            with self.timer.timeit(task=input_filename + '.postprocess'):
                fn = self.predict_postprocess_fn
                original_size = (size[1], size[0])
                if fn:
                    try:
                        result = fn(pred_tensor, original_size=original_size)
                    except TypeError:
                        result = fn(pred_tensor)
                else:
                    result = self._default_seg_postprocess(pred_tensor, size)
            output_filename = self.output_dir / input_filename
            if isinstance(result, Image.Image):
                result.save(output_filename)
            elif isinstance(result, tuple):
                class_id, prob_or_probs = result
                classification_results.append({
                    "filename": input_filename,
                    "class_id": class_id,
                    "prob": prob_or_probs if isinstance(prob_or_probs, (int, float)) else float(np.max(prob_or_probs)),
                })
            elif isinstance(result, (int, float)):
                classification_results.append({"filename": input_filename, "value": float(result)})
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
        if classification_results:
            csv_path = self.output_dir / "predictions.csv"
            all_keys = sorted({k for row in classification_results for k in row})
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
                w.writeheader()
                w.writerows(classification_results)
            self.logger.info(f"Saved {len(classification_results)} prediction rows to {csv_path}")
            self._maybe_print_metric_table(classification_results)
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
        log_memory_cost("Predict", self.logger)

    def _maybe_print_metric_table(self, classification_results: list[dict]) -> None:
        """If config has predict.ground_truth_csv, load labels, compute metrics and print table."""
        c = get_config()
        gt_csv = c.get("predict", {}).get("ground_truth_csv")
        if not gt_csv or not classification_results:
            return
        gt_path = Path(gt_csv)
        if not gt_path.is_absolute():
            gt_path = Path(c.get("output_dir", ".")) / gt_path
        if not gt_path.exists():
            self.logger.warning(f"ground_truth_csv not found: {gt_path}")
            return
        class_labels = c.get("classes", [])
        metric_labels = c.get("metrics", [])
        if not class_labels or not metric_labels:
            return
        num_classes = len(class_labels)
        class_to_idx = {str(l): i for i, l in enumerate(class_labels)}
        with open(gt_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            gt_by_file = {}
            for row in reader:
                fn = row.get("filename") or row.get("file") or row.get("path")
                cls_raw = row.get("class_id") or row.get("class") or row.get("label")
                if fn is None or cls_raw is None:
                    continue
                idx = class_to_idx.get(str(cls_raw))
                if idx is None:
                    try:
                        idx = int(cls_raw)
                    except (ValueError, TypeError):
                        continue
                gt_by_file[fn] = idx
        pred_class_ids = []
        true_class_ids = []
        for row in classification_results:
            fn = row.get("filename")
            if fn is None or fn not in gt_by_file:
                continue
            cid = row.get("class_id")
            if cid is None:
                continue
            if isinstance(cid, (list, np.ndarray)):
                cid = int(np.argmax(cid))
            else:
                cid = int(cid)
            pred_class_ids.append(cid)
            true_class_ids.append(gt_by_file[fn])
        if not pred_class_ids:
            self.logger.warning("No rows aligned with ground_truth_csv; skipping metric table.")
            return
        targets = np.array(true_class_ids, dtype=np.int64)
        pred_onehot = np.zeros((len(pred_class_ids), num_classes), dtype=np.float32)
        pred_onehot[np.arange(len(pred_class_ids)), pred_class_ids] = 1.0
        metric_fns = get_metric_fns(metric_labels)
        metrics_result = many_metrics(
            metric_fns, targets, pred_onehot, class_split=True, class_axis=1
        )
        name_mapping = {}
        for i, fn in enumerate(metric_fns):
            name_mapping[getattr(fn, "__name__", str(fn))] = metric_labels[i]
        mc1_mean = {}
        mc1_std = {}
        mean_scores = {}
        for i, metric_label in enumerate(metric_labels):
            actual_name = name_mapping.get(metric_label, metric_label)
            scores = metrics_result.get(actual_name)
            if scores is None or not hasattr(scores, "__len__"):
                scores = np.array([0.0] * num_classes)
            scores = np.atleast_1d(scores)
            if len(scores) < num_classes:
                scores = np.resize(scores, num_classes)
            mc1_mean[metric_label] = {class_labels[j]: float(scores[j]) for j in range(num_classes)}
            mc1_std[metric_label] = {class_labels[j]: 0.0 for j in range(num_classes)}
            mean_scores[metric_label] = float(np.mean(scores))
        std_scores = {m: 0.0 for m in metric_labels}
        print_metric_scores_table(
            class_labels,
            metric_labels,
            [("Predict", mc1_mean, mc1_std)],
            [("Predict", mean_scores, std_scores)],
            style_key="default",
            title_class="Metric Class Mean Score(Predict)",
            title_summary="Summary of Metric(Predict)",
        )

    def _default_seg_postprocess(self, pred_tensor: torch.Tensor, size: tuple[int, int]) -> Image.Image:
        """Fallback: binary segmentation style (sigmoid, 0.5, grayscale image)."""
        import torch.nn.functional as F
        pred = F.sigmoid(pred_tensor.detach())
        pred = (pred >= 0.5).float()
        pred = pred.squeeze(0).squeeze(0).cpu().numpy()
        pred = (pred * 255).astype(np.uint8)
        img = Image.fromarray(pred, mode="L")
        img = img.resize(size, Image.NEAREST)
        return img

    def _run_model(self, batch_inputs: Any) -> ModelOutput:
        inputs_obj = batch_inputs

        if isinstance(batch_inputs, tuple) and len(batch_inputs) == 2:
            inputs_obj, _ = batch_inputs

        if isinstance(inputs_obj, Mapping):
            model_input = inputs_obj
        else:
            model_input = inputs_obj

        result = self.model(model_input)
        if isinstance(result, tuple):
            outputs_raw = result[0]
        else:
            outputs_raw = result

        if isinstance(outputs_raw, ModelOutput):
            return outputs_raw
        if isinstance(outputs_raw, Mapping):
            return ModelOutput(**outputs_raw)
        return ModelOutput(preds=outputs_raw)
