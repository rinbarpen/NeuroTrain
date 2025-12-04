from __future__ import annotations

import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import copy

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from src.config import get_config
from src.models import get_model
from src.utils import (
    Timer,
    get_transforms,
    load_model,
    log_memory_cost,
    reset_peak_memory_stats,
)
from src.utils.ndict import ModelOutput, Sample


@dataclass(slots=True)
class _ModelCompareEntry:
    alias: str
    model: nn.Module
    device: torch.device
    output_dir: Path


class MultiModelPredictor:
    """
    多模型推理对比器。支持同时在多卡上加载多个模型，并对同一输入进行推理与指标对比。
    """

    def __init__(self, output_dir: Path, compare_conf: dict[str, Any], *, base_config: dict | None = None):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.compare_conf = compare_conf or {}
        self.base_config = base_config or get_config()
        self.logger = logging.getLogger("predict.compare")
        self.timer = Timer()

        self.activation = str(self.compare_conf.get("activation", "sigmoid")).lower()
        self.threshold = self.compare_conf.get("threshold", 0.5)
        self.metrics = [metric.lower() for metric in self.compare_conf.get("metrics", ["mae", "mse"])]
        self.image_mode = self.compare_conf.get("image_mode", "L")
        self.save_difference = self.compare_conf.get("save_difference", True)
        self.save_numpy = self.compare_conf.get("save_numpy", False)
        self.diff_dir = self.output_dir / "diff"
        if self.save_difference:
            self.diff_dir.mkdir(parents=True, exist_ok=True)

        self.transform_override = self.compare_conf.get("transform_config")
        self.reference_alias = self.compare_conf.get("reference")

        self.models: List[_ModelCompareEntry] = self._build_models()
        if not self.models:
            raise ValueError("predict.compare.models 不能为空")

        if not self.reference_alias:
            self.reference_alias = self.models[0].alias
        self.max_workers = int(self.compare_conf.get("max_workers", max(1, len(self.models))))

        self.results: List[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def predict(self, inputs: Sequence[Path], **kwargs: Any) -> None:
        reset_peak_memory_stats()

        transforms = get_transforms(self.transform_override)

        executor = ThreadPoolExecutor(max_workers=self.max_workers) if self.max_workers > 1 else None
        try:
            for idx, input_path in enumerate(inputs, 1):
                input_path = Path(input_path)
                input_filename = input_path.name
                if not input_path.exists():
                    self.logger.warning("输入文件不存在: %s", input_path)
                    continue

                with self.timer.timeit(f"{input_filename}.preprocess"):
                    image = Image.open(input_path).convert(self.image_mode)
                    original_size = image.size
                    image_tensor = transforms(image)
                    if not isinstance(image_tensor, torch.Tensor):
                        image_tensor = torch.as_tensor(np.array(image), dtype=torch.float32)
                    batch_tensor = image_tensor.unsqueeze(0)

                predictions: Dict[str, torch.Tensor] = {}
                latencies: Dict[str, float] = {}

                def _submit(entry: _ModelCompareEntry):
                    return self._run_single_model(entry, batch_tensor)

                if executor:
                    futures = {entry.alias: executor.submit(_submit, entry) for entry in self.models}
                    for alias, future in futures.items():
                        preds, latency = future.result()
                        predictions[alias] = preds
                        latencies[alias] = latency
                else:
                    for entry in self.models:
                        preds, latency = self._run_single_model(entry, batch_tensor)
                        predictions[entry.alias] = preds
                        latencies[entry.alias] = latency

                refer_prob = self._prepare_probability(predictions[self.reference_alias])
                per_model_outputs: Dict[str, Dict[str, Any]] = {}
                metrics = self._compute_metrics(refer_prob, predictions)

                for entry in self.models:
                    prob = self._prepare_probability(predictions[entry.alias])
                    binary = self._apply_threshold(prob)
                    self._save_prediction(entry, binary, original_size, input_filename)
                    per_model_outputs[entry.alias] = {
                        "latency": latencies[entry.alias],
                        "device": str(entry.device),
                    }
                    if self.save_numpy:
                        np_path = entry.output_dir / f"{input_filename}.npy"
                        np.save(np_path, binary.cpu().numpy())

                if self.save_difference:
                    self._save_differences(input_filename, refer_prob, predictions, original_size)

                self.results.append(
                    {
                        "input": input_filename,
                        "metrics": metrics,
                        "per_model": per_model_outputs,
                    }
                )

        finally:
            if executor:
                executor.shutdown(wait=True)

        self._dump_summary()
        total_cost = self.timer.total_elapsed_time()
        self.logger.info("多模型推理耗时 %.2fs，共处理 %d 张图像", total_cost, len(inputs))
        log_memory_cost("MultiPredict", self.logger)

    # ------------------------------------------------------------------ #
    # Internal utilities
    # ------------------------------------------------------------------ #
    def _build_models(self) -> List[_ModelCompareEntry]:
        models_conf = self.compare_conf.get("models") or []
        if not models_conf:
            raise ValueError("predict.compare.models 未配置")

        base_model_block = self.base_config.get("model", {})
        device_pool = self._resolve_device_pool()

        entries: List[_ModelCompareEntry] = []
        for idx, spec in enumerate(models_conf):
            alias = spec.get("alias") or spec.get("label") or spec.get("name") or f"model_{idx}"
            model_section = spec.get("model") or {}
            model_name = spec.get("model_name") or model_section.get("name") or base_model_block.get("name")
            if not model_name:
                raise ValueError(f"模型 {alias} 缺少 model_name 配置")

            raw_config = (
                spec.get("model_config")
                or model_section.get("config")
                or base_model_block.get("config")
                or {}
            )
            model_config = copy.deepcopy(raw_config)

            model = get_model(model_name, model_config)

            checkpoint_path = (
                spec.get("checkpoint")
                or spec.get("weights")
                or spec.get("pretrained")
                or model_section.get("checkpoint")
                or base_model_block.get("pretrained")
            )
            device_str = spec.get("device") or device_pool[idx % len(device_pool)]
            state_device = device_str if "cuda" in device_str else "cpu"
            if checkpoint_path:
                params = load_model(checkpoint_path, map_location=state_device)
                model.load_state_dict(params)
                self.logger.info("加载模型[%s]权重: %s", alias, checkpoint_path)

            torch_device = torch.device(device_str)
            model = model.to(torch_device)
            model.eval()

            model_dir = self.output_dir / alias
            model_dir.mkdir(parents=True, exist_ok=True)
            entries.append(
                _ModelCompareEntry(
                    alias=alias,
                    model=model,
                    device=torch_device,
                    output_dir=model_dir,
                )
            )
        return entries

    def _resolve_device_pool(self) -> List[str]:
        device_conf = self.compare_conf.get("devices")
        if isinstance(device_conf, str):
            parsed = [dev.strip() for dev in device_conf.split(",") if dev.strip()]
            if parsed:
                return parsed
        if isinstance(device_conf, Sequence) and not isinstance(device_conf, str):
            parsed = [str(dev) for dev in device_conf if str(dev).strip()]
            if parsed:
                return parsed

        base_device = self.base_config.get("device", "cuda")
        if isinstance(base_device, str) and "," in base_device:
            return [dev.strip() for dev in base_device.split(",") if dev.strip()]
        return [base_device if isinstance(base_device, str) else "cuda"]

    def _run_single_model(self, entry: _ModelCompareEntry, batch_tensor: torch.Tensor) -> tuple[torch.Tensor, float]:
        start = time.perf_counter()
        inputs = batch_tensor.to(entry.device, non_blocking=True)
        outputs = self._run_model_forward(entry.model, Sample(inputs=inputs))
        pred_tensor = outputs.get("preds")
        if pred_tensor is None or not isinstance(pred_tensor, torch.Tensor):
            raise ValueError(f"模型 {entry.alias} 未返回有效的 preds 张量")

        if entry.device.type == "cuda":
            torch.cuda.synchronize(entry.device)
        latency = time.perf_counter() - start
        return pred_tensor.detach().cpu(), latency

    def _run_model_forward(self, model: nn.Module, batch_inputs: Any) -> ModelOutput:
        inputs_obj = batch_inputs
        if isinstance(batch_inputs, tuple) and len(batch_inputs) == 2:
            inputs_obj, _ = batch_inputs

        if isinstance(inputs_obj, (dict, Sample)):
            model_input = inputs_obj
        else:
            model_input = inputs_obj

        result = model(model_input)
        if isinstance(result, tuple):
            outputs_raw = result[0]
        else:
            outputs_raw = result

        if isinstance(outputs_raw, ModelOutput):
            return outputs_raw
        if isinstance(outputs_raw, dict):
            return ModelOutput(**outputs_raw)
        return ModelOutput(preds=outputs_raw)

    def _prepare_probability(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.clone()
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)
        if tensor.ndim == 3:
            tensor = tensor[0]

        match self.activation:
            case "sigmoid":
                tensor = torch.sigmoid(tensor)
            case "softmax":
                tensor = torch.softmax(tensor, dim=0)
            case _:
                tensor = tensor
        return tensor.float().cpu()

    def _apply_threshold(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.threshold is None:
            return tensor
        return (tensor >= float(self.threshold)).float()

    def _save_prediction(self, entry: _ModelCompareEntry, tensor: torch.Tensor, size: tuple[int, int], filename: str) -> None:
        array = tensor.squeeze().clamp(0, 1).mul(255).byte().numpy()
        image = Image.fromarray(array, mode="L").resize(size)
        image.save(entry.output_dir / filename)

    def _save_differences(
        self,
        filename: str,
        reference: torch.Tensor,
        predictions: Dict[str, torch.Tensor],
        size: tuple[int, int],
    ) -> None:
        ref_tensor = reference.squeeze().float()
        for alias, pred in predictions.items():
            if alias == self.reference_alias:
                continue
            prob = self._prepare_probability(pred)
            diff = torch.abs(ref_tensor - prob).clamp(0, 1)
            diff_img = Image.fromarray((diff.numpy() * 255).astype(np.uint8), mode="L").resize(size)
            diff_dir = self.diff_dir / f"{self.reference_alias}_vs_{alias}"
            diff_dir.mkdir(parents=True, exist_ok=True)
            diff_img.save(diff_dir / filename)

    def _compute_metrics(self, reference: torch.Tensor, predictions: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        results: Dict[str, Dict[str, float]] = {}
        ref_tensor = reference.squeeze().float()
        for alias, pred in predictions.items():
            if alias == self.reference_alias:
                continue
            prob = self._prepare_probability(pred).squeeze().float()
            metrics_for_model: Dict[str, float] = {}
            for metric in self.metrics:
                metrics_for_model[metric] = self._metric_value(metric, ref_tensor, prob)
            results[alias] = metrics_for_model
        return results

    def _metric_value(self, metric: str, ref: torch.Tensor, pred: torch.Tensor) -> float:
        eps = 1e-8
        if metric == "mae":
            return torch.mean(torch.abs(ref - pred)).item()
        if metric == "mse":
            return torch.mean((ref - pred) ** 2).item()
        if metric == "psnr":
            mse = torch.mean((ref - pred) ** 2).item()
            if mse < eps:
                return float("inf")
            return 20 * math.log10(1.0 / math.sqrt(mse))
        if metric == "cosine":
            ref_flat = ref.view(-1)
            pred_flat = pred.view(-1)
            return F.cosine_similarity(ref_flat, pred_flat, dim=0).item()
        if metric == "dice":
            ref_bin = (ref >= float(self.threshold or 0.5)).float()
            pred_bin = (pred >= float(self.threshold or 0.5)).float()
            numerator = 2 * torch.sum(ref_bin * pred_bin)
            denominator = torch.sum(ref_bin) + torch.sum(pred_bin) + eps
            return (numerator / denominator).item()
        self.logger.warning("未知指标 %s，返回0", metric)
        return 0.0

    def _dump_summary(self) -> None:
        summary_path = self.output_dir / "comparison_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

