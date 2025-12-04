from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional
import logging


@dataclass
class _LossAggregationBuffer:
    """缓存单个stage内的loss分量，用于epoch级统计。"""

    sums: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def reset(self):
        self.sums.clear()
        self.counts.clear()

    def update(self, losses: Dict[str, float]):
        for name, value in losses.items():
            if value is None:
                continue
            self.sums[name] += value
            self.counts[name] += 1

    def average(self) -> Dict[str, float]:
        result = {}
        for name, total in self.sums.items():
            count = self.counts.get(name, 0)
            if count > 0:
                result[name] = total / count
        return result


class LossTracker:
    """
    负责记录训练/验证阶段的各类loss分量（step级 + epoch级）。
    数据将通过DataSaver异步写入CSV/Parquet，路径由TrainOutputFilenameEnv提供。
    """

    def __init__(self, saver, filename_env, logger: Optional[logging.Logger] = None):
        self.saver = saver
        self.filename_env = filename_env
        self.logger = logger or logging.getLogger(__name__)
        self._buffers: Dict[str, _LossAggregationBuffer] = defaultdict(_LossAggregationBuffer)

    # ---- epoch lifecycle -------------------------------------------------
    def begin_epoch(self, stage: str):
        self._buffers[stage].reset()

    def finish_epoch(self, stage: str, epoch: int):
        summary = self._buffers[stage].average()
        if not summary:
            return
        record = {"epoch": int(epoch)}
        record.update(summary)
        filename = self._resolve_filename(stage, scope="epoch")
        if filename is None:
            self.logger.debug("[LossTracker] 无法找到stage=%s的epoch文件路径", stage)
            return
        self.saver.save_loss_components(record, filename)

    # ---- step recording --------------------------------------------------
    def record_step(self, stage: str, epoch: int, step: int, global_step: int, losses: Dict[str, float]):
        if not losses:
            return
        record = {
            "epoch": int(epoch),
            "step": int(step),
            "global_step": int(global_step),
        }
        record.update(losses)
        filename = self._resolve_filename(stage, scope="step")
        if filename is None:
            self.logger.debug("[LossTracker] 无法找到stage=%s的step文件路径", stage)
            return
        self.saver.save_loss_components(record, filename)
        self._buffers[stage].update(losses)

    # ---- helpers ---------------------------------------------------------
    def _resolve_filename(self, stage: str, scope: str) -> Optional[str]:
        try:
            if stage == "train" and scope == "step":
                return self.filename_env.output_train_step_loss_components_filename.as_posix()
            if stage == "train" and scope == "epoch":
                return self.filename_env.output_train_epoch_loss_components_filename.as_posix()
            if stage == "valid" and scope == "step":
                return self.filename_env.output_valid_step_loss_components_filename.as_posix()
            if stage == "valid" and scope == "epoch":
                return self.filename_env.output_valid_epoch_loss_components_filename.as_posix()
        except AttributeError:
            self.logger.debug("[LossTracker] filename_env缺少stage=%s scope=%s 对应的属性", stage, scope)
        return None

