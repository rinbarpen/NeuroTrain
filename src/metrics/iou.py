import numpy as np
from typing import Literal
from .utils import metric

@metric
def iou_seg(pred: np.ndarray, target: np.ndarray):
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return intersection / union if union > 0 else 0.0

@metric
def iou_bbox(pred: np.ndarray, target: np.ndarray):
    # Ensure coordinates are floats for correct arithmetic
    pred = pred.astype(float)
    target = target.astype(float)
    x1 = max(pred[0], target[0])
    y1 = max(pred[1], target[1])
    x2 = min(pred[2], target[2])
    y2 = min(pred[3], target[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h
    area_pred = max(0.0, pred[2] - pred[0]) * max(0.0, pred[3] - pred[1])
    area_target = max(0.0, target[2] - target[0]) * max(0.0, target[3] - target[1])
    union = area_pred + area_target - intersection
    return intersection / union if union > 0 else 0.0
