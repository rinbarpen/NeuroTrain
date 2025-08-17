"""Metrics utilities for NeuroTrain.

This module provides various metrics for evaluating model performance,
including IoU-based metrics and surface distance metrics.
"""

# Import utility functions and types
from .utils import many_metrics

# Import dice coefficient metrics
from .dice_coefficient import dice, dice_coefficient

# Import normalized surface dice metrics
from .normalized_surface_dice import normalized_surface_dice, nsd

# Import sklearn-based metrics
from .sk import accuracy, recall, f1, precision, auc

# Import IoU-based metrics
from .iou import iou_seg, iou_bbox

from .x_threshold import (
    at_threshold,
    at_accuracy_threshold,
    at_recall_threshold,
    at_precision_threshold,
    at_f1_threshold,
    at_auc_threshold,
    at_iou_threshold_bbox,
    at_iou_threshold_seg,
    mAP_at_iou_bbox,
    mAP_at_iou_seg,
    mF1_at_iou_bbox,
    mF1_at_iou_seg
)


