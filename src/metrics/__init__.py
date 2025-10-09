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

# Add iou alias for backward compatibility
iou = iou_seg

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

def get_metric_fns(metrics: list[str]) -> list:
    """Get metric functions from their names.

    Args:
        metrics: List of metric names.

    Returns:
        List of metric functions.

    使用示例:
        >>> get_metric_fns(['dice', 'accuracy'])
        [<function dice at 0x...>, <function accuracy at 0x...>]
    """
    return [globals()[metric] for metric in metrics]


detection_metrics = get_metric_fns([
    "iou_bbox",
    "mAP_at_iou_bbox",
    "mF1_at_iou_bbox",
])

seg_metrics = get_metric_fns([
    "dice",
    "accuracy",
    "recall",
    "f1",
    "precision",
    "auc",
    "iou_seg",
    "mAP_at_iou_seg",
    "mF1_at_iou_seg"
])

# Import CLIP metrics
from .clip import (
    clip_accuracy,
    image_retrieval_recall_at_1,
    image_retrieval_recall_at_5,
    text_retrieval_recall_at_1,
    text_retrieval_recall_at_5,
    image_text_similarity,
    zero_shot_classification_accuracy
)

clip_metrics = get_metric_fns([
    'clip_accuracy',
    'image_retrieval_recall_at_1',
    'image_retrieval_recall_at_5',
    'text_retrieval_recall_at_1',
    'text_retrieval_recall_at_5',
    'image_text_similarity',
    'zero_shot_classification_accuracy'
])

# Import BLEU metrics
from .bleu import (
    bleu_1,
    bleu_2,
    bleu_3,
    bleu_4,
    cumulative_bleu,
    corpus_bleu,
    bleu,
    sentence_bleu
)

bleu_metrics = get_metric_fns([
    'bleu_1',
    'bleu_2',
    'bleu_3',
    'bleu_4',
    'cumulative_bleu',
    'corpus_bleu',
    'bleu',
    'sentence_bleu'
])
