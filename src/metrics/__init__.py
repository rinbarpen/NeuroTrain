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
    alias = {
        # Top-K 常用别名
        "top1": "top1_accuracy",
        "top5": "top5_accuracy",
        "top3": "top3_accuracy",
        "top10": "top10_accuracy",
    }

    resolved = []
    for metric in metrics:
        name = alias.get(metric, metric)
        if name not in globals():
            raise KeyError(f"Metric '{metric}' not found (resolved to '{name}').")
        resolved.append(globals()[name])
    return resolved


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

# Import Top-K metrics
from .topk import (
    # 简化版本（不带装饰器）
    topk_accuracy,
    topk_precision,
    topk_recall,
    topk_f1,
    # 工厂函数
    make_topk_accuracy,
    make_topk_precision,
    make_topk_recall,
    make_topk_f1,
    # 预定义的Top-1指标
    top1_accuracy,
    top1_precision,
    top1_recall,
    top1_f1,
    # 预定义的Top-3指标
    top3_accuracy,
    top3_precision,
    top3_recall,
    top3_f1,
    # 预定义的Top-5指标
    top5_accuracy,
    top5_precision,
    top5_recall,
    top5_f1,
    # 预定义的Top-10指标
    top10_accuracy,
    top10_precision,
    top10_recall,
    top10_f1,
    # 便捷函数
    get_topk_metrics
)

# 分类任务的Top-K指标集合
classification_topk_metrics = get_metric_fns([
    'top1_accuracy',
    'top3_accuracy',
    'top5_accuracy',
    'top10_accuracy',
])
