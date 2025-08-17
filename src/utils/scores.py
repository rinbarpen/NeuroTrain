import numpy as np
from sklearn import metrics

from src.utils.typed import *

# y_true, y_pred: (B, C, X)
def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: ClassLabelsList,
    *,
    class_axis: int = 1,
):
    n_labels = len(labels)

    y_true_flatten = [
        yt.flatten() for yt in np.split(y_true, n_labels, axis=class_axis)
    ]
    y_pred_flatten = [
        yp.flatten() for yp in np.split(y_pred, n_labels, axis=class_axis)
    ]

    result = dict()
    for label, y_true, y_pred in zip(labels, y_true_flatten, y_pred_flatten):
        score = metrics.f1_score(y_true, y_pred)
        result[label] = np.float64(score)

    return result

def recall_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: ClassLabelsList,
    *,
    class_axis: int = 1,
):
    n_labels = len(labels)

    y_true_flatten = [
        yt.flatten() for yt in np.split(y_true, n_labels, axis=class_axis)
    ]
    y_pred_flatten = [
        yp.flatten() for yp in np.split(y_pred, n_labels, axis=class_axis)
    ]

    result = dict()
    for label, y_true, y_pred in zip(labels, y_true_flatten, y_pred_flatten):
        score = metrics.recall_score(y_true, y_pred)
        result[label] = np.float64(score)

    return result

def precision_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: ClassLabelsList,
    *,
    class_axis: int = 1,
):
    n_labels = len(labels)

    y_true_flatten = [
        yt.flatten() for yt in np.split(y_true, n_labels, axis=class_axis)
    ]
    y_pred_flatten = [
        yp.flatten() for yp in np.split(y_pred, n_labels, axis=class_axis)
    ]

    result = dict()
    for label, y_true, y_pred in zip(labels, y_true_flatten, y_pred_flatten):
        score = metrics.precision_score(y_true, y_pred)
        result[label] = np.float64(score)

    return result

def accuracy_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: ClassLabelsList,
    *,
    class_axis: int = 1,
):
    n_labels = len(labels)

    y_true_flatten = [
        yt.flatten() for yt in np.split(y_true, n_labels, axis=class_axis)
    ]
    y_pred_flatten = [
        yp.flatten() for yp in np.split(y_pred, n_labels, axis=class_axis)
    ]

    result = dict()
    for label, y_true, y_pred in zip(labels, y_true_flatten, y_pred_flatten):
        score = metrics.accuracy_score(y_true, y_pred)
        result[label] = np.float64(score)

    return result

def dice_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: ClassLabelsList,
    *,
    class_axis: int = 1,
):
    n_labels = len(labels)

    y_true_flatten = [
        yt.flatten() for yt in np.split(y_true, n_labels, axis=class_axis)
    ]
    y_pred_flatten = [
        yp.flatten() for yp in np.split(y_pred, n_labels, axis=class_axis)
    ]


    result = dict()
    for label, y_true, y_pred in zip(labels, y_true_flatten, y_pred_flatten):
        intersection = np.logical_and(y_true, y_pred).sum()
        union = y_true.sum() + y_pred.sum()
        score = 2 * intersection / union if union > 0 else 0.0
        result[label] = np.float64(score)

    return result

def iou_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: ClassLabelsList,
    *,
    class_axis: int = 1,
    average: str = "binary",
):
    n_labels = len(labels)

    y_true_flatten = [
        yt.flatten() for yt in np.split(y_true, n_labels, axis=class_axis)
    ]
    y_pred_flatten = [
        yp.flatten() for yp in np.split(y_pred, n_labels, axis=class_axis)
    ]

    result = dict()
    for label, y_true, y_pred in zip(labels, y_true_flatten, y_pred_flatten):
        intersection = np.logical_and(y_true, y_pred).sum()
        union = np.logical_or(y_true, y_pred).sum()
        score = intersection / union if union > 0 else 0.0
        result[label] = np.float64(score)

    return result

def scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: ClassLabelsList,
    metric_labels: MetricLabelsList,
    *,
    class_axis: int = 1,
):
    result = {metric: {label: FLOAT() for label in labels} for metric in metric_labels}

    MAP = {
        "iou": iou_score,
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
        "dice": dice_score,
    }

    for metric in metric_labels:
        result[metric] = MAP[metric](
            y_true, y_pred, labels, 
            class_axis=class_axis
        )

    return result

# result template:
# labels = ['0', '1', '2']
# result:
# {'iou': {'0': np.float64(0.3340311765483979),
#          '1': np.float64(0.33356158804182917),
#          '2': np.float64(0.3330714067744889)},
#  'accuracy': {'0': 0.5006504058837891,
#               '1': 0.5006542205810547,
#               '2': 0.4997730255126953},
#  'precision': {'0': np.float64(0.5012290920750281),
#                '1': np.float64(0.49975019164686635),
#                '2': np.float64(0.5001565650394085)},
#  'recall': {'0': np.float64(0.5003410212347635),
#             '1': np.float64(0.5007643214736118),
#             '2': np.float64(0.49925479807124207)},
#  'f1': {'0': np.float64(0.500784662938167),
#         '1': np.float64(0.5002567425950282),
#         '2': np.float64(0.499705274724017)},
#  'dice': {'0': np.float64(0.400502026711725),
#           '1': np.float64(0.4001642983878602),
#           '2': np.float64(0.399811353583824)}}
# result_after:
# {'mean': {'iou': np.float64(0.33355472378823864),
#          'accuracy': np.float64(0.5003592173258463),
#          'precision': np.float64(0.5003786162537677),
#          'recall': np.float64(0.5001200469265391),
#          'f1': np.float64(0.5002488934190708),
#          'dice': np.float64(0.4001592262278031)},
# 'argmax': {'iou': '0',
#            'accuracy': '1',
#            'precision': '0',
#            'recall': '1',
#            'f1': '0',
#            'dice': '0'},
# 'argmin': {'iou': '2',
#            'accuracy': '2',
#            'precision': '1',
#            'recall': '2',
#            'f1': '2',
#            'dice': '2'}}

