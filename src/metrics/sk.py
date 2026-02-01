from typing import Sequence, Union, List, Optional, TypeVar, Type, Literal
import sklearn.metrics as sk_metrics
import numpy as np

from .utils import metric, _OutputType


def _average_for_prf(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Choose average for precision/recall/f1: 'macro' if multiclass else 'binary'."""
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    n_unique = len(np.unique(np.concatenate([yt, yp])))
    return "macro" if n_unique > 2 else "binary"


@metric
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> _OutputType:
    return sk_metrics.accuracy_score(y_true, y_pred)


@metric
def recall(y_true: np.ndarray, y_pred: np.ndarray) -> _OutputType:
    avg = _average_for_prf(y_true, y_pred)
    return sk_metrics.recall_score(y_true, y_pred, average=avg, zero_division=0)


@metric
def f1(y_true: np.ndarray, y_pred: np.ndarray) -> _OutputType:
    avg = _average_for_prf(y_true, y_pred)
    return sk_metrics.f1_score(y_true, y_pred, average=avg, zero_division=0)


@metric
def precision(y_true: np.ndarray, y_pred: np.ndarray) -> _OutputType:
    avg = _average_for_prf(y_true, y_pred)
    return sk_metrics.precision_score(y_true, y_pred, average=avg, zero_division=0)

@metric
def auc(y_true: np.ndarray, y_pred: np.ndarray) -> _OutputType:
    return sk_metrics.roc_auc_score(y_true, y_pred)
