from typing import Sequence, Union, List, Optional, TypeVar, Type, Literal
import sklearn.metrics as sk_metrics
import numpy as np

from .utils import metric, _OutputType

@metric
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> _OutputType:
    return sk_metrics.accuracy_score(y_true, y_pred)

@metric 
def recall(y_true: np.ndarray, y_pred: np.ndarray) -> _OutputType:
    return sk_metrics.recall_score(y_true, y_pred)

@metric
def f1(y_true: np.ndarray, y_pred: np.ndarray) -> _OutputType:
    return sk_metrics.f1_score(y_true, y_pred)

@metric
def precision(y_true: np.ndarray, y_pred: np.ndarray) -> _OutputType:
    return sk_metrics.precision_score(y_true, y_pred)

@metric
def auc(y_true: np.ndarray, y_pred: np.ndarray) -> _OutputType:
    return sk_metrics.roc_auc_score(y_true, y_pred)
