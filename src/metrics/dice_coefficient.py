import numpy as np
from .utils import metric, _OutputType

@metric
def dice(y_true: np.ndarray, y_pred: np.ndarray) -> _OutputType:
    intersection = np.logical_and(y_true, y_pred).sum()
    union = y_true.sum() + y_pred.sum()
    score = 2 * intersection / union if union > 0 else 0.0
    return score

dice_coefficient = dice
