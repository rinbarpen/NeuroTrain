from typing import Sequence, Union, List, Optional, TypeVar, Type, Literal
import numpy as np
from functools import wraps


_OutputType = np.float64 | list[np.float64]


def metric(metric_fn):
    """装饰器函数,用于包装各种评估指标
    参数说明见内部wrapper注释
    """
    @wraps(metric_fn)
    def wrapper(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_split: bool = False,
        weights: Sequence[float]|np.ndarray|None = None,
        *,
        class_axis: int = 1,
    ) -> _OutputType:
        """
        Parameters
        ----------
        y_true : np.ndarray
            真实标签数组
        y_pred : np.ndarray
            预测标签数组
        class_split : bool, optional
            是否按类别分别计算, by default False
        weights : Sequence[float]|np.ndarray|None, optional
            各类别权重, by default None
        class_axis : int, optional
            类别所在轴, by default 1

        Returns
        -------
        Union[float, List[float]]
            当class_split=False时返回float,否则返回List[float]
        """
        # 如果指定了权重，强制启用类别分割
        if weights is not None:
            class_split = True
        if class_split:
            # 将类别轴交换到第0轴，然后对每个类别分别计算基础指标
            r = [
                metric_fn(y_hat.flatten(), y.flatten())
                for y_hat, y in zip(
                    y_true.swapaxes(0, class_axis), y_pred.swapaxes(0, class_axis)
                )
            ]
            if weights is not None:
                return np.average(r, weights=weights)
            return r
        # 不做类别分割，直接对整体展开后计算基础指标
        return metric_fn(y_true.flatten(), y_pred.flatten())
    return wrapper


def many_metrics(
    metrics: Sequence[Type],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_split: bool = False,
    weights: Sequence[float]|np.ndarray|None = None,
    *,
    class_axis: int = 1,
) -> dict[str, _OutputType]:
    result = {}
    for metric in metrics:
        # wraps后，metric.__name__为原始函数名
        name = getattr(metric, '__name__', str(metric))
        result[name] = metric(y_true, y_pred, class_split, weights, class_axis=class_axis)
    return result
