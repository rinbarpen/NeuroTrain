from typing import Sequence, Union, List, Optional, TypeVar, Type, Literal
import numpy as np
from functools import wraps


_OutputType = Union[np.float64, List[np.float64], np.ndarray, float]


def metric(metric_fn, *, use_meter: bool = True):
    """装饰器函数,用于包装各种评估指标
    参数说明见内部wrapper注释
    
    Args:
        metric_fn: 基础指标计算函数
        use_meter: 是否使用Meter来管理scores，默认为True
    """
    @wraps(metric_fn)
    def wrapper(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_split: bool = False,
        weights: Optional[Union[Sequence[float], np.ndarray]] = None,
        *,
        class_axis: int = 1,
        meter_name: Optional[str] = None,
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
        meter_name : str, optional
            Meter实例的名称，用于管理scores历史记录

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
                result = np.average(r, weights=weights)
            else:
                result = r
        else:
            # 不做类别分割，直接对整体展开后计算基础指标
            result = metric_fn(y_true.flatten(), y_pred.flatten())
        
        # 如果启用了Meter管理，将结果记录到Meter中
        if use_meter and meter_name is not None:
            # 延迟导入避免循环导入
            from src.recorder.meter import Meter
            
            meter = Meter.instance(meter_name)
            if meter is None:
                # 创建新的Meter实例
                meter = Meter(meter_name)
            
            # 根据结果类型更新Meter
            if isinstance(result, list):
                # 对于多类别结果，记录平均值
                avg_result = np.mean(result)
                meter.update(avg_result)
            else:
                meter.update(result)
        
        return result
    return wrapper


def many_metrics(
    metrics: Sequence[Type],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_split: bool = False,
    weights: Optional[Union[Sequence[float], np.ndarray]] = None,
    *,
    class_axis: int = 1,
    use_meter: bool = True,
    meter_prefix: Optional[str] = None,
) -> dict[str, _OutputType]:
    """
    计算多个指标并可选地使用Meter管理scores
    
    Args:
        metrics: 指标函数列表
        y_true: 真实标签数组
        y_pred: 预测标签数组
        class_split: 是否按类别分别计算
        weights: 各类别权重
        class_axis: 类别所在轴
        use_meter: 是否使用Meter来管理scores
        meter_prefix: Meter名称前缀，如果为None则不使用Meter
    
    Returns:
        dict: 指标名称到结果的映射
    """
    result = {}
    for metric in metrics:
        # wraps后，metric.__name__为原始函数名
        name = getattr(metric, '__name__', str(metric))
        
        # 构造meter名称
        meter_name = None
        if use_meter and meter_prefix is not None:
            meter_name = f"{meter_prefix}_{name}"
        
        # 调用指标函数，传递meter_name参数
        result[name] = metric(
            y_true, 
            y_pred, 
            class_split, 
            weights, 
            class_axis=class_axis,
            meter_name=meter_name
        )
    return result


def get_meters_by_prefix(prefix: str) -> List:
    """
    根据前缀获取所有匹配的 Meter 实例。
    
    Args:
        prefix: Meter 名称的前缀
        
    Returns:
        匹配的 Meter 实例列表
    """
    # 延迟导入避免循环导入
    from src.recorder.meter import Meter, _meter_manager
    
    return [meter for name, meter in _meter_manager.items() if name.startswith(prefix)]


def reset_meters_by_prefix(prefix: str) -> None:
    """
    重置所有匹配前缀的 Meter 实例。
    
    Args:
        prefix: Meter 名称的前缀
    """
    # 延迟导入避免循环导入
    from src.recorder.meter import Meter, _meter_manager
    
    for name, meter in _meter_manager.items():
        if name.startswith(prefix):
            meter.reset()


def get_meter_summary_by_prefix(prefix: str) -> dict:
    """
    获取所有匹配前缀的 Meter 实例的摘要信息。
    
    Args:
        prefix: Meter 名称的前缀
        
    Returns:
        包含 Meter 摘要信息的字典
    """
    # 延迟导入避免循环导入
    from src.recorder.meter import Meter, _meter_manager
    
    summary = {}
    for name, meter in _meter_manager.items():
        if name.startswith(prefix):
            summary[name] = {
                'count': meter.count,
                'sum': meter.sum,
                'avg': meter.avg
            }
    return summary
