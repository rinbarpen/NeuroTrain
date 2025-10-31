"""Top-K指标模块

提供各种Top-K评估指标，包括accuracy、precision、recall、f1等。
支持多分类任务中评估模型是否能够在预测的前K个结果中包含正确答案。
"""

from typing import List, Optional, Union
import numpy as np
from functools import wraps, partial

from .utils import _OutputType


def topk_metric(metric_fn, *, use_meter: bool = True):
    """Top-K指标专用装饰器
    
    与标准的@metric装饰器不同，这个装饰器：
    1. 不会flatten y_pred，保持其2D形状（样本数 x 类别数）
    2. 只flatten y_true（因为标签是1D的）
    3. 支持class_split和weights参数
    4. 支持Meter管理
    
    Args:
        metric_fn: 基础指标计算函数
        use_meter: 是否使用Meter来管理scores，默认为True
    """
    @wraps(metric_fn)
    def wrapper(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_split: bool = False,
        weights: Optional[Union[List[float], np.ndarray]] = None,
        *,
        class_axis: int = 1,
        meter_name: Optional[str] = None,
    ) -> _OutputType:
        """
        Parameters
        ----------
        y_true : np.ndarray
            真实标签数组（1D）
        y_pred : np.ndarray
            预测概率数组（2D: 样本数 x 类别数）
        class_split : bool, optional
            是否按类别分别计算, by default False
        weights : List[float]|np.ndarray|None, optional
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
        # 确保y_true是1D的
        if y_true.ndim > 1:
            y_true = y_true.flatten()
        
        # 确保y_pred是2D的
        if y_pred.ndim == 1:
            raise ValueError("y_pred must be 2D array (n_samples, n_classes) for topk metrics")
        
        # 如果指定了权重，强制启用类别分割
        if weights is not None:
            class_split = True
        
        if class_split:
            # 对每个类别分别计算指标
            n_classes = y_pred.shape[class_axis]
            r = []
            
            for class_idx in range(n_classes):
                # 获取该类别的样本
                class_mask = (y_true == class_idx)
                if not np.any(class_mask):
                    r.append(0.0)
                    continue
                
                # 只计算该类别的指标
                class_y_true = y_true[class_mask]
                class_y_pred = y_pred[class_mask]
                
                class_score = metric_fn(class_y_true, class_y_pred)
                r.append(class_score)
            
            if weights is not None:
                result = np.average(r, weights=weights)
            else:
                result = r
        else:
            # 不做类别分割，直接计算整体指标
            result = metric_fn(y_true, y_pred)
        
        # 如果启用了Meter管理，将结果记录到Meter中
        if use_meter and meter_name is not None:
            try:
                # 延迟导入避免循环导入
                from src.recorder.meter import Meter
                
                meter = Meter.instance(meter_name)
                if meter is None:
                    # 创建新的Meter实例
                    meter = Meter(meter_name)
                
                # 根据结果类型更新Meter
                if isinstance(result, list):
                    # 对于多类别结果，记录平均值
                    avg_result = float(np.mean(result))
                    meter.update(avg_result)
                else:
                    meter.update(float(result))
            except ImportError:
                # 如果导入失败，忽略Meter功能
                pass
        
        return result  # type: ignore
    
    return wrapper


def _compute_topk_accuracy(
    y_true: np.ndarray, 
    y_pred_probs: np.ndarray, 
    k: int = 1
) -> float:
    """计算Top-K准确率的核心函数
    
    Args:
        y_true: 真实标签，形状为 (n_samples,)
        y_pred_probs: 预测概率，形状为 (n_samples, n_classes)
        k: Top-K中的K值
        
    Returns:
        Top-K准确率
    """
    # 获取预测概率最高的k个类别的索引
    top_k_preds = np.argsort(y_pred_probs, axis=1)[:, -k:]
    
    # 检查真实标签是否在top-k预测中
    correct = np.any(top_k_preds == y_true.reshape(-1, 1), axis=1)
    
    return np.mean(correct)


def _compute_topk_precision(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    k: int = 1
) -> float:
    """计算Top-K精度的核心函数
    
    对于每个样本，如果真实标签在预测的top-k中，则计为正确。
    精度 = 正确预测的样本数 / 总样本数
    
    Args:
        y_true: 真实标签，形状为 (n_samples,)
        y_pred_probs: 预测概率，形状为 (n_samples, n_classes)
        k: Top-K中的K值
        
    Returns:
        Top-K精度
    """
    top_k_preds = np.argsort(y_pred_probs, axis=1)[:, -k:]
    correct = np.any(top_k_preds == y_true.reshape(-1, 1), axis=1)
    return np.mean(correct)


def _compute_topk_recall(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    k: int = 1
) -> float:
    """计算Top-K召回率的核心函数
    
    对于每个类别，计算该类别的样本中，有多少比例在预测的top-k中被正确识别。
    
    Args:
        y_true: 真实标签，形状为 (n_samples,)
        y_pred_probs: 预测概率，形状为 (n_samples, n_classes)
        k: Top-K中的K值
        
    Returns:
        Top-K召回率（整体平均值）
    """
    top_k_preds = np.argsort(y_pred_probs, axis=1)[:, -k:]
    correct = np.any(top_k_preds == y_true.reshape(-1, 1), axis=1)
    return float(np.mean(correct))


def _compute_topk_f1(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    k: int = 1
) -> float:
    """计算Top-K F1分数的核心函数
    
    基于Top-K的精度和召回率计算F1分数。
    在Top-K场景下，precision和recall通常相等，因此F1等于它们的值。
    
    Args:
        y_true: 真实标签，形状为 (n_samples,)
        y_pred_probs: 预测概率，形状为 (n_samples, n_classes)
        k: Top-K中的K值
        
    Returns:
        Top-K F1分数
    """
    precision = _compute_topk_precision(y_true, y_pred_probs, k)
    recall = _compute_topk_recall(y_true, y_pred_probs, k)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def _compute_topk_per_class(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    k: int = 1,
    metric_type: str = 'accuracy'
) -> np.ndarray:
    """计算每个类别的Top-K指标
    
    Args:
        y_true: 真实标签，形状为 (n_samples,)
        y_pred_probs: 预测概率，形状为 (n_samples, n_classes)
        k: Top-K中的K值
        metric_type: 指标类型，可选 'accuracy', 'precision', 'recall', 'f1'
        
    Returns:
        每个类别的指标值数组
    """
    n_classes = y_pred_probs.shape[1]
    per_class_scores = np.zeros(n_classes)
    
    top_k_preds = np.argsort(y_pred_probs, axis=1)[:, -k:]
    
    for class_idx in range(n_classes):
        # 找到真实标签为当前类别的样本
        class_mask = (y_true == class_idx)
        if not np.any(class_mask):
            per_class_scores[class_idx] = 0.0
            continue
            
        # 检查这些样本的预测是否在top-k中包含正确标签
        class_correct = np.any(
            top_k_preds[class_mask] == class_idx, 
            axis=1
        )
        per_class_scores[class_idx] = np.mean(class_correct)
    
    return per_class_scores


def make_topk_accuracy(k: int):
    """创建指定K值的Top-K准确率指标函数
    
    Args:
        k: Top-K中的K值
        
    Returns:
        带@topk_metric装饰器的Top-K准确率函数
        
    Example:
        >>> topk3_acc = make_topk_accuracy(3)
        >>> result = topk3_acc(y_true, y_pred_probs)
    """
    @topk_metric
    def topk_accuracy_k(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return _compute_topk_accuracy(y_true, y_pred, k)
    topk_accuracy_k.__name__ = f'top{k}_accuracy'
    return topk_accuracy_k


def make_topk_precision(k: int):
    """创建指定K值的Top-K精度指标函数
    
    Args:
        k: Top-K中的K值
        
    Returns:
        带@topk_metric装饰器的Top-K精度函数
    """
    @topk_metric
    def topk_precision_k(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return _compute_topk_precision(y_true, y_pred, k)
    topk_precision_k.__name__ = f'top{k}_precision'
    return topk_precision_k


def make_topk_recall(k: int):
    """创建指定K值的Top-K召回率指标函数
    
    Args:
        k: Top-K中的K值
        
    Returns:
        带@topk_metric装饰器的Top-K召回率函数
    """
    @topk_metric
    def topk_recall_k(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return _compute_topk_recall(y_true, y_pred, k)
    topk_recall_k.__name__ = f'top{k}_recall'
    return topk_recall_k


def make_topk_f1(k: int):
    """创建指定K值的Top-K F1分数指标函数
    
    Args:
        k: Top-K中的K值
        
    Returns:
        带@topk_metric装饰器的Top-K F1函数
    """
    @topk_metric
    def topk_f1_k(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return _compute_topk_f1(y_true, y_pred, k)
    topk_f1_k.__name__ = f'top{k}_f1'
    return topk_f1_k


def topk_accuracy(y_true: np.ndarray, y_pred_probs: np.ndarray, k: int = 1) -> float:
    """计算Top-K准确率（简化版本，不带装饰器）
    
    评估真实标签是否在模型预测概率最高的K个类别中。
    
    Args:
        y_true: 真实标签，形状为 (n_samples,)
        y_pred_probs: 预测概率，形状为 (n_samples, n_classes)
        k: Top-K中的K值，默认为1（即普通准确率）
        
    Returns:
        Top-K准确率分数
        
    Example:
        >>> y_true = np.array([0, 1, 2, 1])
        >>> y_pred = np.array([
        ...     [0.1, 0.3, 0.6],  # 预测为2，真实为0
        ...     [0.2, 0.5, 0.3],  # 预测为1，真实为1
        ...     [0.4, 0.4, 0.2],  # 预测为0或1，真实为2
        ...     [0.3, 0.4, 0.3],  # 预测为1，真实为1
        ... ])
        >>> topk_accuracy(y_true, y_pred, k=1)
        0.5  # 只有第2和第4个样本预测正确
        >>> topk_accuracy(y_true, y_pred, k=2)
        0.75  # 前2个最高概率中，3个样本包含正确答案
    """
    return _compute_topk_accuracy(y_true, y_pred_probs, k)


def topk_precision(y_true: np.ndarray, y_pred_probs: np.ndarray, k: int = 1) -> float:
    """计算Top-K精度（简化版本，不带装饰器）
    
    Args:
        y_true: 真实标签，形状为 (n_samples,)
        y_pred_probs: 预测概率，形状为 (n_samples, n_classes)
        k: Top-K中的K值
        
    Returns:
        Top-K精度分数
    """
    return _compute_topk_precision(y_true, y_pred_probs, k)


def topk_recall(y_true: np.ndarray, y_pred_probs: np.ndarray, k: int = 1) -> float:
    """计算Top-K召回率（简化版本，不带装饰器）
    
    Args:
        y_true: 真实标签，形状为 (n_samples,)
        y_pred_probs: 预测概率，形状为 (n_samples, n_classes)
        k: Top-K中的K值
        
    Returns:
        Top-K召回率分数
    """
    return _compute_topk_recall(y_true, y_pred_probs, k)


def topk_f1(y_true: np.ndarray, y_pred_probs: np.ndarray, k: int = 1) -> float:
    """计算Top-K F1分数（简化版本，不带装饰器）
    
    Args:
        y_true: 真实标签，形状为 (n_samples,)
        y_pred_probs: 预测概率，形状为 (n_samples, n_classes)
        k: Top-K中的K值
        
    Returns:
        Top-K F1分数
    """
    return _compute_topk_f1(y_true, y_pred_probs, k)


# ============================================================================
# 预定义的Top-1指标（等同于标准指标）
# ============================================================================

@topk_metric
def top1_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Top-1准确率（等同于标准准确率）"""
    return _compute_topk_accuracy(y_true, y_pred, k=1)


@topk_metric
def top1_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Top-1精度"""
    return _compute_topk_precision(y_true, y_pred, k=1)


@topk_metric
def top1_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Top-1召回率"""
    return _compute_topk_recall(y_true, y_pred, k=1)


@topk_metric
def top1_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Top-1 F1分数"""
    return _compute_topk_f1(y_true, y_pred, k=1)


# ============================================================================
# 预定义的Top-3指标
# ============================================================================

@topk_metric
def top3_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Top-3准确率"""
    return _compute_topk_accuracy(y_true, y_pred, k=3)


@topk_metric
def top3_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Top-3精度"""
    return _compute_topk_precision(y_true, y_pred, k=3)


@topk_metric
def top3_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Top-3召回率"""
    return _compute_topk_recall(y_true, y_pred, k=3)


@topk_metric
def top3_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Top-3 F1分数"""
    return _compute_topk_f1(y_true, y_pred, k=3)


# ============================================================================
# 预定义的Top-5指标
# ============================================================================

@topk_metric
def top5_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Top-5准确率"""
    return _compute_topk_accuracy(y_true, y_pred, k=5)


@topk_metric
def top5_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Top-5精度"""
    return _compute_topk_precision(y_true, y_pred, k=5)


@topk_metric
def top5_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Top-5召回率"""
    return _compute_topk_recall(y_true, y_pred, k=5)


@topk_metric
def top5_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Top-5 F1分数"""
    return _compute_topk_f1(y_true, y_pred, k=5)


# ============================================================================
# 预定义的Top-10指标
# ============================================================================

@topk_metric
def top10_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Top-10准确率"""
    return _compute_topk_accuracy(y_true, y_pred, k=10)


@topk_metric
def top10_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Top-10精度"""
    return _compute_topk_precision(y_true, y_pred, k=10)


@topk_metric
def top10_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Top-10召回率"""
    return _compute_topk_recall(y_true, y_pred, k=10)


@topk_metric
def top10_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Top-10 F1分数"""
    return _compute_topk_f1(y_true, y_pred, k=10)


# ============================================================================
# 便捷函数：获取所有topk指标
# ============================================================================

def get_topk_metrics(k_values: List[int] = [1, 3, 5, 10]) -> dict:
    """获取指定k值的所有topk指标函数
    
    Args:
        k_values: k值列表
        
    Returns:
        指标名称到函数的映射字典
        
    Example:
        >>> metrics = get_topk_metrics([1, 5])
        >>> metrics.keys()
        dict_keys(['top1_accuracy', 'top1_precision', 'top1_recall', 'top1_f1',
                   'top5_accuracy', 'top5_precision', 'top5_recall', 'top5_f1'])
    """
    metrics = {}
    for k in k_values:
        metrics[f'top{k}_accuracy'] = partial(topk_accuracy, k=k)
        metrics[f'top{k}_precision'] = partial(topk_precision, k=k)
        metrics[f'top{k}_recall'] = partial(topk_recall, k=k)
        metrics[f'top{k}_f1'] = partial(topk_f1, k=k)
    return metrics

