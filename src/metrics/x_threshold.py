import numpy as np
from typing import Literal, Sequence, Type
from .iou import iou_bbox, iou_seg
from .sk import f1, recall, precision, auc, accuracy


def at_threshold(base_metric_fn):
    """
    装饰器：创建一个新的阈值比较函数
    - base_metric_fn: 被@metric装饰后的基础指标函数，应返回形如 (N,) 的每类分数
    - 返回函数形参为 (pred, target, thresholds)，输出为与 thresholds 等长的列表，
      每个元素是形状为 (B, N) 的布尔数组，表示每个样本、每个类别是否达到阈值。
    """
    def threshold_fn(pred: np.ndarray, target: np.ndarray, thresholds: float|Sequence[float] = 0.5) -> list[np.ndarray]:
        if isinstance(thresholds, (float, int)):
            thresholds = [float(thresholds)]
        # 汇总每个 batch 的分数 (B, N)
        batch_scores = []
        for p, t in zip(pred, target):
            # 期望 base_metric_fn 返回 (N,) 的每类分数
            scores = base_metric_fn(p, t, class_split=True, class_axis=0)
            batch_scores.append(np.asarray(scores, dtype=float))
        if len(batch_scores) == 0:
            return [np.zeros((0, 0), dtype=bool) for _ in thresholds]
        x = np.stack(batch_scores, axis=0)  # (B, N)
        return [x >= thr for thr in thresholds]
    return threshold_fn


@at_threshold
def at_iou_threshold_bbox(pred: np.ndarray, target: np.ndarray, **kwargs) -> np.ndarray:
    """
    按类别计算边界框IoU分数，返回 (N,) 数组，供阈值装饰器使用。
    允许透传class_split、class_axis等关键字参数。
    """
    return iou_bbox(pred, target, **kwargs)


@at_threshold
def at_iou_threshold_seg(pred: np.ndarray, target: np.ndarray, **kwargs) -> np.ndarray:
    """
    按类别计算分割IoU分数，返回 (N,) 数组，供阈值装饰器使用。
    允许透传class_split、class_axis等关键字参数。
    """
    return iou_seg(pred, target, **kwargs)


@at_threshold
def at_f1_threshold(pred: np.ndarray, target: np.ndarray, **kwargs) -> np.ndarray:
    """
    按类别计算F1分数，返回 (N,) 数组，供阈值装饰器使用。
    允许透传class_split、class_axis等关键字参数。
    """
    return f1(pred, target, **kwargs)


@at_threshold
def at_recall_threshold(pred: np.ndarray, target: np.ndarray, **kwargs) -> np.ndarray:
    """
    按类别计算Recall分数，返回 (N,) 数组，供阈值装饰器使用。
    允许透传class_split、class_axis等关键字参数。
    """
    return recall(pred, target, **kwargs)


@at_threshold
def at_precision_threshold(pred: np.ndarray, target: np.ndarray, **kwargs) -> np.ndarray:
    """
    按类别计算Precision分数，返回 (N,) 数组，供阈值装饰器使用。
    允许透传class_split、class_axis等关键字参数。
    """
    return precision(pred, target, **kwargs)


@at_threshold
def at_accuracy_threshold(pred: np.ndarray, target: np.ndarray, **kwargs) -> np.ndarray:
    """
    按类别计算Accuracy分数，返回 (N,) 数组，供阈值装饰器使用。
    允许透传class_split、class_axis等关键字参数。
    """
    return accuracy(pred, target, **kwargs)


@at_threshold
def at_auc_threshold(pred: np.ndarray, target: np.ndarray, **kwargs) -> np.ndarray:
    """
    按类别计算AUC分数，返回 (N,) 数组，供阈值装饰器使用。
    允许透传class_split、class_axis等关键字参数。
    """
    return auc(pred, target, **kwargs)


def mF1_at_iou_bbox(pred: np.ndarray, target: np.ndarray, thresholds: float|Sequence[float] = 0.5) -> np.ndarray:
    """
    在给定IoU阈值列表下，基于边界框的每样本、每类别映射计算平均F1。
    返回与thresholds等长的列表，每个元素为该阈值下的平均F1。
    """
    # 获取IoU阈值映射列表 [(B, N), (B, N), ...] 对应每个阈值
    mapping_list = at_iou_threshold_bbox(pred, target, thresholds)
    
    results = []
    # 遍历每个阈值对应的映射
    for mapping in mapping_list:  # mapping 是 (B, N) 布尔数组
        f1_scores = []
        
        # 遍历批次中的每个样本
        for i in range(mapping.shape[0]):  # B
            sample_mapping = mapping[i]  # (N,) 布尔数组
            
            # 计算该样本的检测指标
            # sample_mapping[j] = True 表示第j个检测框的IoU >= 阈值
            tp = sample_mapping.sum()  # True Positives: IoU >= 阈值的检测框数量
            fp = (~sample_mapping).sum()  # False Positives: IoU < 阈值的检测框数量
            
            # 注意：这里的FN计算需要根据具体的数据格式来确定
            # 如果pred和target的框数量不同，需要更复杂的匹配逻辑
            fn = 0  # False Negatives: 未被正确检测的真实框数量
            
            # 计算precision, recall, f1（防止除零）
            precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_val = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0.0
            
            f1_scores.append(f1_val)
        
        # 计算该阈值下的平均F1分数
        avg_f1 = np.mean(f1_scores)
        results.append(avg_f1)
    
    return results


def mF1_at_iou_seg(pred: np.ndarray, target: np.ndarray, thresholds: float|Sequence[float] = 0.5) -> np.ndarray:
    """
    在给定IoU阈值列表下，基于分割IoU的每样本、每类别映射计算平均F1。
    返回与thresholds等长的列表，每个元素为该阈值下的平均F1。
    """
    # 获取IoU阈值映射列表 [(B, N), (B, N), ...] 对应每个阈值
    mapping_list = at_iou_threshold_seg(pred, target, thresholds)
    
    results = []
    # 遍历每个阈值对应的映射
    for mapping in mapping_list:  # mapping 是 (B, N) 布尔数组
        f1_scores = []
        
        # 遍历批次中的每个样本
        for i in range(mapping.shape[0]):  # B
            sample_mapping = mapping[i]  # (N,) 布尔数组
            
            # 计算该样本的分割指标
            # sample_mapping[j] = True 表示第j个类别的IoU >= 阈值
            tp = sample_mapping.sum()  # True Positives: IoU >= 阈值的类别数量
            fp = (~sample_mapping).sum()  # False Positives: IoU < 阈值的类别数量
            fn = target[i].sum() - tp  # False Negatives: 目标中存在但未被正确分割的类别数量
            
            # 计算precision, recall, f1（防止除零）
            precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_val = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0.0
            
            f1_scores.append(f1_val)
        
        # 计算该阈值下的平均F1分数
        avg_f1 = np.mean(f1_scores)
        results.append(avg_f1)
    
    return results


def mAP_at_iou_bbox(pred: np.ndarray, target: np.ndarray, thresholds: float|Sequence[float] = 0.5) -> np.ndarray:
    """
    在给定IoU阈值列表下，基于边界框的每样本、每类别映射计算平均AP（用P-R乘积近似）。
    返回与thresholds等长的列表，每个元素为该阈值下的平均AP。
    """
    # 获取IoU阈值映射列表 [(B, N), (B, N), ...] 对应每个阈值
    mapping_list = at_iou_threshold_bbox(pred, target, thresholds)
    
    results = []
    # 遍历每个阈值对应的映射
    for mapping in mapping_list:  # mapping 是 (B, N) 布尔数组
        ap_scores = []
        # 遍历批次中的每个样本
        for i in range(mapping.shape[0]):  # B
            sample_mapping = mapping[i]  # (N,) 布尔数组
            # 计算该样本的AP
            
            # 计算每个类别的precision和recall
            tp = sample_mapping.sum()  # True Positives
            fp = (~sample_mapping).sum()  # False Positives
            fn = target[i].sum() - tp  # False Negatives
            
            # 计算precision和recall（防止除零）
            precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # 计算AP (用简单乘积近似PR曲线面积)
            ap = precision_val * recall_val
            ap_scores.append(ap)
        # 计算该阈值下的平均AP
        avg_ap = np.mean(ap_scores)
        results.append(avg_ap)
    return results


def mAP_at_iou_seg(pred: np.ndarray, target: np.ndarray, thresholds: float|Sequence[float] = 0.5) -> np.ndarray:
    """
    在给定IoU阈值列表下，基于分割IoU的每样本、每类别映射计算平均AP（用P-R乘积近似）。
    返回与thresholds等长的列表，每个元素为该阈值下的平均AP。
    """
    # 获取IoU阈值映射列表 [(B, N), (B, N), ...] 对应每个阈值
    mapping_list = at_iou_threshold_seg(pred, target, thresholds)
    
    results = []
    # 遍历每个阈值对应的映射
    for mapping in mapping_list:  # mapping 是 (B, N) 布尔数组
        ap_scores = []
        # 遍历批次中的每个样本
        for i in range(mapping.shape[0]):  # B
            sample_mapping = mapping[i]  # (N,) 布尔数组
            # 计算该样本的AP
            
            # 计算每个类别的precision和recall
            tp = sample_mapping.sum()  # True Positives
            fp = (~sample_mapping).sum()  # False Positives
            fn = target[i].sum() - tp  # False Negatives
            
            # 计算precision和recall（防止除零）
            precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # 计算AP (用简单乘积近似PR曲线面积)
            ap = precision_val * recall_val
            ap_scores.append(ap)
        # 计算该阈值下的平均AP
        avg_ap = np.mean(ap_scores)
        results.append(avg_ap)
    return results
