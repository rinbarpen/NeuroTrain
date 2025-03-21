import numpy as np
from sklearn import metrics

from config import ALL_METRIC_LABELS
from utils.typed import *

# y_true, y_pred: (B, C, X)
def f1_score(y_true: np.ndarray, y_pred: np.ndarray, labels: ClassLabelsList, *, class_axis: int=1, average: str='binary'):
    n_labels = len(labels)

    y_true_flatten = [yt.flatten() for yt in np.split(y_true, n_labels, axis=class_axis)]
    y_pred_flatten = [yp.flatten() for yp in np.split(y_pred, n_labels, axis=class_axis)]

    result = dict()
    for (label, y_true, y_pred) in zip(labels, y_true_flatten, y_pred_flatten):
        score = metrics.f1_score(y_true, y_pred, average=average)
        result[label] = np.float64(score)

    return result

def recall_score(y_true: np.ndarray, y_pred: np.ndarray, labels: ClassLabelsList, *, class_axis: int=1, average: str='binary'):
    n_labels = len(labels)

    y_true_flatten = [yt.flatten() for yt in np.split(y_true, n_labels, axis=class_axis)]
    y_pred_flatten = [yp.flatten() for yp in np.split(y_pred, n_labels, axis=class_axis)]

    result = dict()
    for (label, y_true, y_pred) in zip(labels, y_true_flatten, y_pred_flatten):
        score = metrics.recall_score(y_true, y_pred, average=average)
        result[label] = np.float64(score)

    return result

def precision_score(y_true: np.ndarray, y_pred: np.ndarray, labels: ClassLabelsList, *, class_axis: int=1, average: str='binary'):
    n_labels = len(labels)

    y_true_flatten = [yt.flatten() for yt in np.split(y_true, n_labels, axis=class_axis)]
    y_pred_flatten = [yp.flatten() for yp in np.split(y_pred, n_labels, axis=class_axis)]

    result = dict()
    for (label, y_true, y_pred) in zip(labels, y_true_flatten, y_pred_flatten):
        score = metrics.precision_score(y_true, y_pred, average=average)
        result[label] = np.float64(score)

    return result

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray, labels: ClassLabelsList, *, class_axis: int=1, average: str='binary'):
    n_labels = len(labels)

    y_true_flatten = [yt.flatten() for yt in np.split(y_true, n_labels, axis=class_axis)]
    y_pred_flatten = [yp.flatten() for yp in np.split(y_pred, n_labels, axis=class_axis)]

    result = dict()
    for (label, y_true, y_pred) in zip(labels, y_true_flatten, y_pred_flatten):
        score = metrics.accuracy_score(y_true, y_pred)
        result[label] = np.float64(score)

    return result

def dice_score(y_true: np.ndarray, y_pred: np.ndarray, labels: ClassLabelsList, *, class_axis: int=1, average: str='binary'):
    n_labels = len(labels) 

    y_true_flatten = [yt.flatten() for yt in np.split(y_true, n_labels, axis=class_axis)]
    y_pred_flatten = [yp.flatten() for yp in np.split(y_pred, n_labels, axis=class_axis)]

    result = dict()
    for (label, y_true, y_pred) in zip(labels, y_true_flatten, y_pred_flatten):
        intersection = np.logical_and(y_true, y_pred).sum()
        union = np.logical_or(y_true, y_pred).sum()
        score = 2 * intersection / (union + 2 * intersection)
        result[label] = np.float64(score)

    return result

def iou_score(y_true: np.ndarray, y_pred: np.ndarray, labels: ClassLabelsList, *, class_axis: int=1, average: str='binary'):
    n_labels = len(labels)

    y_true_flatten = [yt.flatten() for yt in np.split(y_true, n_labels, axis=class_axis)]
    y_pred_flatten = [yp.flatten() for yp in np.split(y_pred, n_labels, axis=class_axis)]

    result = dict()
    for (label, y_true, y_pred) in zip(labels, y_true_flatten, y_pred_flatten):
        intersection = np.logical_and(y_true, y_pred).sum()
        union = np.logical_or(y_true, y_pred).sum()
        score = intersection / union if union > 0 else 0.0
        result[label] = np.float64(score)

    return result

def scores(y_true: np.ndarray, y_pred: np.ndarray, labels: ClassLabelsList, 
           metric_labels: MetricLabelsList=ALL_METRIC_LABELS, 
           *, class_axis: int=1, average: str='binary'):
    result: MetricClassOneScoreDict = dict()
    result_after: MetricAfterDict = create_MetricAfterDict(labels)

    MAP = {
        'iou': iou_score,
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score,
        'dice': dice_score,
    }
    def score(m: str):
        result[m] = MAP[m](y_true, y_pred, labels, class_axis=class_axis, average=average)
        values = np.array(result[m].values())
        result_after['mean'][m] = values.mean()
        result_after['argmax'][m] = labels[values.argmax()]
        result_after['argmin'][m] = labels[values.argmin()]

    for metric in metric_labels:
        try:
            score(metric)
        except Exception:
            pass

    return result, result_after

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

def dice_loss(y_true: np.ndarray, y_pred: np.ndarray, labels: ClassLabelsList, *, class_axis: int=1, average: str='binary'):
    result = dice_score(y_true, y_pred, labels, class_axis=class_axis, average=average)
    score = np.array(result.values(), dtype=np.float64).mean()
    return -score

def kl_divergence_loss(y_true: np.ndarray, y_pred: np.ndarray, *, epsilon: float=1e-7):
    """计算 KL 散度损失
    
    Args:
        y_true: 真实概率分布 (B, C, ...)
        y_pred: 预测概率分布 (B, C, ...)
        epsilon: 数值稳定性的小量
        
    Returns:
        float: KL 散度损失值
        
    Example:
        >>> # 假设有批次大小为2，3个类别的预测
        >>> y_true = np.array([[[0.7, 0.2, 0.1], [0.3, 0.6, 0.1]]])  # (1, 2, 3)
        >>> y_pred = np.array([[[0.6, 0.3, 0.1], [0.4, 0.5, 0.1]]])  # (1, 2, 3)
        >>> loss = kl_divergence_loss(y_true, y_pred)
    """
    # 确保输入为概率分布
    y_true = np.clip(y_true, epsilon, 1.0 - epsilon)
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    
    # 计算 KL 散度: sum(p * log(p/q))
    kl_div = np.sum(y_true * np.log(y_true / y_pred), axis=1)
    return np.mean(kl_div)


if __name__ == '__main__':
    n_batches = 2
    n_classes = 3
    labels = [str(i) for i in range(n_classes)]
    y_true = np.random.rand(n_batches, n_classes, 512, 512)
    y_pred = np.random.rand(n_batches, n_classes, 512, 512)
    y_true[y_true >= 0.5] = 1
    y_true[y_true < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0

    result = scores(y_true, y_pred, labels)
    from pprint import pp
    pp(result)

# TODO: 添加 kl loss 和 soft loss and hard loss