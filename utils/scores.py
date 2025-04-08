import numpy as np
from sklearn import metrics
from pathlib import Path
import wandb

from config import get_config, ALL_METRIC_LABELS
from utils.typed import *
from utils.recorder import Recorder
from utils.painter import Plot
from utils.util import get_logger

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

class ScoreCalculator:
    def __init__(self, class_labels: ClassLabelsList, metric_labels: MetricLabelsList=ALL_METRIC_LABELS, *, logger=None):
        self.logger = logger if logger is not None else get_logger()
        self.class_labels = class_labels
        self.metric_labels = metric_labels
        self.is_prepared = False

        # {'recall': {
        #   '0': [], '1': []},
        #  'precision: {
        #   '0': [], '1': []}}
        self.record_metrics: MetricClassManyScoreDict = {metric_label: {class_label: []} for metric_label in metric_labels for class_label in class_labels}
        # but epoch_metrics is saved by epoch
        self.epoch_metrics: MetricClassManyScoreDict = {metric_label: {class_label: []} for metric_label in metric_labels for class_label in class_labels}

        # 
        # {'label': {'f1': [], 
        #         'recall': []}}
        # 
        self.all_record: ClassMetricManyScoreDict = {class_label: {metric: [] for metric in metric_labels} for class_label in class_labels}
        # 
        # {'label': {'f1': mean score, 
        #         'recall': mean score}}
        # 
        self.mean_record: ClassMetricOneScoreDict = {class_label: {metric: 0.0 for metric in metric_labels} for class_label in class_labels}
        # 
        # {'f1': {'label': mean_score}}
        # 
        self.metric_label_record: MetricClassOneScoreDict = {metric: {label: 0.0 for label in class_labels} for metric in metric_labels}
        # 
        # {'f1': mean_label_score}
        # 
        self.metric_record: MetricLabelOneScoreDict = {metric: 0.0 for metric in metric_labels}


    def add_one_batch(self, targets: np.ndarray, outputs: np.ndarray):
        metrics, _ = scores(targets, outputs, self.class_labels, metric_labels=self.metric_labels)
        for metric_label, label_scores in metrics.items():
            for label, score in label_scores.items():
                self.record_metrics[metric_label][label].append(score)

    def finish_one_epoch(self):
        for metric_label, label_scores in self.record_metrics.items():
            for label, scores in label_scores.items():
                self.epoch_metrics[metric_label][label].append(np.mean(scores))


    def record_batches(self, output_dir: Path):
        self._prepare(output_dir)

        for label, metrics in self.mean_record.items():
            label_dir = output_dir / label
            Recorder.record_mean_metrics(metrics, label_dir, logger=self.logger)
        Recorder.record_metrics(self.metric_record, output_dir, logger=self.logger)

        # paint mean metrics for all classes
        # n = len(self.class_labels)
        # nrows, ncols = (n + 2) // 3, 3 if n >= 3 else 1, n
        # Plot(nrows, ncols).metrics(self.metric_record).save(output_dir / "mean-metrics.png")
        Plot(1, 1).subplot().many_metrics(self.mean_record).complete().save(output_dir / "mean-metrics.png")

        c = get_config()
        if c['private']['wandb']:
            wandb.log({
                'metric': {
                    'score': {
                        'all': self.all_record,
                        'mean': self.mean_record,
                    },
                    'image': {
                        'mean': output_dir / "mean-metrics.png",
                    }
                },
            })

    def record_epochs(self, output_dir: Path, n_epochs: int):
        self._prepare(output_dir)

        epoch_mean_metrics: dict[str, dict[str, list]] = {class_label: {metric: [] for metric in self.metric_labels} for class_label in self.class_labels}
        for metric_name, label_scores in self.epoch_metrics.items():
            for label, scores in label_scores.items():
                epoch_mean_metrics[label][metric_name] = scores

        for label, metrics in epoch_mean_metrics.items():
            label_dir = output_dir / label
            Recorder.record_all_metrics(metrics, label_dir, logger=self.logger)
        for label, metrics in self.mean_record.items():
            label_dir = output_dir / label
            Recorder.record_mean_metrics(metrics, label_dir, logger=self.logger)
        Recorder.record_metrics(self.metric_record, output_dir, logger=self.logger)

        # paint metrics curve for all classes in one figure
        n = len(self.metric_labels)
        nrows, ncols = ((n + 2) // 3, 3) if n > 3 else (1, n)
        plot = Plot(nrows, ncols)
        for metric in self.metric_labels:
            plot.subplot().many_epoch_metrics(n_epochs, self.epoch_metrics[metric], self.class_labels, title=metric).complete()
        plot.save(output_dir / "epoch-metrics.png")

        # paint mean metrics for all classes
        # n = len(self.class_labels)
        # nrows, ncols = ((n + 2) // 3, 3) if n > 3 else (1, n)
        Plot(1, 1).subplot().many_metrics(self.mean_record).complete().save(output_dir / "mean-metrics.png")

        c = get_config()
        if c['private']['wandb']:
            wandb.log({
                'metric': {
                    'score': {
                        'all': self.all_record,
                        'mean': self.mean_record,
                    },
                    'image': {
                        'mean': output_dir / "mean-metrics.png",
                        'epoch': output_dir / "epoch-metrics.png",
                    }
                },
            })

    def _prepare(self, output_dir: Path):
        if self.is_prepared:
            return

        for metric_name, label_scores in self.record_metrics.items():
            for label, scores in label_scores.items():
                self.all_record[label][metric_name] = scores
        for label, metrics in self.all_record.items():
            for metric_name, scores in metrics.items():
                self.mean_record[label][metric_name] = np.mean(scores)
        for metric_name, label_scores in self.record_metrics.items():
            for label, scores in label_scores.items():
                self.metric_label_record[metric_name][label] = np.mean(scores)
        for metric_name, label_scores in self.record_metrics.items():
            mean_scores = []
            for label, scores in label_scores.items():
                mean_scores.append(np.mean(scores))
            self.metric_record[metric_name] = np.mean(mean_scores)
        
        for label in self.class_labels:
            label_dir = output_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)
        self.is_prepared = True


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