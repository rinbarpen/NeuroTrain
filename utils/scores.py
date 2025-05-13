import logging
import numpy as np
from sklearn import metrics
from pathlib import Path
import wandb

from config import get_config
from utils.typed import *
from utils.painter import Plot
from utils.data_saver import DataSaver
from utils.typed import convert_to_ClassLabelManyScoreDict

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

def scores(y_true: np.ndarray, y_pred: np.ndarray, 
           labels: ClassLabelsList, metric_labels: MetricLabelsList, 
           *, class_axis: int=1, average: str='binary'):
    result: MetricClassOneScoreDict = create_MetricClassOneScoreDict(metric_labels, labels)
    result_after: MetricAfterDict = create_MetricAfterDict(labels)

    MAP = {
        'iou': iou_score,
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score,
        'dice': dice_score,
    }

    for metric in metric_labels:
        result[metric] = MAP[metric](y_true, y_pred, labels, class_axis=class_axis, average=average)
        values = np.array(list(result[metric].values()))
        result_after['mean'][metric] = values.mean()
        result_after['std'][metric] = values.std()
        result_after['argmax'][metric] = labels[values.argmax()]
        result_after['argmin'][metric] = labels[values.argmin()]

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

def dice_loss(y_true: np.ndarray, y_pred: np.ndarray, *, class_axis: int=1, average: str='binary'):
    labels = [str(i) for i in range(y_true.shape[class_axis])]
    result = dice_score(y_true, y_pred, labels, class_axis=class_axis, average=average)
    score = np.array(result.values(), dtype=np.float64).mean()
    return -score

def kl_divergence_loss(y_true: np.ndarray, y_pred: np.ndarray, *, epsilon: float=1e-7):
    # 确保输入为概率分布
    y_true = np.clip(y_true, epsilon, 1.0 - epsilon)
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    
    # 计算 KL 散度: sum(p * log(p/q))
    kl_div = np.sum(y_true * np.log(y_true / y_pred), axis=1)
    return np.mean(kl_div)

class ScoreCalculator:
    def __init__(self, class_labels: ClassLabelsList, metric_labels: MetricLabelsList, *, logger=None, saver: DataSaver):
        self.logger = logger or logging.getLogger()
        self.saver = saver
        self.class_labels = class_labels
        self.metric_labels = metric_labels
        self.is_prepared = False

        # {'recall': {
        #   '0': [], '1': []},
        #  'precision: {
        #   '0': [], '1': []}}
        self.all_metric_label_scores: MetricClassManyScoreDict = create_MetricClassManyScoreDict(self.metric_labels, self.class_labels)
        # but epoch_metric_label_scores is saved by epoch
        self.epoch_metric_label_scores: MetricClassManyScoreDict = create_MetricClassManyScoreDict(self.metric_labels, self.class_labels)

        # 
        # {'label': {'f1': [], 
        #         'recall': []}}
        # 
        self.all_record: ClassMetricManyScoreDict = create_ClassMetricManyScoreDict(self.metric_labels, self.class_labels)
        # 
        # {'label': {'f1': mean score, 
        #         'recall': mean score}}
        # 
        self.mean_record: ClassMetricOneScoreDict = create_ClassMetricOneScoreDict(self.metric_labels, self.class_labels)
        # 
        # {'f1': {'label': mean_score}}
        # 
        self.metric_label_record: MetricClassOneScoreDict = create_MetricClassOneScoreDict(self.metric_labels, self.class_labels)
        # 
        # {'f1': mean_label_score}
        # 
        self.metric_record: MetricLabelOneScoreDict = create_MetricLabelOneScoreDict(self.metric_labels)

    def clear(self):
        self.is_prepared = False

        # {'recall': {
        #   '0': [], '1': []},
        #  'precision: {
        #   '0': [], '1': []}}
        self.all_metric_label_scores: MetricClassManyScoreDict = create_MetricClassManyScoreDict(self.metric_labels, self.class_labels)
        # but epoch_metric_label_scores is saved by epoch
        self.epoch_metric_label_scores: MetricClassManyScoreDict = create_MetricClassManyScoreDict(self.metric_labels, self.class_labels)

        # 
        # {'label': {'f1': [], 
        #         'recall': []}}
        # 
        self.all_record: ClassMetricManyScoreDict = create_ClassMetricManyScoreDict(self.metric_labels, self.class_labels)
        # 
        # {'label': {'f1': mean score, 
        #         'recall': mean score}}
        # 
        self.mean_record: ClassMetricOneScoreDict = create_ClassMetricOneScoreDict(self.metric_labels, self.class_labels)
        # 
        # {'label': {'f1': std score, 
        #         'recall': std score}}
        # 
        self.std_record: ClassMetricOneScoreDict = create_ClassMetricOneScoreDict(self.metric_labels, self.class_labels)
        # 
        # {'f1': {'label': mean_score}}
        # 
        self.metric_label_record: MetricClassOneScoreDict = create_MetricClassOneScoreDict(self.metric_labels, self.class_labels)
        # 
        # {'f1': mean_label_score}
        # 
        self.metric_record: MetricLabelOneScoreDict = create_MetricLabelOneScoreDict(self.metric_labels)
        # 
        # {'f1': std_label_score}
        # 
        self.metric_record_std: MetricLabelOneScoreDict = create_MetricLabelOneScoreDict(self.metric_labels)

    def add_one_batch(self, targets: np.ndarray, outputs: np.ndarray):
        metrics, metrics_after = scores(targets, outputs, labels=self.class_labels, metric_labels=self.metric_labels)

        for metric_label in self.metric_labels:
            for class_label in self.class_labels:
                score = metrics[metric_label][class_label]
                self.all_metric_label_scores[metric_label][class_label].append(score)

        return metrics, metrics_after

    def finish_one_epoch(self):
        for metric_label in self.metric_labels:
            for class_label in self.class_labels:
                score = np.mean(self.all_metric_label_scores[metric_label][class_label])
                self.epoch_metric_label_scores[metric_label][class_label].append(score)
        return self.epoch_metric_label_scores

    def record_batches(self, output_dir: Path):
        self._prepare(output_dir)
        mean_metrics_image = output_dir / "mean_metrics.png"

        self.saver.save_all_metric_by_class(self.all_metric_label_scores)
        self.saver.save_mean_metric_by_class(self.mean_record)
        self.saver.save_mean_metric(self.metric_record)

        # paint mean metrics for all classes
        Plot(1, 1).subplot().many_metrics(self.mean_record).complete().save(mean_metrics_image)

        c = get_config()
        if c['private']['wandb']:
            wandb.log({
                'metric': {
                    'score': {
                        'all': self.all_record,
                        'mean': self.mean_record,
                    },
                    'image': {
                        'mean': mean_metrics_image,
                    },
                },
            })

    def record_epochs(self, output_dir: Path, n_epochs: int):
        self._prepare(output_dir)

        self.saver.save_all_metric_by_class(convert_to_ClassLabelManyScoreDict(self.epoch_metric_label_scores))
        self.saver.save_mean_metric_by_class(self.mean_record)
        self.saver.save_mean_metric(self.metric_record)

        epoch_metrics_image = output_dir / "epoch_metrics.png"
        mean_metrics_image = output_dir / "mean_metrics.png"

        # paint metrics curve for all classes in one figure
        n = len(self.metric_labels)
        if n > 4:
            nrows, ncols = (n + 2) // 3, 3
        elif n == 4:
            nrows, ncols = 2, 2
        else:
            nrows, ncols = 1, n

        plot = Plot(nrows, ncols)
        for metric in self.metric_labels:
            plot.subplot().many_epoch_metrics(n_epochs, self.epoch_metric_label_scores[metric], self.class_labels, title=metric).complete()
        plot.save(epoch_metrics_image)

        # paint mean metrics for all classes
        Plot(1, 1).subplot().many_metrics(self.mean_record).complete().save(mean_metrics_image)

        c = get_config()
        if c['private']['wandb']:
            wandb.log({
                'metric': {
                    'score': {
                        'all': self.all_record,
                        'mean': self.mean_record,
                    },
                    'image': {
                        'epoch': epoch_metrics_image,
                        'mean': mean_metrics_image,
                    }
                },
            })

    def _prepare(self, output_dir: Path):
        if self.is_prepared:
            return

        for metric_label in self.metric_labels:
            metric_mean_score = []
            for class_label in self.class_labels:
                scores = self.all_metric_label_scores[metric_label][class_label]
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                self.all_record[class_label][metric_label] = scores
                self.mean_record[class_label][metric_label] = mean_score
                self.std_record[class_label][metric_label] = std_score
                self.metric_label_record[metric_label][class_label] = mean_score
                metric_mean_score.append(mean_score)
            self.metric_record[metric_label] = np.mean(metric_mean_score)
            self.metric_record_std[metric_label] = np.std(metric_mean_score)

        for class_label in self.class_labels:
            class_dir = output_dir / class_label
            class_dir.mkdir(parents=True, exist_ok=True)
        self.is_prepared = True

from utils.db import ScoreDB
import os.path
from shutil import copyfile
import uuid
from copy import deepcopy
class _ScoreCalculator:
    def __init__(self, db_filepath: str, class_labels: ClassLabelsList, metric_labels: MetricLabelsList, *, logger=None, saver: DataSaver, epoch_mode=False):
        self.logger = logger or logging.getLogger()
        self.saver = saver
        self.class_labels = class_labels
        self.metric_labels = metric_labels
        self.is_prepared = False
        self.epoch_mode = epoch_mode

        self.temp_db = ScoreDB(f"TEMP/{uuid.uuid1()}.db")
        self.db = ScoreDB(db_filepath)

    def finish_one_batch(self, targets: np.ndarray, outputs: np.ndarray):
        metrics, metrics_after = scores(targets, outputs, labels=self.class_labels, metric_labels=self.metric_labels)

        for metric_label in self.metric_labels:
            for class_label in self.class_labels:
                score = metrics[metric_label][class_label]
                if self.epoch_mode:
                    self.db.add(metric_label, class_label, score)
                else:
                    self.temp_db.add(metric_label, class_label, score)

        return metrics, metrics_after

    def finish_one_epoch(self):
        scores = self.temp_db.metric_class_scores(self.metric_labels, self.class_labels)
        self.temp_db.clear()
        for metric_label in self.metric_labels:
            for class_label in self.class_labels:
                score = np.mean(scores[metric_label][class_label])
                self.db.add(metric_label, class_label, score)

    def record_batches(self, output_dir: Path):
        mean_metrics_image = output_dir / "mean_metrics.png"

        all_scores = self.db.class_metric_scores(self.metric_labels, self.class_labels, mode='all')
        mean_scores = self.db.class_metric_scores(self.metric_labels, self.class_labels, mode='mean')
        std_scores = self.db.class_metric_scores(self.metric_labels, self.class_labels, mode='std')
        mean_metrics = self.db.metric_scores(self.metric_labels, mode='mean') 
        std_metrics = self.db.metric_scores(self.metric_labels, mode='std')
        self.saver.save_all_metric_by_class(all_scores)
        self.saver.save_mean_metric_by_class(mean_scores)
        self.saver.save_std_metric_by_class(std_scores)
        self.saver.save_mean_metric(mean_metrics)
        self.saver.save_std_metric(std_metrics)

        # paint mean metrics for all classes
        scores = self.db.class_metric_scores(self.metric_labels, self.class_labels, mode='mean')
        Plot(1, 1).subplot().metrics(scores).complete().save(mean_metrics_image)
        for label, metric_scores in scores:
            metric_image = output_dir / label / "metrics.png"
            metric_image.parent.mkdir(exist_ok=True)
            Plot(1, 1).subplot().metrics(metric_scores).complete().save(metric_image)

    def record_epochs(self, output_dir: Path, n_epochs: int):
        all_scores = self.db.class_metric_scores(self.metric_labels, self.class_labels, mode='all')
        mean_scores = self.db.class_metric_scores(self.metric_labels, self.class_labels, mode='mean')
        std_scores = self.db.class_metric_scores(self.metric_labels, self.class_labels, mode='std')
        mean_metrics = self.db.metric_scores(self.metric_labels, mode='mean') 
        std_metrics = self.db.metric_scores(self.metric_labels, mode='std')
        self.saver.save_all_metric_by_class(all_scores)
        self.saver.save_mean_metric_by_class(mean_scores)
        self.saver.save_std_metric_by_class(std_scores)
        self.saver.save_mean_metric(mean_metrics)
        self.saver.save_std_metric(std_metrics)

        mean_metrics_image = output_dir / "mean_metrics.png"
        epoch_metrics_image = output_dir / "epoch_metrics.png"

        n = len(self.metric_labels)
        if n > 4:
            nrows, ncols = (n + 2) // 3, 3
        elif n == 4:
            nrows, ncols = 2, 2
        else:
            nrows, ncols = 1, n

        scores = self.db.metric_class_scores(self.metric_labels, self.class_labels, mode='all')
        plot = Plot(nrows, ncols)
        for metric in self.metric_labels:
            plot.subplot().many_epoch_metrics(n_epochs, scores[metric], self.class_labels, title=metric).complete()
        plot.save(epoch_metrics_image)

        # paint mean metrics for all classes
        scores = self.db.class_metric_scores(self.metric_labels, self.class_labels, mode='mean')
        Plot(1, 1).subplot().metrics(scores).complete().save(mean_metrics_image)
        for label, metric_scores in scores:
            metric_image = output_dir / label / "metrics.png"
            metric_image.parent.mkdir(exist_ok=True)
            Plot(1, 1).subplot().metrics(metric_scores).complete().save(metric_image)

# class _ScoreCalculator:
#     def __init__(self, class_labels: ClassLabelsList, metric_labels: MetricLabelsList, *, logger=None, saver: DataSaver):
#         self.logger = logger or logging.getLogger()
#         self.saver = saver
#         self.class_labels = class_labels
#         self.metric_labels = metric_labels
#         self.is_prepared = False

#         self.all_batch_scores = MetricClassMap()
#         self.all_epoch_scores = MetricClassMap()

#     def add_one_batch(self, targets: np.ndarray, outputs: np.ndarray):
#         metrics, metrics_after = scores(targets, outputs, labels=self.class_labels, metric_labels=self.metric_labels)

#         for metric_label in self.metric_labels:
#             for class_label in self.class_labels:
#                 score = metrics[metric_label][class_label]
#                 self.all_batch_scores.add(metric_label, class_label, score)

#         return metrics, metrics_after

#     def finish_one_epoch(self):
#         for unit in self.all_batch_scores.units:
#             self.all_epoch_scores.add(unit.metric_label, unit.class_label, unit.mean())
#         self.all_batch_scores = MetricClassMap()
#         return self.all_epoch_scores

#     def record_batches(self, output_dir: Path):
#         self._prepare(output_dir)
#         mean_metrics_image = output_dir / "mean_metrics.png"

#         self.saver.save_all_metric_by_class(self.all_metric_label_scores)
#         self.saver.save_mean_metric_by_class(self.mean_record)
#         self.saver.save_mean_metric(self.metric_record)

#         # paint mean metrics for all classes
#         Plot(1, 1).subplot().many_metrics(self.mean_record).complete().save(mean_metrics_image)

#     def record_epochs(self, output_dir: Path, n_epochs: int):
#         self._prepare(output_dir)

#         self.saver.save_all_metric_by_class(convert_to_ClassLabelManyScoreDict(self.epoch_metric_label_scores))
#         self.saver.save_mean_metric_by_class(self.mean_record)
#         self.saver.save_mean_metric(self.metric_record)

#         epoch_metrics_image = output_dir / "epoch_metrics.png"
#         mean_metrics_image = output_dir / "mean_metrics.png"

#         # paint metrics curve for all classes in one figure
#         n = len(self.metric_labels)
#         if n > 4:
#             nrows, ncols = (n + 2) // 3, 3
#         elif n == 4:
#             nrows, ncols = 2, 2
#         else:
#             nrows, ncols = 1, n

#         plot = Plot(nrows, ncols)
#         for metric in self.metric_labels:
#             plot.subplot().many_epoch_metrics(n_epochs, self.epoch_metric_label_scores[metric], self.class_labels, title=metric).complete()
#         plot.save(epoch_metrics_image)

#         # paint mean metrics for all classes
#         Plot(1, 1).subplot().many_metrics(self.mean_record).complete().save(mean_metrics_image)
