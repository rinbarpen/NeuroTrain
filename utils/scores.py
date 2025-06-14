import logging
import numpy as np
from sklearn import metrics

from utils.typed import *
from utils.painter import Plot
from utils.data_saver import DataSaver
from utils.typed import ScoreAggregator


# y_true, y_pred: (B, C, X)
def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: ClassLabelsList,
    *,
    class_axis: int = 1,
    average: str = "binary",
):
    n_labels = len(labels)

    y_true_flatten = [
        yt.flatten() for yt in np.split(y_true, n_labels, axis=class_axis)
    ]
    y_pred_flatten = [
        yp.flatten() for yp in np.split(y_pred, n_labels, axis=class_axis)
    ]

    result = dict()
    for label, y_true, y_pred in zip(labels, y_true_flatten, y_pred_flatten):
        score = metrics.f1_score(y_true, y_pred, average=average)
        result[label] = np.float64(score)

    return result


def recall_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: ClassLabelsList,
    *,
    class_axis: int = 1,
    average: str = "binary",
):
    n_labels = len(labels)

    y_true_flatten = [
        yt.flatten() for yt in np.split(y_true, n_labels, axis=class_axis)
    ]
    y_pred_flatten = [
        yp.flatten() for yp in np.split(y_pred, n_labels, axis=class_axis)
    ]

    result = dict()
    for label, y_true, y_pred in zip(labels, y_true_flatten, y_pred_flatten):
        score = metrics.recall_score(y_true, y_pred, average=average)
        result[label] = np.float64(score)

    return result


def precision_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: ClassLabelsList,
    *,
    class_axis: int = 1,
    average: str = "binary",
):
    n_labels = len(labels)

    y_true_flatten = [
        yt.flatten() for yt in np.split(y_true, n_labels, axis=class_axis)
    ]
    y_pred_flatten = [
        yp.flatten() for yp in np.split(y_pred, n_labels, axis=class_axis)
    ]

    result = dict()
    for label, y_true, y_pred in zip(labels, y_true_flatten, y_pred_flatten):
        score = metrics.precision_score(y_true, y_pred, average=average)
        result[label] = np.float64(score)

    return result


def accuracy_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: ClassLabelsList,
    *,
    class_axis: int = 1,
    average: str = "binary",
):
    n_labels = len(labels)

    y_true_flatten = [
        yt.flatten() for yt in np.split(y_true, n_labels, axis=class_axis)
    ]
    y_pred_flatten = [
        yp.flatten() for yp in np.split(y_pred, n_labels, axis=class_axis)
    ]

    result = dict()
    for label, y_true, y_pred in zip(labels, y_true_flatten, y_pred_flatten):
        score = metrics.accuracy_score(y_true, y_pred)
        result[label] = np.float64(score)

    return result


def dice_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: ClassLabelsList,
    *,
    class_axis: int = 1,
    average: str = "binary",
):
    n_labels = len(labels)

    y_true_flatten = [
        yt.flatten() for yt in np.split(y_true, n_labels, axis=class_axis)
    ]
    y_pred_flatten = [
        yp.flatten() for yp in np.split(y_pred, n_labels, axis=class_axis)
    ]

    eps = 1e-6

    result = dict()
    for label, y_true, y_pred in zip(labels, y_true_flatten, y_pred_flatten):
        intersection = np.logical_and(y_true, y_pred).sum()
        union = np.logical_or(y_true, y_pred).sum()
        score = (2 * intersection + eps) / (union + eps)
        result[label] = np.float64(score)

    return result


def iou_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: ClassLabelsList,
    *,
    class_axis: int = 1,
    average: str = "binary",
):
    n_labels = len(labels)

    y_true_flatten = [
        yt.flatten() for yt in np.split(y_true, n_labels, axis=class_axis)
    ]
    y_pred_flatten = [
        yp.flatten() for yp in np.split(y_pred, n_labels, axis=class_axis)
    ]

    result = dict()
    for label, y_true, y_pred in zip(labels, y_true_flatten, y_pred_flatten):
        intersection = np.logical_and(y_true, y_pred).sum()
        union = np.logical_or(y_true, y_pred).sum()
        score = intersection / union if union > 0 else 0.0
        result[label] = np.float64(score)

    return result


def scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: ClassLabelsList,
    metric_labels: MetricLabelsList,
    *,
    class_axis: int = 1,
    average: str = "binary",
):
    result: MetricClassOneScoreDict = create_MetricClassOneScoreDict(
        metric_labels, labels
    )
    result_after: MetricAfterDict = create_MetricAfterDict(labels)

    MAP = {
        "iou": iou_score,
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
        "dice": dice_score,
    }

    for metric in metric_labels:
        result[metric] = MAP[metric](
            y_true, y_pred, labels, class_axis=class_axis, average=average
        )
        values = np.array(list(result[metric].values()))
        result_after["mean"][metric] = values.mean()
        result_after["argmax"][metric] = labels[values.argmax()]
        result_after["argmin"][metric] = labels[values.argmin()]

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


def dice_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_axis: int = 1,
    average: str = "binary",
) -> np.float64:
    labels = [str(i) for i in range(y_true.shape[class_axis])]
    result = dice_score(y_true, y_pred, labels, class_axis=class_axis, average=average)
    score = np.fromiter(result.values(), dtype=np.float64).mean()
    return 1 - score


def kl_divergence_loss(
    y_true: np.ndarray, y_pred: np.ndarray, *, epsilon: float = 1e-7
):
    # 确保输入为概率分布
    y_true = np.clip(y_true, epsilon, 1.0 - epsilon)
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)

    # 计算 KL 散度: sum(p * log(p/q))
    kl_div = np.sum(y_true * np.log(y_true / y_pred), axis=1)
    return np.mean(kl_div)


class ScoreCalculator:
    def __init__(
        self,
        output_dir: FilePath,
        class_labels: ClassLabelsList,
        metric_labels: MetricLabelsList,
        *,
        logger=None,
        saver: DataSaver,
    ):
        self.logger = logger or logging.getLogger()
        self.saver = saver
        self.output_dir = Path(output_dir)
        self.class_labels = class_labels
        self.metric_labels = metric_labels

        # {'recall': {
        #   '0': [], '1': []},
        #  'precision: {
        #   '0': [], '1': []}}
        self.all_metric_label_scores: MetricClassManyScoreDict = {
            metric: {label: [] for label in class_labels} for metric in metric_labels
        }
        # but epoch_metric_label_scores is saved by epoch
        self.epoch_metric_label_scores: MetricClassManyScoreDict = {
            metric: {label: [] for label in class_labels} for metric in metric_labels
        }

        for class_label in class_labels:
            class_dir = self.output_dir / class_label
            class_dir.mkdir(parents=True, exist_ok=True)

    def finish_one_batch(self, targets: np.ndarray, outputs: np.ndarray):
        metrics, metrics_after = scores(
            targets, outputs, labels=self.class_labels, metric_labels=self.metric_labels
        )

        for metric_label in self.metric_labels:
            for class_label in self.class_labels:
                score = metrics[metric_label][class_label]
                self.all_metric_label_scores[metric_label][class_label].append(score)

        return metrics, metrics_after

    def finish_one_epoch(self):
        for metric_label in self.metric_labels:
            for class_label in self.class_labels:
                score = np.mean(self.all_metric_label_scores[metric_label][class_label])
                self.all_metric_label_scores[metric_label][class_label] = []
                self.epoch_metric_label_scores[metric_label][class_label].append(score)
        return self.epoch_metric_label_scores

    def record_batches(self):
        self._record(self.all_metric_label_scores)

    def record_epochs(self, n_epochs: int):
        self._record(self.epoch_metric_label_scores)

        epoch_metrics_image = self.output_dir / "epoch_metrics_curve_per_classes.png"
        # paint metrics curve for all classes in one figure
        n = len(self.metric_labels)
        if n > 4:
            nrows, ncols = (n + 2) // 3, 3
        elif n == 4:
            nrows, ncols = 2, 2
        else:
            nrows, ncols = 1, n

        scores = ScoreAggregator(self.epoch_metric_label_scores)
        # for global, plot all classes in one figure with all metrics
        plot = Plot(nrows, ncols)
        for metric in self.metric_labels:
            plot.subplot().many_epoch_metrics_by_class(
                n_epochs,
                self.epoch_metric_label_scores[metric],
                self.class_labels,
                title=metric,
            ).complete()
        plot.save(epoch_metrics_image)
        # for global, plot all metrics in one figure
        epoch_metrics_image = self.output_dir / "epoch_metrics_curve.png"
        plot = Plot(1, 1)
        plot.subplot().many_epoch_metrics(
            n_epochs, scores.m2_mean, metric_labels=self.metric_labels, title="All Metrics"
        ).complete()
        plot.save(epoch_metrics_image)
        # for every classes
        for label in self.class_labels:
            epoch_metric_image = self.output_dir / label / "metrics.png"
            plot = Plot(1, 1)
            plot.subplot().many_epoch_metrics(n_epochs, scores.cmm[label], metric_labels=self.metric_labels, title=label).complete()
            plot.save(epoch_metric_image)
        # for every classes with sole metric
        for metric in self.metric_labels:
            for label in self.class_labels:
                m_scores = np.array(
                    self.epoch_metric_label_scores[metric][label], dtype=np.float64
                )
                epoch_metric_image = self.output_dir / label / f"{metric}.png"
                plot = Plot(1, 1)
                plot = (
                    plot.subplot()
                    .epoch_metrics(
                        n_epochs, m_scores, label, title=f"Epoch-{label}-{metric}"
                    )
                    .complete()
                )
                plot.save(epoch_metric_image)

    def _record(self, mc_scores: MetricClassManyScoreDict):
        scores = ScoreAggregator(mc_scores)

        self.saver.save_all_metric_by_class(scores.cmm)
        self.saver.save_mean_metric_by_class(scores.cm1_mean)
        self.saver.save_std_metric_by_class(scores.cm1_std)
        self.saver.save_mean_metric(scores.ml1_mean)
        self.saver.save_std_metric(scores.ml1_std)

        mean_metrics_image = self.output_dir / "mean_metrics_per_classes.png"
        # paint mean metrics for all classes
        Plot(1, 1).subplot().many_metrics(scores.cm1_mean).complete().save(
            mean_metrics_image
        )
        for label in self.class_labels:
            mean_image = self.output_dir / label / "mean_metric.png"
            Plot(1, 1).subplot().metrics(scores.cm1_mean[label], label).complete().save(
                mean_image
            )
        mean_metrics_image = self.output_dir / "mean_metrics.png"
        # metric mean
        Plot(1, 1).subplot().metrics(scores.ml1_mean).complete().save(
            mean_metrics_image
        )
