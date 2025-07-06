import logging
import numpy as np
from pathlib import Path
from src.utils.typed import *
from src.utils.painter import Plot
from src.utils.data_saver import DataSaver
from src.defines import neighbour_code_to_normals


# ScoreAggregator: 用于聚合和统计分数
class ScoreAggregator:
    def __init__(self, mcm_scores: MetricClassManyScoreDict):
        """
        Initializes the ScoreAggregator with the input many scores dictionary.

        Args:
            mcm_scores: The input data in MetricClassManyScoreDict format.
        """
        self._mcm_scores: MetricClassManyScoreDict = mcm_scores

        # Internal storage for computed results (lazy computation via properties)
        self._mc1_mean: MetricClassOneScoreDict | None = None
        self._mc1_std: MetricClassOneScoreDict | None = None
        self._cmm: ClassMetricManyScoreDict | None = None
        self._cm1_mean: ClassMetricOneScoreDict | None = None
        self._cm1_std: ClassMetricOneScoreDict | None = None
        self._ml1_mean: MetricLabelOneScoreDict | None = None
        self._ml1_std: MetricLabelOneScoreDict | None = None

        self._m2_mean: MetricLabelManyScoreDict | None = None # by classes

    @property
    def mc1_mean(self) -> MetricClassOneScoreDict:
        """Metric -> Class -> Mean Score"""
        if self._mc1_mean is None:
            self._mc1_mean = self._compute_mc1(np.mean)
        return self._mc1_mean

    @property
    def mc1_std(self) -> MetricClassOneScoreDict:
        """Metric -> Class -> Standard Deviation Score"""
        if self._mc1_std is None:
            self._mc1_std = self._compute_mc1(np.std)
        return self._mc1_std

    @property
    def cmm(self) -> ClassMetricManyScoreDict:
        """Class -> Metric -> Many Scores (Transposed)"""
        if self._cmm is None:
            self._cmm = self._compute_cmm()
        return self._cmm

    @property
    def cm1_mean(self) -> ClassMetricOneScoreDict:
        """Class -> Metric -> Mean Score (from transposed data)"""
        if self._cm1_mean is None:
            # We can compute this directly or from the cmm data structure
            # Computing from cmm is natural if cmm is also needed.
            if self._cmm is None:
                self._cmm = self._compute_cmm()  # Ensure cmm is computed
            self._cm1_mean = self._compute_cm1(self._cmm, np.mean)
        return self._cm1_mean

    @property
    def cm1_std(self) -> ClassMetricOneScoreDict:
        """Class -> Metric -> Standard Deviation Score (from transposed data)"""
        if self._cm1_std is None:
            if self._cmm is None:
                self._cmm = self._compute_cmm()
            self._cm1_std = self._compute_cm1(self._cmm, np.std)
        return self._cm1_std

    @property
    def ml1_mean(self) -> MetricLabelOneScoreDict:
        """Metric -> Mean Score (aggregated across all classes)"""
        if self._ml1_mean is None:
            self._ml1_mean = self._compute_ml1(np.mean)
        return self._ml1_mean

    @property
    def ml1_std(self) -> MetricLabelOneScoreDict:
        """Metric -> Standard Deviation Score (aggregated across all classes)"""
        if self._ml1_std is None:
            self._ml1_std = self._compute_ml1(np.std)
        return self._ml1_std

    @property
    def m2_mean(self) -> MetricLabelManyScoreDict:
        """Metric -> Class -> Mean Score (aggregated across all classes)"""
        if self._m2_mean is None:
            self._m2_mean = self._compute_m2()
        return self._m2_mean

    # --- Internal computation methods ---

    def _compute_mc1(
        self, func: Callable[[List[FLOAT]], FLOAT]
    ) -> MetricClassOneScoreDict:
        """
        Helper to compute MetricClassOneScoreDict (mean or std) by
        applying a function to each list of scores.
        """
        result: MetricClassOneScoreDict = {}
        for metric, class_scores_dict in self._mcm_scores.items():
            result[metric] = {}
            for class_label, scores_list in class_scores_dict.items():
                # Handle potential empty lists gracefully (np.mean/std return NaN)
                if scores_list:
                    result[metric][class_label] = FLOAT(func(scores_list))
                else:
                    # Or handle as required, e.g., 0.0 or raise error
                    result[metric][class_label] = FLOAT(np.nan)  # Using NaN is standard

        return result

    def _compute_cmm(self) -> ClassMetricManyScoreDict:
        """
        Helper to compute ClassMetricManyScoreDict by transposing
        the input MetricClassManyScoreDict.
        """
        result: ClassMetricManyScoreDict = {}
        for metric, class_scores_dict in self._mcm_scores.items():
            for class_label, scores_list in class_scores_dict.items():
                if class_label not in result:
                    result[class_label] = {}
                result[class_label][
                    metric
                ] = scores_list  # Storing the list reference/copy
        return result

    def _compute_cm1(
        self, cmm_scores: ClassMetricManyScoreDict, func: Callable[[List[FLOAT]], FLOAT]
    ) -> ClassMetricOneScoreDict:
        """
        Helper to compute ClassMetricOneScoreDict (mean or std) from
        ClassMetricManyScoreDict.
        """
        result: ClassMetricOneScoreDict = {}
        for class_label, metric_scores_dict in cmm_scores.items():
            result[class_label] = {}
            for metric, scores_list in metric_scores_dict.items():
                if scores_list:
                    result[class_label][metric] = FLOAT(func(scores_list))
                else:
                    result[class_label][metric] = FLOAT(np.nan)
        return result

    def _compute_ml1(
        self, func: Callable[[List[FLOAT]], FLOAT]
    ) -> MetricLabelOneScoreDict:
        """
        Helper to compute MetricLabelOneScoreDict (mean or std) by
        aggregating scores across all classes for each metric.
        """
        result: MetricLabelOneScoreDict = {}
        for metric, class_scores_dict in self._mcm_scores.items():
            all_scores_for_metric: List[FLOAT] = []
            for class_label, scores_list in class_scores_dict.items():
                all_scores_for_metric.extend(
                    scores_list
                )  # Concatenate all scores for this metric

            if all_scores_for_metric:
                result[metric] = FLOAT(func(all_scores_for_metric))
            else:
                result[metric] = FLOAT(np.nan)
        return result

    def _compute_m2(self) -> MetricLabelManyScoreDict:
        """
        Helper to compute MetricLabelManyScoreDict by
        aggregating scores across all classes for each metric.
        """
        result: MetricLabelManyScoreDict = {}
        n = len(next(iter(next(iter(self._mcm_scores.values())).values())))
        print(f'{n=}')
        for metric, class_scores_dict in self._mcm_scores.items():
            result[metric] = []
            for i in range(n):
                s = [class_scores_dict[k][i] for k in class_scores_dict.keys()]
                result[metric].append(FLOAT(np.mean(s)))
        return result

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

        # {'recall': {'0': [], '1': []}, ...}
        self.all_metric_label_scores: MetricClassManyScoreDict = {
            metric: {label: [] for label in class_labels} for metric in metric_labels
        }
        # 按epoch保存
        self.epoch_metric_label_scores: MetricClassManyScoreDict = {
            metric: {label: [] for label in class_labels} for metric in metric_labels
        }

        for class_label in class_labels:
            class_dir = self.output_dir / class_label
            class_dir.mkdir(parents=True, exist_ok=True)

    def finish_one_batch(self, targets: np.ndarray, outputs: np.ndarray):
        from src.utils.scores import scores
        metrics = scores(
            targets, outputs, labels=self.class_labels, metric_labels=self.metric_labels
        )
        for metric_label in self.metric_labels:
            for class_label in self.class_labels:
                score = metrics[metric_label][class_label]
                self.all_metric_label_scores[metric_label][class_label].append(score)
        return metrics

    def finish_one_epoch(self):
        import numpy as np
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
        n = len(self.metric_labels)
        if n > 4:
            nrows, ncols = (n + 2) // 3, 3
        elif n == 4:
            nrows, ncols = 2, 2
        else:
            nrows, ncols = 1, n

        scores_agg = ScoreAggregator(self.epoch_metric_label_scores)
        plot = Plot(nrows, ncols)
        for metric in self.metric_labels:
            plot.subplot().many_epoch_metrics_by_class(
                n_epochs,
                self.epoch_metric_label_scores[metric],
                self.class_labels,
                title=metric,
            ).complete()
        plot.save(epoch_metrics_image)
        # 全局所有metric
        epoch_metrics_image = self.output_dir / "epoch_metrics_curve.png"
        plot = Plot(1, 1)
        plot.subplot().many_epoch_metrics(
            n_epochs, scores_agg.m2_mean, metric_labels=self.metric_labels, title="All Metrics"
        ).complete()
        plot.save(epoch_metrics_image)
        # 每个类别
        for label in self.class_labels:
            epoch_metric_image = self.output_dir / label / "metrics.png"
            plot = Plot(1, 1)
            plot.subplot().many_epoch_metrics(n_epochs, scores_agg.cmm[label], metric_labels=self.metric_labels, title=label).complete()
            plot.save(epoch_metric_image)
        # 每个类别每个metric
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
        scores_agg = ScoreAggregator(mc_scores)
        self.saver.save_all_metric_by_class(scores_agg.cmm)
        self.saver.save_mean_metric_by_class(scores_agg.cm1_mean)
        self.saver.save_std_metric_by_class(scores_agg.cm1_std)
        self.saver.save_mean_metric(scores_agg.ml1_mean)
        self.saver.save_std_metric(scores_agg.ml1_std)

        mean_metrics_image = self.output_dir / "mean_metrics_per_classes.png"
        Plot(1, 1).subplot().many_metrics(scores_agg.cm1_mean).complete().save(
            mean_metrics_image
        )
        for label in self.class_labels:
            mean_image = self.output_dir / label / "mean_metric.png"
            Plot(1, 1).subplot().metrics(scores_agg.cm1_mean[label], label).complete().save(
                mean_image
            )
        mean_metrics_image = self.output_dir / "mean_metrics.png"
        Plot(1, 1).subplot().metrics(scores_agg.ml1_mean).complete().save(
            mean_metrics_image
        )
