from typing import List, Literal, Union, Dict, TypedDict, Callable
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

FilePath = Union[str, Path]
ImageInstance = Union[Image.Image, cv2.Mat]

FLOAT = np.float64
ClassLabel = str
ClassLabelsList = List[ClassLabel]
MetricLabel = str # Literal['iou', 'accuracy', 'precision', 'recall', 'f1', 'dice']
MetricLabelsList = List[MetricLabel]

ClassLabelOneScoreDict = Dict[ClassLabel, FLOAT]
ClassLabelManyScoreDict = Dict[ClassLabel, List[FLOAT]]
MetricLabelOneScoreDict = Dict[MetricLabel, FLOAT]
MetricLabelManyScoreDict = Dict[MetricLabel, List[FLOAT]]

MetricClassOneScoreDict = Dict[MetricLabel, ClassLabelOneScoreDict]
MetricClassManyScoreDict = Dict[MetricLabel, ClassLabelManyScoreDict]
ClassMetricOneScoreDict = Dict[ClassLabel, MetricLabelOneScoreDict]
ClassMetricManyScoreDict = Dict[ClassLabel, MetricLabelManyScoreDict]

MetricLabelArgmaxDict = Dict[MetricLabel, ClassLabel]
MetricLabelArgminDict = Dict[MetricLabel, ClassLabel]

class MetricAfterDict(TypedDict):
    mean: MetricLabelOneScoreDict
    std: MetricLabelOneScoreDict
    argmax: MetricLabelArgmaxDict
    argmin: MetricLabelArgminDict

class ScoreAggregator:
    """
    Aggregates and transforms a MetricClassManyScoreDict into various
    mean, std, and transposed formats.
    """

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

    # --- Properties to access computed results ---

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

def create_ClassLabelOneScoreDict(class_labels: ClassLabelsList, 
                                  default_score: FLOAT=0.0) -> ClassLabelOneScoreDict:
    return {label: default_score for label in class_labels}
def create_ClassLabelManyScoreDict(class_labels: ClassLabelsList) -> ClassLabelManyScoreDict:
    return {label: [] for label in class_labels}
def create_MetricLabelOneScoreDict(metric_labels: MetricLabelsList, 
                                   default_score: FLOAT=0.0) -> MetricLabelOneScoreDict:
    return {label: default_score for label in metric_labels}
def create_MetricLabelManyScoreDict(metric_labels: MetricLabelsList) -> MetricLabelManyScoreDict:
    return {label: [] for label in metric_labels}
def create_MetricClassOneScoreDict(metric_labels: MetricLabelsList, 
                                   class_labels: ClassLabelsList, 
                                   default_score: FLOAT=0.0) -> MetricClassOneScoreDict:
    return {metric: {label: default_score for label in class_labels} for metric in metric_labels}
def create_MetricClassManyScoreDict(metric_labels: MetricLabelsList, 
                                    class_labels: ClassLabelsList) -> MetricClassManyScoreDict:
    return {metric: {label: [] for label in class_labels} for metric in metric_labels}
def create_ClassMetricOneScoreDict(metric_labels: MetricLabelsList, 
                                   class_labels: ClassLabelsList, 
                                   default_score: FLOAT=0.0) -> ClassMetricOneScoreDict:
    return {label: {metric: default_score for metric in metric_labels} for label in class_labels}
def create_ClassMetricManyScoreDict(metric_labels: MetricLabelsList, 
                                    class_labels: ClassLabelsList) -> ClassMetricManyScoreDict:
    return {label: {metric: [] for metric in metric_labels} for label in class_labels}

def create_MetricAfterDict(metric_labels: MetricLabelsList) -> MetricAfterDict:
    return {
        'mean': create_MetricLabelOneScoreDict(metric_labels),
        'std': create_MetricLabelOneScoreDict(metric_labels),
        'argmax': {metric: '' for metric in metric_labels},
        'argmin': {metric: '' for metric in metric_labels},
    }

def convert_to_ClassMetricManyScoreDict(data: MetricClassManyScoreDict) -> ClassMetricManyScoreDict:
    result = {}
    for metric_label, class_map in data.items():
        for class_label, scores in class_map.items():
            if class_label not in result:
                result[class_label] = {}
            result[class_label][metric_label] = scores
    return result

def convert_to_MetricClassManyScoreDict(data: ClassMetricManyScoreDict) -> MetricClassManyScoreDict:
    result = {}
    for class_label, metric_map in data.items():
        for metric_label, scores in metric_map.items():
            if metric_label not in result:
                result[metric_label] = {}
            result[metric_label][class_label] = scores
    return result

def convert_to_ClassMetricOneScoreDict(data: MetricClassOneScoreDict) -> ClassMetricOneScoreDict:
    result = {}
    for metric_label, class_map in data.items():
        for class_label, score in class_map.items():
            if class_label not in result:
                result[class_label] = {}
            result[class_label][metric_label] = score
    return result

def convert_to_MetricClassOneScoreDict(data: ClassMetricOneScoreDict) -> MetricClassOneScoreDict:
    result = {}
    for class_label, metric_map in data.items():
        for metric_label, score in metric_map.items():
            if metric_label not in result:
                result[metric_label] = {}
            result[metric_label][class_label] = score
    return result

def mean_from_ClassMetricManyScoreDict(data: ClassMetricManyScoreDict) -> ClassMetricOneScoreDict:
    result = {}
    for class_label, metric_map in data.items():
        result[class_label] = {}
        for metric_label, scores in metric_map.items():
            result[class_label][metric_label] = np.mean(scores)
    return result

def mean_from_MetricClassManyScoreDict(data: MetricClassManyScoreDict) -> MetricClassOneScoreDict:
    result = {}
    for metric_label, class_map in data.items():
        result[metric_label] = {}
        for class_label, scores in class_map.items():
            result[metric_label][class_label] = np.mean(scores)
    return result

def std_from_ClassMetricManyScoreDict(data: ClassMetricManyScoreDict) -> ClassMetricOneScoreDict:
    result = {}
    for class_label, metric_map in data.items():
        result[class_label] = {}
        for metric_label, scores in metric_map.items():
            result[class_label][metric_label] = np.std(scores)
    return result

def std_from_MetricClassManyScoreDict(data: MetricClassManyScoreDict) -> MetricClassOneScoreDict:
    result = {}
    for metric_label, class_map in data.items():
        result[metric_label] = {}
        for class_label, scores in class_map.items():
            result[metric_label][class_label] = np.std(scores)
    return result

def mean_from_MetricLabelManyScoreDict(data: MetricLabelManyScoreDict) -> MetricLabelOneScoreDict:
    result = {}
    for metric_label, scores in data.items():
        result[metric_label] = np.mean(scores)
    return result
def std_from_MetricLabelManyScoreDict(data: MetricLabelManyScoreDict) -> MetricLabelOneScoreDict:
    result = {}
    for metric_label, scores in data.items():
        result[metric_label] = np.std(scores)
    return result
def mean_from_ClassLabelManyScoreDict(data: ClassLabelManyScoreDict) -> ClassLabelOneScoreDict:
    result = {}
    for class_label, scores in data.items():
        result[class_label] = np.mean(scores)
    return result
def std_from_ClassLabelManyScoreDict(data: ClassLabelManyScoreDict) -> ClassLabelOneScoreDict:
    result = {}
    for class_label, scores in data.items():
        result[class_label] = np.std(scores)
    return result
