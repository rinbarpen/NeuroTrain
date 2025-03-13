from typing import List, Literal, Union, Dict, TypedDict
import numpy as np

FLOAT = np.float64
ClassLabel = str
ClassLabelsList = List[ClassLabel]
MetricLabel = Literal['iou', 'accuracy', 'precision', 'recall', 'f1', 'dice']
MetricLabelsList = List[MetricLabel]

ClassLabelOneScoreDict = Dict[ClassLabel, FLOAT]
ClassLabelManyScoreDict = Dict[ClassLabel, List[FLOAT]]
MetricLabelOneScoreDict = Dict[MetricLabel, FLOAT]
MetricLabelManyScoreDict = Dict[MetricLabel, List[FLOAT]]

MetricClassOneScoreDict = Dict[MetricLabel, ClassLabelOneScoreDict]
MetricClassManyScoreDict = Dict[MetricLabel, ClassLabelManyScoreDict]
ClassMetricOneScoreDict = Dict[ClassLabel, MetricLabelOneScoreDict]
ClassMetricManyScoreDict = Dict[ClassLabel, MetricLabelManyScoreDict]

class MetricAfterDict(TypedDict):
    mean: MetricLabelOneScoreDict
    argmax: Dict[MetricLabel, ClassLabel]
    argmin: Dict[MetricLabel, ClassLabel]

ALL_METRIC_LABELS = ['iou', 'accuracy', 'precision', 'recall', 'f1', 'dice']
