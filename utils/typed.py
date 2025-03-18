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
        'argmax': {metric: '' for metric in metric_labels},
        'argmin': {metric: '' for metric in metric_labels},
    }
