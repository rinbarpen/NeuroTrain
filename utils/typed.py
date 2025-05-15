from typing import List, Literal, Union, Dict, TypedDict, Set
import numpy as np
from pydantic import BaseModel, Field

from pathlib import Path
from PIL import Image
import cv2

FilePath = Union[str, Path]
ImageInstance = Union[Image.Image, cv2.Mat]

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

MetricLabelArgmaxDict = Dict[MetricLabel, ClassLabel]
MetricLabelArgminDict = Dict[MetricLabel, ClassLabel]

class MetricAfterDict(TypedDict):
    mean: MetricLabelOneScoreDict
    std: MetricLabelOneScoreDict
    argmax: MetricLabelArgmaxDict
    argmin: MetricLabelArgminDict

# class MetricClassUnit(BaseModel):
#     metric_label: str = Field(default=None)
#     class_label: str = Field(default=None)
#     scores: List[FLOAT] = Field(default=[])

#     @staticmethod
#     def create(metric_label: str, class_label: str, scores: list[FLOAT]):
#         return MetricClassUnit(metric_label=metric_label, class_label=class_label, scores=scores)

#     def append(self, scores: FLOAT|list[FLOAT]):
#         if isinstance(scores, FLOAT):
#             scores = [scores]
#         self.scores.extend(scores)

#     def mean(self) -> FLOAT:
#         return np.mean(self.scores)
#     def std(self) -> FLOAT:
#         return np.std(self.scores)
#     def argmax(self) -> tuple[FLOAT, int]:
#         index = np.argmax(self.scores)
#         return self.scores[index], index
#     def argmin(self) -> tuple[FLOAT, int]:
#         index = np.argmin(self.scores)
#         return self.scores[index], index

# class MetricClassMap(BaseModel):
#     units: List[MetricClassUnit]
#     metric_labels: Set[str] = set()
#     class_labels: Set[str] = set()

#     def add(self, metric_label: str, class_label: str, scores: list[FLOAT]|FLOAT):
#         for unit in self.units:
#             if unit.metric_label == metric_label and unit.class_label == class_label:
#                 unit.append(scores)
#         else:
#             unit = MetricClassUnit.create(metric_label=metric_label, class_label=class_label, scores=scores)
#             self.units.append(unit)

#         self.metric_labels.add(metric_label)
#         self.class_labels.add(class_label)

#     def get(self, metric_label: str, class_label: str) -> list[FLOAT] | None:
#         for unit in self.units:
#             if unit.metric_label == metric_label and unit.class_label == class_label:
#                 return unit
#         return None

#     def mean_by_metric(self, metric_label: str) -> FLOAT:
#         x = [unit.mean() for unit in self.units if unit.metric_label == metric_label]
#         return np.mean(x)
#     def std_by_metric(self, metric_label: str) -> FLOAT:
#         x = [unit.mean() for unit in self.units if unit.metric_label == metric_label]
#         return np.std(x)
#     def mean_std_by_metric(self, metric_label: str) -> FLOAT:
#         x = [unit.mean() for unit in self.units if unit.metric_label == metric_label]
#         return np.mean(x), np.std(x)
#     def argmax_by_metric(self, metric_label: str) -> tuple[str, int]:
#         class_labels = [unit.class_label for unit in self.units if unit.metric_label == metric_label]
#         mean_scores = [unit.mean() for unit in self.units if unit.metric_label == metric_label]
#         index = np.argmax(mean_scores)
#         return class_labels[index], index
#     def argmin_by_metric(self, metric_label: str) -> tuple[str, int]:
#         class_labels = [unit.class_label for unit in self.units if unit.metric_label == metric_label]
#         mean_scores = [unit.mean() for unit in self.units if unit.metric_label == metric_label]
#         index = np.argmin(mean_scores)
#         return class_labels[index], index
#     def mean_all_by_metric(self) -> MetricLabelOneScoreDict:
#         # metric_labels = set([unit.metric_label for unit in self.units])
#         return {metric_label: self.mean_by_metric(metric_label) for metric_label in self.metric_labels}
#     def std_all_by_metric(self) -> MetricLabelOneScoreDict:
#         # metric_labels = set([unit.metric_label for unit in self.units])
#         return {metric_label: self.std_by_metric(metric_label) for metric_label in self.metric_labels}
#     def mean_std_all_by_metric(self) -> MetricLabelOneScoreDict:
#         # metric_labels = set([unit.metric_label for unit in self.units])
#         return {metric_label: self.mean_std_by_metric(metric_label) for metric_label in self.metric_labels}
#     def argmax_all_by_metric(self) -> MetricLabelArgmaxDict:
#         # metric_labels = set([unit.metric_label for unit in self.units])
#         return {metric_label: self.argmax_by_metric(metric_label)[0] for metric_label in self.metric_labels}
#     def argmin_all_by_metric(self) -> MetricLabelArgminDict:
#         # metric_labels = set([unit.metric_label for unit in self.units])
#         return {metric_label: self.argmin_by_metric(metric_label)[0] for metric_label in self.metric_labels}

#     def metric_after_dict(self) -> MetricAfterDict:
#         return {
#             'mean': self.mean_all_by_metric(),
#             'std': self.std_all_by_metric(),
#             'argmax': self.argmax_all_by_metric(),
#             'argmin': self.argmin_all_by_metric(),
#         }

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

def convert_to_ClassLabelManyScoreDict(data: MetricClassManyScoreDict) -> ClassMetricManyScoreDict:
    rdata = {}
    for metric, class_scores in data.items():
        for label, scores in class_scores.items():
            rdata[label] = {}
    for metric, class_scores in data.items():
        for label, scores in class_scores.items():
            rdata[label][metric] = scores
    return rdata
def convert_to_MetricClassManyScoreDict(data: ClassMetricManyScoreDict) -> MetricClassManyScoreDict:
    rdata = {}
    for label, metric_scores in data.items():
        for metric, scores in metric_scores.items():
            rdata[label] = {}
    for label, metric_scores in data.items():
        for metric, scores in metric_scores.items():
            rdata[label][metric] = scores
    return rdata
def convert_to_ClassLabelOneScoreDict(data: MetricClassOneScoreDict) -> ClassMetricOneScoreDict:
    rdata = {}
    for metric, class_scores in data.items():
        for label, scores in class_scores.items():
            rdata[label] = {}
    for metric, class_scores in data.items():
        for label, scores in class_scores.items():
            rdata[label][metric] = scores
    return rdata
def convert_to_MetricClassOneScoreDict(data: ClassMetricOneScoreDict) -> MetricClassOneScoreDict:
    rdata = {}
    for label, metric_scores in data.items():
        for metric, scores in metric_scores.items():
            rdata[label] = {}
    for label, metric_scores in data.items():
        for metric, scores in metric_scores.items():
            rdata[label][metric] = scores
    return rdata

def mean_from_ClassLabelOneScoreDict(data: ClassMetricManyScoreDict) -> ClassMetricOneScoreDict:
    rdata = {}
    for metric, class_scores in data.items():
        for label, scores in class_scores.items():
            rdata[label] = {}
    for metric, class_scores in data.items():
        for label, scores in class_scores.items():
            rdata[label][metric] = np.mean(scores)
    return rdata
def mean_from_MetricClassOneScoreDict(data: MetricClassManyScoreDict) -> MetricClassOneScoreDict:
    rdata = {}
    for label, metric_scores in data.items():
        for metric, scores in metric_scores.items():
            rdata[label] = {}
    for label, metric_scores in data.items():
        for metric, scores in metric_scores.items():
            rdata[label][metric] = np.mean(scores)
    return rdata

# def get_MetricAfterDict(data: MetricClassManyScoreDict) -> MetricAfterDict:
#     r = {
#         'mean': {metric_label: np.mean([np.mean(scores) for scores in class_map.values()]) for metric_label, class_map in data.items()},
#     }

#         # 'argmax': {metric_label: [np.mean(scores)] for metric_label, class_map in data.items()},
#         # 'argmin': {metric_label: {class_label: np.mean(scores) for class_label, scores in class_map.items()} for metric_label, class_map in data.items()},
