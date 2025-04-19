from typing import List, Literal, Union, Dict, TypedDict
import numpy as np
# from pydantic import BaseModel, Field

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

# class MetricClassUnit(BaseModel):
#     metric_label: str
#     class_label: str
#     scores: list[FLOAT]

#     def mean(self):
#         return np.mean(self.scores)
#     def argmax(self):
#         index = np.max(self.scores)
#         return self.scores[index], index
#     def argmin(self):
#         index = np.min(self.scores)
#         return self.scores[index], index

# class MetricClassMap(BaseModel):
#     units: list[MetricClassUnit]
    
#     def find(self, metric_label: str, class_label: str) -> list[FLOAT]:
#         for unit in self.units:
#             if unit.metric_label == metric_label and unit.class_label == class_label:
#                 return unit.scores
#         return None
    
#     def mean(self, metric_label: str) -> FLOAT:
#         x = [unit.mean() for unit in self.units if unit.metric_label == metric_label]
#         return np.mean(x)
#     def argmax(self, metric_label: str) -> str:
#         class_labels = [unit.class_label for unit in self.units if unit.metric_label == metric_label]
#         mean_scores = [unit.mean() for unit in self.units if unit.metric_label == metric_label]
#         index = np.argmax(mean_scores)
#         return class_labels[index], index
#     def argmin(self, metric_label: str) -> str:
#         class_labels = [unit.class_label for unit in self.units if unit.metric_label == metric_label]
#         mean_scores = [unit.mean() for unit in self.units if unit.metric_label == metric_label]
#         index = np.argmin(mean_scores)
#         return class_labels[index], index

#     def mean_all(self) -> MetricLabelOneScoreDict:
#         metric_labels = set(unit.metric_label for unit in self.units)
#         return {metric_label: self.mean(metric_label) for metric_label in metric_labels}
#     def argmax_all(self) -> MetricLabelOneScoreDict:
#         metric_labels = set(unit.metric_label for unit in self.units)
#         return {metric_label: self.argmax(metric_label) for metric_label in metric_labels}
#     def argmin_all(self) -> MetricLabelOneScoreDict:
#         metric_labels = set(unit.metric_label for unit in self.units)
#         return {metric_label: self.argmin(metric_label) for metric_label in metric_labels}

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

def convert_to_ClassLabelManyScoreDict(data: MetricClassManyScoreDict) -> ClassMetricManyScoreDict:
    metric_labels = data.keys()
    class_labels = data[metric_labels[0]].keys()
    
    r = {class_label: {metric_label: data[metric_label][class_label] for metric_label in metric_labels} for class_label in class_labels}
    return r
def convert_to_MetricClassManyScoreDict(data: ClassMetricManyScoreDict) -> MetricClassManyScoreDict:
    class_labels = data.keys()
    metric_labels = data[class_labels[0]].keys()
    
    r = {metric_label: {class_label: data[metric_label][class_label] for class_label in class_labels} for metric_label in metric_labels}
    return r
def convert_to_ClassLabelOneScoreDict(data: MetricClassOneScoreDict) -> ClassMetricOneScoreDict:
    metric_labels = data.keys()
    class_labels = data[metric_labels[0]].keys()
    
    r = {class_label: {metric_label: data[metric_label][class_label] for metric_label in metric_labels} for class_label in class_labels}
    return r
def convert_to_MetricClassOneScoreDict(data: ClassMetricOneScoreDict) -> MetricClassOneScoreDict:
    class_labels = data.keys()
    metric_labels = data[class_labels[0]].keys()
    
    r = {metric_label: {class_label: data[metric_label][class_label] for class_label in class_labels} for metric_label in metric_labels}
    return r

def mean_from_ClassLabelOneScoreDict(data: ClassMetricManyScoreDict) -> ClassMetricOneScoreDict:
    metric_labels = data.keys()
    class_labels = data[metric_labels[0]].keys()
    
    r = {class_label: {metric_label: np.mean(data[metric_label][class_label]) for metric_label in metric_labels} for class_label in class_labels}
    return r
def mean_from_MetricClassOneScoreDict(data: MetricClassManyScoreDict) -> MetricClassOneScoreDict:
    class_labels = data.keys()
    metric_labels = data[class_labels[0]].keys()
    
    r = {metric_label: {class_label: np.mean(data[metric_label][class_label]) for class_label in class_labels} for metric_label in metric_labels}
    return r

# def get_MetricAfterDict(data: MetricClassManyScoreDict) -> MetricAfterDict:
#     r = {
#         'mean': {metric_label: np.mean([np.mean(scores) for scores in class_map.values()]) for metric_label, class_map in data.items()},
#     }

#         # 'argmax': {metric_label: [np.mean(scores)] for metric_label, class_map in data.items()},
#         # 'argmin': {metric_label: {class_label: np.mean(scores) for class_label, scores in class_map.items()} for metric_label, class_map in data.items()},
