from typing import List, Literal, Union, Dict, TypedDict, Callable
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

FilePath = Union[str, Path]
ImageInstance = Union[Image.Image, cv2.Mat]

def to_path(p: FilePath) -> Path:
    return Path(p)

def to_pil_image(img: ImageInstance) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    return Image.fromarray(img)

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
