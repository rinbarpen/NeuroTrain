# analyze the number of model params and each project metrics
# TODO:
import os.path
import torch
from typing import TypedDict
from pathlib import Path
import os
from utils.util import load_model, load_model_ext, model_gflops, Timer

from utils.typed import ClassLabelManyScoreDict

MODEL_FILE = str|Path
from pydantic import BaseModel, Field

class AnalyzeParams(BaseModel):
    model: str|Path = Field(default=None)
    model_ext: str|Path = Field(default=None)

class AnalyzeMetricParams(BaseModel):
    model_name: str
    task: str
    class_metrics: ClassLabelManyScoreDict
    super_params: dict

class Analyzer:
    def __init__(self):
        pass

    def __del__(self):
        pass

    def __call__(self, x: AnalyzeParams):
        pass

    def analyze_metric(self, params: list[AnalyzeMetricParams]):
        header = " ".join(["Methods"].extend(["#" + k for k in p.super_params.keys()]))
        content = ""
        for p in params:
            content += p.model_name.join(["<b>", "<\b>"])
