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

"""
Data Format:
task:
    run_id:
        predict:
            {xx}
            config[.json|.toml|.yaml]
        test:
            {class}:
                mean_metrics[.csv|.parquet]
            mean_metric.png
            mean_metrics[.csv|.parquet]
            config[.json|.toml|.yaml]
        train:
            {class}:
                all_metrics[.csv|.parquet]
                mean_metrics[.csv|.parquet]
            weights:
                best.pt
                last.pt
                {net}-{epoch}of{num_epochs}.pt
                best-ext.pt                         | optional
                last-ext.pt                         | optional
                {net}-{epoch}of{num_epochs}-ext.pt  | optional
            train_loss[.csv|.parquet]
            train_epoch_loss.png
            valid_loss[.csv|.parquet]               | optional
            valid_epoch_loss.png                    | optional
            epoch_metrics.png
            mean_metric.png
            mean_metrics[.csv|.parquet]
            config[.json|.toml|.yaml]
            best.pt                                 | optional, soft link
            last.pt                                 | optional, soft link
"""

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
