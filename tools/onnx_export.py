# This file is to export the given model to ONNX format
# TODO:

import torch
import torch.nn as nn
from pathlib import Path


class OnnxExport:
    def __init__(self, model: nn.Module, input_size: tuple):
        self.model = model
        self.input_size = input_size

    def save(self, path: Path):
        dummy_input = torch.randn(self.input_size)
        torch.onnx.export(self.model, dummy_input, path)
