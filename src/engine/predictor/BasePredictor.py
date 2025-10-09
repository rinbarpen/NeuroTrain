from pathlib import Path
from torch import nn
import logging
from typing import Sequence
import torch
import numpy as np

from src.config import get_config
from src.utils.timer import Timer

class BasePredictor:
    def __init__(self, output_dir: str|Path, model: nn.Module, device: str='cuda', **kwargs):
        self.output_dir = Path(output_dir)
        self.model = model
        self.device = device
        self.logger = logging.getLogger('predictor')
        self.timer = Timer()

        self.output_dir.mkdir(parents=True, exist_ok=True)

    @torch.inference_mode()
    def predict(self, inputs, preprocess_fn=None, postprocess_fn=None, **kwargs):
        ...
