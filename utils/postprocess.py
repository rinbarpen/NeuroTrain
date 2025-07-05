import torch
import torch.nn.functional as F
from typing import Callable

def postprocess_binary_segmentation(targets: torch.Tensor, outputs: torch.Tensor, is_multitask=False) -> tuple[torch.Tensor, torch.Tensor]:
    if is_multitask:
        targets = targets[:, 1:, ...]
        outputs = outputs[:, 1:, ...]
    targets = F.sigmoid(targets) >= 0.5
    outputs = F.sigmoid(outputs) >= 0.5
    # targets = targets >= 0.5
    # outputs = outputs >= 0.5
    return targets, outputs

def postprocess_instance_segmentation(targets: torch.Tensor, outputs: torch.Tensor, num_classes: int, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    ...


def select_postprocess_fn(name: str) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]|None:
    if "segment" in name:
        if "instance" in name:
            return postprocess_instance_segmentation
        else:
            return postprocess_binary_segmentation
    return None
