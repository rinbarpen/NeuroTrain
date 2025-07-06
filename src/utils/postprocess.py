import torch
import torch.nn.functional as F
from typing import Callable

def postprocess_binary_segmentation(targets: torch.Tensor, outputs: torch.Tensor, is_multitask=False) -> tuple[torch.Tensor, torch.Tensor]:
    if is_multitask:
        targets = targets[:, 1:, ...]
        outputs = outputs[:, 1:, ...]
    targets = targets.bool()
    outputs = F.sigmoid(outputs) >= 0.5
    return targets, outputs

def postprocess_instance_segmentation(targets: torch.Tensor, outputs: torch.Tensor, is_multitask=False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对实例分割输出进行后处理。
    假设outputs为[N, C, H, W]，C为类别数，targets同shape。
    返回二值化后的targets和outputs。
    """
    if is_multitask:
        outputs = outputs[:, 1:, ...]
        targets = targets[:, 1:, ...]
    # softmax后取最大类别
    outputs = torch.softmax(outputs, dim=1)
    outputs = torch.argmax(outputs, dim=1, keepdim=True)
    targets = torch.argmax(targets, dim=1, keepdim=True)
    return targets, outputs


def select_postprocess_fn(name: str) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]|None:
    if "segment" in name:
        if "instance" in name:
            return postprocess_instance_segmentation
        else:
            return postprocess_binary_segmentation
    return None
