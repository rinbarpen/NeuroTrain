import torch
import torch.nn.functional as F
from typing import Callable

def postprocess_binary_classification(targets: torch.Tensor, outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # 对二分类输出进行后处理
    # 假设outputs为[N, C]，C为类别数，targets同shape
    # 返回二值化后的targets和outputs
    targets = targets.long()
    # 对输出进行sigmoid激活，然后二值化
    outputs = (F.sigmoid(outputs) >= 0.5).long()
    return targets, outputs

def postprocess_multiclass_classification(targets: torch.Tensor, outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # 对多分类输出进行后处理
    # 假设outputs为[N, C]，C为类别数，targets为[N]或[N, 1]的类别索引
    # 返回处理后的targets和outputs
    targets = targets.long()
    if targets.dim() > 1 and targets.size(1) > 1:
        # 如果targets是one-hot编码，转换为类别索引
        targets = torch.argmax(targets, dim=1)
    # 对输出进行softmax，然后取最大值的索引
    outputs = torch.softmax(outputs, dim=1)
    outputs = torch.argmax(outputs, dim=1)
    return targets, outputs

def postprocess_regression(targets: torch.Tensor, outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # 对回归输出进行后处理
    # 回归任务通常不需要特殊处理，直接返回原始值
    # 但可以确保数据类型一致
    return targets.float(), outputs.float()

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
    elif "class" in name:
        if "multi" in name or "multiple" in name:
            return postprocess_multiclass_classification
        else:
            return postprocess_binary_classification
    elif "regress" in name:
        return postprocess_regression
    return None
