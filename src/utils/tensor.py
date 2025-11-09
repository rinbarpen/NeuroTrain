from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def move_to_device(value: Any, device: torch.device | str):
    """递归地将张量/容器移动到指定设备。"""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return type(value)(move_to_device(v, device) for v in value)
    return value

