import torch

def postprocess_binary_segmentation(targets: torch.Tensor, outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    targets[targets >= 0.5] = 1
    targets[targets < 0.5] = 0
    outputs[outputs >= 0.5] = 1
    outputs[outputs < 0.5] = 0
    return targets, outputs

def postprocess_instance_segmentation(targets: torch.Tensor, outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # instance segmentation w/o background
    targets[targets >= 0.5] = 1
    targets[targets < 0.5] = 0
    outputs[outputs >= 0.5] = 1
    outputs[outputs < 0.5] = 0
    n = targets.shape[1]
    t, o = torch.zeros_like(targets[:, 0:1]), torch.zeros_like(outputs[:, 0:1])
    for i in range(n):
        target = targets[:, i : i + 1]
        output = outputs[:, i : i + 1]
        t[target == 1] = i + 1
        o[output == 1] = i + 1
    return t, o
