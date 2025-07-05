import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def dice_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    *,
    class_axis: int = 1,
    eps: float = 1e-6,
) -> torch.Tensor:
    # 确保 class_axis 是正数索引
    if class_axis < 0:
        class_axis += y_true.ndim

    reduction_dims = tuple(i for i in range(y_true.ndim) if i != class_axis)

    intersection = torch.sum(y_true * y_pred, dim=reduction_dims)
    union = torch.sum(y_true, dim=reduction_dims) + torch.sum(y_pred, dim=reduction_dims)

    dice_scores = (2.0 * intersection + eps) / (union + eps)

    mean_dice_score = dice_scores.mean()

    return 1.0 - mean_dice_score

def kl_divergence_loss(
    y_true: torch.Tensor, y_pred: torch.Tensor, *, epsilon: float = 1e-7
):
    # 确保输入为概率分布
    y_true = torch.clamp(y_true, epsilon, 1.0 - epsilon)
    y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)

    # 计算 KL 散度: sum(p * log(p/q))
    kl_div = torch.sum(y_true * torch.log(y_true / y_pred), dim=1)
    return torch.mean(kl_div)


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float=1.0, reduction: str='batchmean'):
    tearcher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()
    # probs曲线更加平滑
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1).detach()
    # kl_div的学生输入必须是log的，而教师的不需要log
    kl = F.kl_div(
        student_log_probs,
        tearcher_probs,
        reduction=reduction
    )
    # 让loss曲线更加突出
    return (temperature ** 2) * kl


class CombineCriterion(nn.Module):
    def __init__(self, loss_fns: nn.Module|list[nn.Module]):
        super(CombineCriterion, self).__init__()
        if isinstance(loss_fns, list):
            self.criterions = loss_fns
        else:
            self.criterions = [loss_fns]

    def forward(self, targets, preds):
        all_loss = []
        for criterion in self.criterions:
            loss = criterion(targets, preds)
            all_loss.append(loss)
        return all_loss

class Loss(nn.Module):
    def __init__(self, loss_fn: nn.Module|None=None, weight: float=1.0):
        super(Loss, self).__init__()
        self.loss_fn = loss_fn
        self.weight = weight
    
    def forward(self, targets: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
        if self.loss_fn is None:
            raise NotImplementedError("Loss function is not defined.")
        return self.loss_fn(targets, preds) * self.weight


class DiceLoss(Loss):
    def __init__(self, weight: float=1.0):
        super(DiceLoss, self).__init__(weight=weight)

    def forward(self, targets: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
        loss = dice_loss(targets, preds)
        x = torch.tensor(loss)
        x.requires_grad = True
        return x * self.weight

class KLLoss(Loss):
    def __init__(self, weight: float=1.0):
        super(KLLoss, self).__init__(weight=weight)

    def forward(self, targets: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
        loss = kl_divergence_loss(targets, preds)
        return loss * self.weight

class DistillationLoss(Loss):
    def __init__(self, temperature: float=1.0, weight: float=1.0):
        super(DistillationLoss, self).__init__(weight=weight)
        self.temperature = temperature

    def forward(self, targets: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
        loss = distillation_loss(
            student_logits=preds,
            teacher_logits=targets,
            temperature=self.temperature
        )
        return loss * self.weight