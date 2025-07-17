import torch
from torch import nn
import torch.nn.functional as F

def dice_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor, # y_pred 预期为 logits
    *,
    class_axis: int = 1,
    smooth: float = 1e-6,
    # 新增参数，指定是否在内部进行激活
    apply_activation: bool = True,
    activation_type: str = 'sigmoid', # 'sigmoid' for binary/multilabel, 'softmax' for multiclass
) -> torch.Tensor:
    if class_axis < 0:
        class_axis += y_true.ndim

    # 对预测值应用激活函数
    if apply_activation:
        if activation_type == 'sigmoid':
            y_pred = torch.sigmoid(y_pred)
        elif activation_type == 'softmax':
            # 对于 softmax，需要知道类别维度
            y_pred = torch.softmax(y_pred, dim=class_axis)
        else:
            raise ValueError(f"Unsupported activation_type: {activation_type}")

    reduction_dims = tuple(i for i in range(y_true.ndim) if i != class_axis)

    intersection = (y_true * y_pred).sum(dim=reduction_dims)
    union = (y_true + y_pred).sum(dim=reduction_dims)
    union = torch.where(union == 0, intersection, union)

    dice_scores = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice_scores.mean()

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
    def __init__(self, *loss_fns):
        super(CombineCriterion, self).__init__()
        self.loss_fns = loss_fns

    def forward(self, targets: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
        return torch.stack([loss_fn(targets, preds) for loss_fn in self.loss_fns]).sum()

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
    def __init__(self, weight: float=1.0, activation_type: str='sigmoid'): # 增加 activation_type 参数
        super(DiceLoss, self).__init__(weight=weight)
        self.activation_type = activation_type

    def forward(self, targets: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
        loss = dice_loss(targets, preds, apply_activation=True, activation_type=self.activation_type)
        return loss * self.weight

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
    


def get_criterion(c: dict):
    c_type = c['type'].lower()
    weight = c.get('weight', 1)
    cc = c['config']

    if 'dice' in c_type:
        return DiceLoss(weight, **cc)
    elif 'bce' in c_type:
        return Loss(nn.BCEWithLogitsLoss(**cc), weight)
    elif 'ce' in c_type:
        return Loss(nn.CrossEntropyLoss(**cc), weight)

    return None
