import torch
from torch import nn
import torch.nn.functional as F
from utils.scores import kl_divergence_loss, dice_loss
import numpy as np

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
        return self.loss_fn(targets, preds) * self.weight


class DiceLoss(Loss):
    def __init__(self, weight: float=1.0):
        super(DiceLoss, self).__init__(weight=weight)

    def forward(self, targets: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
        loss = dice_loss(targets.detach().cpu().numpy(), preds.detach().cpu().numpy())
        x = torch.tensor(loss)
        x.requires_grad = True
        return x * self.weight

class KLLoss(Loss):
    def __init__(self, weight: float=1.0):
        super(KLLoss, self).__init__(weight=weight)

    def forward(self, targets: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
        loss = kl_divergence_loss(targets.detach().cpu().numpy(), preds.detach().cpu().numpy())
        x = torch.from_numpy(loss)
        x.requires_grad = True
        return x * self.weight
