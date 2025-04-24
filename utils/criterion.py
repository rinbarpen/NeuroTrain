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
        return torch.stack(all_loss).sum()

class MultiLabelCombineCriterion(nn.Module):
    def __init__(self, loss_fns: nn.Module|list[nn.Module]):
        super(MultiLabelCombineCriterion, self).__init__()
        if isinstance(loss_fns, list):
            self.criterions = loss_fns
        else:
            self.criterions = [loss_fns]

    def forward(self, 
                targets: tuple[torch.Tensor, ...], 
                preds: tuple[torch.Tensor, ...]):
        all_loss = []
        for criterion in self.criterions:
            loss = []
            for target, pred in zip(targets, preds):
                loss.append(criterion(target, pred))
            all_loss.append(torch.stack(loss).mean())
        return torch.stack(all_loss).sum()

class MultiSizeCriterion(nn.Module):
    def __init__(self, loss_fn: nn.Module):
        super(MultiSizeCriterion, self).__init__()

        self.loss_fn = loss_fn
        self.scale_ratio = 2

    def forward(self, 
                targets: torch.Tensor, 
                preds: tuple[torch.Tensor, ...]):
        ts = [targets]
        for i in range(len(preds)-1):
            t = torch.max_pool2d(targets, self.scale_ratio)
            ts.append(t)

        loss = [self.loss_fn(t, pred) for t, pred in zip(ts, preds)]
        loss = torch.tensor(np.mean(loss))
        return loss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, targets: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
        loss = dice_loss(targets.detach().cpu().numpy(), preds.detach().cpu().numpy())
        x = torch.from_numpy(loss)
        x.requires_grad = True
        return x
class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, targets: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
        loss = kl_divergence_loss(targets.detach().cpu().numpy(), preds.detach().cpu().numpy())
        x = torch.from_numpy(loss)
        x.requires_grad = True
        return x
