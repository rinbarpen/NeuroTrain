from torch import nn
import torch
from utils.scores import kl_divergence_loss, dice_loss

class CombineCriterion(nn.Module):
    def __init__(self, loss_fns: nn.Module|list[nn.Module]):
        super(CombineCriterion, self).__init__()
        if isinstance(loss_fns, list):
            self.criterions = loss_fns
        else:
            self.criterions = [loss_fns]

    def forward(self, targets: torch.Tensor, preds: torch.Tensor):
        all_loss = []
        for criterion in self.criterions:
            loss = criterion(targets, preds)
            all_loss.append(loss)
        return all_loss

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
