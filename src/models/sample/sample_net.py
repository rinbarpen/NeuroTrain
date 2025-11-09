import torch
import torch.nn as nn
import torch.nn.functional as F

class SampleNet(nn.Module):
    def __init__(self):
        super(SampleNet, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, batch_inputs):
        images = batch_inputs['images']
        targets = batch_inputs['targets']
        preds = self.conv(images)
        loss = F.cross_entropy(preds, targets.squeeze(-1))
        return {
            'targets': targets,
            'preds': preds,
            'loss': loss,
        }