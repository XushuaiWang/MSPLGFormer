import torch
import torch.nn as nn
from torch.nn import functional as F


def get_loss_module():
    return NoFussCrossEntropyLoss(reduction='none')  # outputs loss for each batch sample


def l2_reg_loss(model):
    """Return the squared norm of output_layer.weight when that parameter exists."""

    for name, param in model.named_parameters():
        if name == 'output_layer.weight':
            return torch.sum(torch.square(param))


class NoFussCrossEntropyLoss(nn.CrossEntropyLoss):
    """Cross-entropy wrapper that casts class-index targets to int64."""

    def forward(self, inp, target):
        return F.cross_entropy(inp, target.long(), weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)
