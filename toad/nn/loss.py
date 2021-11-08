from typing import Dict, List

import torch
from torch.nn import Module

from .functional import focal_loss


class FocalLoss(Module):
    def __init__(self, alpha = 1., gamma = 2., reduction = 'mean'):
        super(FocalLoss, self).__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        return focal_loss(
            input,
            target,
            alpha = self.alpha,
            gamma = self.gamma,
            reduction = self.reduction,
        )


class DictLoss(Module):
    def __init__(self, torch_loss, weights: Dict[str, float] = None):
        super(DictLoss, self).__init__()
        self.torch_loss = torch_loss
        self.weights = weights or {}

    def forward(self, input: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]):
        loss = 0
        weight_sum = 0
        for key, _target in target.items():
            if key not in input:
                continue
            weight = self.weights.get(key, 1)
            mask = torch.bitwise_not(torch.isnan(_target))
            _target = _target.to(input[key].device)
            loss += weight * self.torch_loss(input[key][mask], _target[mask])
            weight_sum += weight

        return loss / weight_sum


class ListLoss(Module):
    def __init__(self, torch_loss, weights: List[float] = None):
        super(ListLoss, self).__init__()
        self.torch_loss = torch_loss
        self.weights = weights

    def forward(self, input: List[torch.Tensor], target: List[torch.Tensor]):
        loss = 0
        weight_sum = 0
        for i, (_input, _target) in enumerate(zip(input, target)):
            if self.weights:
                weight = self.weights[i]
            else:
                weight = 1
            _target = _target.to(_input.device)
            mask = torch.bitwise_not(torch.isnan(_target))
            loss += weight * self.torch_loss(_input[mask], _target[mask])
            weight_sum += weight

        return loss / weight_sum
