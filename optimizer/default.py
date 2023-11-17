import torch
from torch import nn
from .ALRS import ALRS


def default_optimizer(model: nn.Module, lr=1e-1, ) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=lr)


def default_lr_scheduler(optimizer):
    return ALRS(optimizer)
