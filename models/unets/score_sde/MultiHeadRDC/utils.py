import torch
from torch import Tensor
from typing import List


def clamp(x, min=0, max=1):
    return torch.clamp(x, min=min, max=max)


def inplace_clamp(x, min=0, max=1):
    return torch.clamp(x, min=min, max=max)


def L2Loss(x: Tensor, y: Tensor) -> Tensor:
    """
    :param x: N, C, H, D
    :param y: N, C, H, D
    :return: dim=0 tensor
    """
    x = (x - y) ** 2
    x = x.view(x.shape[0], -1)
    x = torch.norm(x, dim=1, p=2)
    x = x.mean(0)
    return x


def abs_loss(x: Tensor, y: Tensor) -> Tensor:
    diff = torch.abs(x - y)
    diff = diff.view(diff.shape[0], -1)
    diff = torch.sum(diff, dim=1)
    diff = torch.mean(diff, dim=0)
    return diff


def L2_each_instance(x: Tensor, y: Tensor) -> Tensor:
    """
    N, ?, ?, ?, ...
    :return:  N,
    """
    x = (x - y) ** 2
    x = x.view(x.shape[0], -1)
    x = torch.norm(x, dim=1, p=2)
    return x


def list_mean(x: List) -> float:
    return sum(x) / len(x)
