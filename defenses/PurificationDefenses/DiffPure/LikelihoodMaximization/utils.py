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


def l2_clamp(x: Tensor, ori_x: Tensor, eps=1.0, inplace=False) -> Tensor:
    """
    Here, x should be in [-1, 1]. eps=1 actually corresponding to 0.5 in [0, 1]
    """
    B = x.shape[0]
    difference = x - ori_x
    distance = torch.norm(difference.view(B, -1), p=2, dim=1)
    mask = distance > eps
    if torch.sum(mask) > 0:
        difference[mask] = difference[mask] / distance[mask].view(torch.sum(mask), 1, 1, 1) * eps
        if inplace:
            x.mul_(0).add_(ori_x + difference)
        else:
            x = ori_x + difference
    return x
