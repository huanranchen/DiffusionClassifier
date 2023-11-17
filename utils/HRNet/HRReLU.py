import torch
from torch import nn, Tensor
from torch.autograd import Function
from typing import Any


class HRReLUFunction(Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor) -> Tensor:
        mask = x < 0
        ctx.mask = mask
        x[mask] = 0
        return x

    @staticmethod
    def backward(ctx: Any, dy: Tensor) -> Tensor:
        mask = ctx.mask
        grad = dy.clone()
        grad[mask] = 0
        return grad


class HRReLU(nn.Module):
    def __init__(self):
        super(HRReLU, self).__init__()

    def forward(self, x: Tensor):
        return HRReLUFunction.apply(x)
