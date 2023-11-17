import torch
from torch import nn, Tensor
from torch.autograd import Function
from typing import Any
import torch.nn.functional as F


class HRLinearFunction(Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, weight: Tensor, bias: Tensor) -> Any:
        """
        :param x: N, D1 -> N, D2
        :param weight: D2, D1
        :param bias: D2
        :return:
        """
        ctx.weight = weight
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx: Any, dy: Tensor) -> Any:
        """
        :param dy: N, D2
        :return:
        """
        weight = ctx.weight  # D2, D1
        dx = dy @ weight  # N, D1
        return dx, None, None


class HRLinear(nn.Module):
    def __init__(self, weight: Tensor, bias: Tensor):
        super(HRLinear, self).__init__()
        self.weight = weight
        self.bias = bias
        # self.weight.requires_grad_(False)
        # self.bias.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        return HRLinearFunction.apply(x, self.weight, self.bias)
