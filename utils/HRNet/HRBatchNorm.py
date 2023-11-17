import torch
from torch import nn, Tensor
from torch.autograd import Function
from typing import Any, Optional
from torch.nn import functional as F


class HRBatchNorm2DFunction(Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, weight: Tensor, bias: Tensor) -> Any:
        """
        :param ctx:
        :param x:  N, C, H, D
        :param weight: C
        :param bias: C
        :return:
        """
        ctx.weight = weight
        x = x.permute(0, 2, 3, 1)  # N, H, D, C
        x = x * weight + bias
        x = x.permute(0, 3, 1, 2)  # N, C, H, D
        return x

    @staticmethod
    def backward(ctx: Any, dy) -> Any:
        """
        :param ctx:
        :param dy: N, C, H, D
        :return:
        """
        dy = dy.permute(0, 2, 3, 1)  # N, H, D, C
        weight = ctx.weight
        dx = dy * weight
        dx = dx.permute(0, 3, 1, 2)
        return dx, None, None


class HRBatchNorm2D(nn.Module):
    def __init__(self, mean: Tensor, std: Tensor, weight: Tensor, bias: Tensor):
        """
        y = weight*(x-mean)/std+bias = weight/std * x + bias - weight/std * mean
        :param mean:C
        :param std:C
        :param weight:C
        :param bias:C
        """
        super(HRBatchNorm2D, self).__init__()
        self.weight = weight / std
        self.bias = bias - self.weight * mean

    def forward(self, x: Tensor):
        """
        :param x: N, C, H, D
        """
        return HRBatchNorm2DFunction.apply(x, self.weight, self.bias)
