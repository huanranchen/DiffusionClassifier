import torch
from torch import nn, Tensor
from torch.autograd import Function
from typing import Any, Optional
from torch.nn import functional as F
import torchvision


"""

"""


class HRConvFunction(Function):
    @staticmethod
    def forward(ctx: Any, input: Tensor, weight: Tensor, bias: Optional[Tensor],
                stride: int, padding: int = 0, dilation: bool = False, groups: int = 1) -> Any:
        ctx.weight = weight
        return F.conv2d(input, weight, bias, stride,
                        padding, dilation, groups)

    @staticmethod
    def backward(ctx: Any, dy) -> Any:
        weight = ctx.weight


class HRConv(nn.Module):
    """
    padding only support zero padding!!
    not support dilation convolution!!!!!!

    """

    def __init__(self, weight: Tensor, bias: Optional[Tensor],
                stride: int, padding: int = 0, dilation: bool = False, groups: int = 1):
        super(HRConv, self).__init__()

    def forward(self, input: Tensor):
        pass