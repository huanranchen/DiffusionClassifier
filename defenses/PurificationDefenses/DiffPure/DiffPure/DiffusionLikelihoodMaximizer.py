from torch import nn, Tensor
from torch.autograd import Function

from defenses.PurificationDefenses.PurificationDefense import PurificationDefense
from typing import Callable


class DiffusionLikelihoodMaximizerFunction(Function):
    function = None

    @staticmethod
    def forward(ctx, x: Tensor):
        x = DiffusionLikelihoodMaximizerFunction.function(x.detach().clone())
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad


def diffusion_likelihood_maximizer_defense(
    classifier: nn.Module,
    likelihood_maximization: Callable,
):
    DiffusionLikelihoodMaximizerFunction.function = likelihood_maximization

    class Purifier(nn.Module):
        def __init__(self):
            super(Purifier, self).__init__()
            self.requires_grad_(False).eval().cuda()

        def forward(self, x):
            return DiffusionLikelihoodMaximizerFunction.apply(x)

    return PurificationDefense(Purifier(), classifier)
