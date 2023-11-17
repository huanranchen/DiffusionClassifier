"""
Similarity of Neural Network Representations Revisited
"""
from torch import nn, Tensor
import torch


class UnNormalizedSimpleCKA(nn.Module):
    def __init__(self):
        super(UnNormalizedSimpleCKA, self).__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        x, y should be (N, D) tensor
        """
        assert x.shape == y.shape, 'feature shape should be equal if you use CKA'
        return torch.sum((x.T @ y) ** 2)


class NormalizedSimpleCKA(nn.Module):
    def __init__(self):
        super(NormalizedSimpleCKA, self).__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        x, y should be (N, D) tensor
        """
        assert x.shape == y.shape, 'feature shape should be equal if you use CKA'
        x_f, y_f = torch.norm(x @ x.T, p='fro'), torch.norm(y @ y.T, p='fro')
        return torch.sum((x.T @ y) ** 2) / (x_f * y_f)


class CenteredKernelAnalysis(nn.Module):
    def __init__(self, kernel_function):
        """
        :param kernel_function: given a matrix (N, D), return a matrix K with shape (N, N),
                                where K_{ij} = kernel(x_i, x_j)
        """
        super(CenteredKernelAnalysis, self).__init__()
        self.kernel_function = kernel_function

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        x, y should be (N, D) tensor
        """
        assert x.shape == y.shape, 'feature shape should be equal if you use CKA'
        k_x, k_y = self.kernel_function(x), self.kernel_function(y)
        x_f, y_f = torch.norm(k_x, p='fro'), torch.norm(k_y, p='fro')
        return (k_x.view(-1) @ k_y.view(-1)) / (x_f * y_f)
