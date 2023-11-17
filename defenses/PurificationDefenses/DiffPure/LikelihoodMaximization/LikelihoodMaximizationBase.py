from torch import nn
import torch


class LikelihoodMaximizationBase(nn.Module):
    def __init__(self, dc: nn.Module, device=torch.device("cuda"), transform=lambda x: (x - 0.5) * 2):
        super().__init__()
        self.dc = dc
        self.device = device
        self.transform = transform
        self.eval().requires_grad_(False).to(self.device)

    def optimize_back(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        return self.optimize_back(*args, **kwargs)
