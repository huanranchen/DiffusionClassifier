import torch
from torch import nn, Tensor
from typing import Callable


class PurificationDefense(nn.Module):
    def __init__(self, purifier: Callable, classifier: Callable, device=torch.device("cuda")):
        super(PurificationDefense, self).__init__()
        self.purifier = purifier
        self.classifier = classifier
        self.device = device
        self.eval().requires_grad_(False).to(self.device)

    def forward(self, x: Tensor) -> Tensor:
        x = self.purifier(x)
        x = self.classifier(x)
        return x
