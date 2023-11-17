import torch
from torch.optim import Optimizer
from typing import Callable, List, Iterable
from attacks.utils import *
from .utils import *


class Perturbation():
    def __init__(self,
                 optimizer: Optimizer or Callable,
                 perturbation_size: tuple = (3, 299, 299),
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 epsilon=16 / 255, ):
        self.perturbation = torch.zeros(perturbation_size, device=device)
        self.device = device
        self.optimizer = optimizer
        self.eps = epsilon

    def gaussian_init(self, is_clamp=True, scale=0.5, mean=0.5):
        self.perturbation = torch.randn_like(self.perturbation, device=self.device) * scale + mean
        if is_clamp:
            self.perturbation = clamp(self.perturbation)
        self.optimizer = self.optimizer([self.perturbation])

    def uniform_init(self):
        self.perturbation = torch.rand_like(self.perturbation, device=self.device)
        self.optimizer = self.optimizer([self.perturbation])

    def constant_init(self, constant=0):
        self.perturbation = torch.zeros_like(self.perturbation, device=self.device) + constant
        self.optimizer = self.optimizer([self.perturbation])

    def requires_grad(self, requires_grad: bool = True):
        self.perturbation.requires_grad = requires_grad

    @torch.no_grad()
    def step(self):
        self.optimizer.step()
        self.perturbation.clamp_(min=-self.eps, max=self.eps)

    def clamp(self):
        self.perturbation.clamp_(min=-self.eps, max=self.eps)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def grad(self):
        return self.perturbation.grad

    def assign_grad(self, grad):
        self.perturbation.grad = grad
